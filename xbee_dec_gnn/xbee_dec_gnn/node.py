#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import socket
import threading
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from prettytable import PrettyTable

from xbee_dec_gnn.decentralized_gnns.dec_gnn import DecentralizedGNN
from xbee_dec_gnn.encoder import pack_tensor, unpack_tensor
from xbee_dec_gnn.utils import LEDMatrix, ObjectWithLogger
from xbee_dec_gnn.utils.zigbee_comm import (
    DataExchangeMessage,
    GraphMessage,
    Topic,
    ZigbeeNodeInterface,
)


class Node(ObjectWithLogger):
    def __init__(self, params):
        super().__init__(logger_name="xbee_dec_gnn.node")

        # Unique identifier for the node. # DOC: We assume all nodes have the same format of the name.
        self.node_prefix = "node_"
        self.node_id = params.node_id or socket.gethostname()[-1]
        try:
            self.node_id = int(self.node_id)
        except (TypeError, ValueError):
            pass
        self.node_name = self.node_prefix + str(self.node_id)
        self.starting_data = None

        self.get_logger().info(f"Node online: {self.node_name}")

        self.layer = 0  # The current layer of the GNN being processed.
        self.received_mp = defaultdict(dict)  # Message passing values received from neighbors, indexed by iteration
                                              # number. Also used for synchronization.
        self.received_pooling = defaultdict(dict)  # Pooling values received from neighbors, indexed by iteration number

        self.round_counter = 0
        self.local_subgraph = nx.Graph()
        self.active_neighbors = []

        self.stats = {"inference_time": [], "message_passing_time": [], "pooling_time": [], "round_time": []}

        # Load parameters
        self.num_nodes = params.num_nodes
        self.gnn_model_path = params.model_path

        # Load the GNN model. # DOC: Change this to customize how the model is loaded.
        dist_model_kwargs = dict(pooling_protocol="consensus", consensus_sigma=1 / self.num_nodes)
        self.decentralized_model = DecentralizedGNN.from_simple_gnn_wrapper(self.gnn_model_path, **dist_model_kwargs)

        # Initialize the LED matrix if available.
        self.led = LEDMatrix()

        self.zigbee = ZigbeeNodeInterface(
            params.port,
            params.baud,
            self.node_name,
            logger=self.get_logger(),
        )
        self.graph_lock = threading.Event()

        # Publisher dictionaries (initialized after handshake completes)
        self.mp_pubs = {}
        self.pooling_pubs = {}

        # Register message handlers (only application-level messages)
        self.zigbee.register_handler(Topic.GRAPH, self._handle_graph)
        self.zigbee.register_handler(Topic.MP, self._handle_mp)
        self.zigbee.register_handler(Topic.POOLING, self._handle_pooling)

    def _init_neighbor_publishers(self):
        """
        Initialize publishers for message passing and pooling to all neighbors.
        Called after handshake is complete.
        """
        for neighbor_id in range(self.num_nodes):
            if neighbor_id == int(self.node_id):
                continue  # Skip self

            neighbor_name = f"{self.node_prefix}{neighbor_id}"

            # Create message passing publisher (ZigbeeInterface handles address resolution)
            self.mp_pubs[neighbor_name] = self.zigbee.create_publisher(
                target_name=neighbor_name,
                topic=Topic.MP,
            )

            # Create pooling publisher
            self.pooling_pubs[neighbor_name] = self.zigbee.create_publisher(
                target_name=neighbor_name,
                topic=Topic.POOLING,
            )

        self.get_logger().info(f"Initialized publishers for {len(self.mp_pubs)} nodes.")


    def _handle_graph(self, msg: GraphMessage):
        """Handle GRAPH message with topology and features."""
        # Use provided node_id, fallback to self.node_id
        self.get_logger().info(f"Graph received: neighbors={msg.neighbors}")

        self.local_subgraph = nx.Graph()
        self.local_subgraph.add_node(self.node_id)
        for nb in msg.neighbors:
            self.local_subgraph.add_edge(self.node_id, nb)

        if msg.features is None:
            raise ValueError("Missing node features in GRAPH message")

        x_tensor = torch.tensor(msg.features, dtype=torch.float32)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)

        self.starting_data = x_tensor
        self.graph_lock.set()

    def _handle_mp(self, msg: DataExchangeMessage):
        """Handle message passing (MP) message."""
        round = msg.round_id
        layer = msg.layer
        sender = msg.sender_name

        if msg.data is None or msg.shape is None:
            self.get_logger().warning("MP message missing data or shape")
            return

        tensor_data = unpack_tensor(msg.data, msg.shape)
        self.received_mp[(round, layer)][sender] = tensor_data

        self.get_logger().debug(
            f"RX: MP stored from node {sender} (r={round}, l={layer}) "
            f"now {len(self.received_mp[(round, layer)])}/{len(self.active_neighbors)}"
        )

    def _handle_pooling(self, msg: DataExchangeMessage):
        """Handle pooling message."""
        sources = msg.sources or []
        if msg.data is None or msg.shape is None:
            self.get_logger().warning("Pooling message missing data or shape")
            return

        data = unpack_tensor(msg.data, msg.shape)
        if len(sources) > 0:
            tensor_data = {source: data[i] for i, source in enumerate(sources)}
        else:
            tensor_data = data
        self.received_pooling[msg.iteration][msg.sender_name] = tensor_data

    def run(self):
        # Main loop of the node.
        next_call = time.perf_counter()

        while True:
            self.compute_gnn()

            next_call += 0.5
            sleep_time = next_call - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_call = time.perf_counter()  # If we're behind schedule, skip sleeping to catch up.

    def start(self):
        self.zigbee.start()

        # Wait for handshake to complete (handled internally by ZigbeeInterface)
        if not self.zigbee.wait_for_handshake(timeout=30.0):
            raise RuntimeError("Handshake failed or timed out")

        # Initialize publishers to neighbors
        self._init_neighbor_publishers()

        self.get_logger().info("Setup complete: graph/features loaded; starting GNN loop")

    def get_neighbors(self):
        # Still waiting for the first graph message to be received and processed.
        if not self.graph_lock.is_set():
            return False

        self.active_neighbors = []

        # We know the whole graph in development mode.
        if self.local_subgraph.number_of_nodes() > 0:
            self.active_neighbors = list(self.local_subgraph.neighbors(self.node_id))

        if not nx.is_connected(self.local_subgraph):
            self.get_logger().fatal(
                "Local subgraph is not connected! This should never happen. If it does, it is probably due to a "
                "mismatch of communication radius used in this node and for the creation of the discovery messages."
            )
            raise RuntimeError("Local subgraph is not connected.")
        ready = len(self.active_neighbors) > 0 and self.starting_data is not None
        if ready:
            self.get_logger().debug(f"Neighbors: {self.active_neighbors}")

        return ready

    def get_initial_features(self):
        # Compute the initial feature vector for this node (already received over XBee).
        if self.starting_data is None:
            self.get_logger().error("Initial features not loaded! This should never happen if the graph message is properly received and processed.")
            raise RuntimeError("Initial features not loaded.")

        return self.starting_data

    def run_message_passing(self, initial_features: torch.Tensor):
        inference_time = 0.0

        mp_start = time.perf_counter()
        node_value = initial_features
        for layer in range(self.decentralized_model.num_layers):
            # Send the current node representation to neighbors.
            self.send_message_passing(layer, node_value)

            # Wait until values are received from all neighbors.
            wait_time_start = time.time()
            key = (self.round_counter, layer)
            while len(self.received_mp[key]) < len(self.active_neighbors):
                if time.time() - wait_time_start > 30:  # 30 seconds timeout
                    raise TimeoutError("Timeout waiting for message passing messages.")
                # self.get_logger().debug(
                #     f"Waiting for MP layer {layer}: "
                #     f"{len(self.received_mp[layer])}/{len(self.active_neighbors)} received"
                # )
                time.sleep(0.1)

            # Update the node's representation using the GNN layer.
            neighbor_values = list(self.received_mp[key].values())
            inference_start = time.perf_counter()
            node_value = self.decentralized_model.update_gnn(layer, node_value, neighbor_values)
            inference_time += time.perf_counter() - inference_start
            del self.received_mp[key]

            self.get_logger().debug(f"MP layer {layer} complete")

        self.stats["inference_time"].append(inference_time)
        self.stats["message_passing_time"].append(time.perf_counter() - mp_start)
        self.get_logger().info(
            f"  Message passing complete ({time.perf_counter() - mp_start:.2f}s)"
        )

        return node_value

    def run_pooling(self, init_node_value: torch.Tensor) -> torch.Tensor:
        pooling_start = time.perf_counter()

        if self.decentralized_model.pooling is None:
            return init_node_value

        node_value = self.decentralized_model.init_pooling(init_node_value)
        for iteration in range(self.num_nodes - 1 + self.decentralized_model.pooling.convergence_min):
            # Send the current node representation to neighbors.
            self.send_pooling(iteration, node_value)

            # Wait until values are received from all neighbors.
            wait_time_start = time.time()
            while len(self.received_pooling[iteration]) < len(self.active_neighbors):
                if time.time() - wait_time_start > 30:  # 30 seconds timeout
                    raise TimeoutError("Timeout waiting for pooling messages.")
                time.sleep(0.1)

            # Update the pooled value at current iteration.
            node_value, final_value, error = self.decentralized_model.update_pooling(
                node_value, list(self.received_pooling[iteration].values())
            )
            del self.received_pooling[iteration]

            self.get_logger().debug(
                f"Pooling iteration {iteration} complete (error={error:.6f})"
            )

        if final_value is None:
            raise RuntimeError("Pooling did not converge.")

        self.stats["pooling_time"].append(time.perf_counter() - pooling_start)
        self.get_logger().info(
            f"  Pooling complete ({time.perf_counter() - pooling_start:.2f}s)"
        )
        return final_value

    def run_prediction(self, graph_value: torch.Tensor):
        with torch.no_grad():
            graph_value = self.decentralized_model.predictor_model(graph_value)
        return graph_value

    def send_message_passing(self, layer: int, value: torch.Tensor):
        """
        Send message passing values to all active neighbors.

        Args:
            layer: GNN layer index
            value: node representation tensor
        """
        data, shape = pack_tensor(value)
        msg = DataExchangeMessage.for_message_passing(
            sender_name=self.node_name,
            layer=layer,
            round_id=self.round_counter,
            data=data,
            shape=shape,
        )

        for neighbor in self.active_neighbors:
            self.mp_pubs[neighbor].publish(msg, add_random_delay=True)

    def send_pooling(self, iteration: int, value: dict[str, torch.Tensor] | torch.Tensor):
        """
        Send pooling values to all active neighbors.

        Args:
            iteration: pooling iteration index
            value: pooling value (single tensor or dict of tensors)
        """
        if isinstance(value, dict):  # This enables pooling by flooding
            data = torch.stack(list(value.values()), dim=0)
            payload, shape = pack_tensor(data)
            msg = DataExchangeMessage.for_pooling(
                sender_name=self.node_name,
                iteration=iteration,
                data=payload,
                shape=shape,
                sources=list(value.keys()),
            )
        else:
            payload, shape = pack_tensor(value)
            msg = DataExchangeMessage.for_pooling(
                sender_name=self.node_name,
                iteration=iteration,
                data=payload,
                shape=shape,
            )

        for neighbor in self.active_neighbors:
            self.pooling_pubs[neighbor].publish(msg, add_random_delay=False)

    def compute_gnn(self):
        ready = self.get_neighbors()
        if not ready:
            self.get_logger().info("Waiting for neighbors...")
            time.sleep(1.0)
            return

        self.round_counter += 1
        self.get_logger().info("─" * 40)
        self.get_logger().info(f"ROUND {self.round_counter}")
        round_start = time.perf_counter()

        initial_features = self.get_initial_features()

        try:
            node_value = self.run_message_passing(initial_features)
            graph_value = self.run_pooling(node_value)
            graph_value = self.run_prediction(graph_value)
            graph_value = graph_value.item()
        except TimeoutError as e:
            self.get_logger().warning(f"Timeout: {e}")
            return

        elapsed = time.perf_counter() - round_start
        self.stats["round_time"].append(elapsed)
        self.get_logger().info(
            f"ROUND {self.round_counter} DONE: value={graph_value:.4f} ({elapsed:.2f}s)"
        )

        if graph_value < 0.5:
            mids = "not "
            color = (50, 50, 50)
        else:
            mids = ""
            color = (50, 0, 0)
        self.get_logger().info(f"The node is \033[7m{mids}in MIDS\033[0m.\n")
        self.led.set_all(color)

    def print_stats(self):
        table = PrettyTable()
        table.add_column("", ["Mean", "Std", "Max", "Min"])
        for key, values in self.stats.items():
            table.add_column(
                key,
                [f"{np.mean(values):.4f}", f"{np.std(values):.4f}", f"{np.max(values):.4f}", f"{np.min(values):.4f}"],
            )
        self.get_logger().info(f"\n{table}")

    def stop(self):
        self.zigbee.device.close()
        self.get_logger().info("Node stopped.")
        # self.print_stats()
        self.led.exit()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--model-path", default="/root/resources/models/MIDS_model.pth")
    parser.add_argument("--num-nodes", type=int, default=5)
    parser.add_argument("--node-id", type=str, default=None, help="Unique identifier for this node")
    cli_args = parser.parse_args(args=args)

    gnn_node = Node(params=cli_args)
    gnn_node.start()
    try:
        gnn_node.run()
    except KeyboardInterrupt:
        gnn_node.stop()


if __name__ == "__main__":
    main()
