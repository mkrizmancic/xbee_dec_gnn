#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import random
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from torch import Tensor
import torch_geometric.utils as tg_utils
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
from torch_geometric.data import Data, InMemoryDataset
from xbee_dec_gnn.utils import ObjectWithLogger
from xbee_dec_gnn.utils.zigbee_comm import GraphMessage, Topic, ZigbeeCentralInterface


class GraphGenerator(ObjectWithLogger):
    def __init__(self,
        graph_mode="load",
        gui_mode=False,
        port="/dev/ttyUSB0",
        baud_rate=9600,
        num_nodes=5,
    ):

        super().__init__(logger_name="central")

        self.graph_mode = graph_mode
        self.gui_mode = gui_mode
        self.num_nodes = num_nodes
        self.node_prefix = "node_" # TODO

        # Load graph dataset and and check it has required attributes
        self.dataset = InMemoryDataset()
        self.dataset.load("/root/resources/data/MIDS_data.pt")

        # Select appropriate size range
        dataset_range = {}
        curr_index = 0
        for size, num_graphs in zip(range(3, 9), [2, 6, 21, 112, 853, 11117]):
            dataset_range[size] = (curr_index, curr_index + num_graphs - 1)
            curr_index += num_graphs
        self.dataset_range = dataset_range[self.num_nodes]
        self.current_graph_index = 0
        self.data: Data = self.dataset[self.current_graph_index]  # type: ignore

        self.G = nx.cycle_graph(self.num_nodes)
        self.node_positions = {}

        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Dynamic Graph Visualization")
        self.ax.format_coord = lambda x, y: ''

        # Create button for GUI mode
        if self.gui_mode:
            self.button_ax = self.fig.add_axes([0.02, 0.02, 0.12, 0.05])  # type: ignore
            self.next_button = Button(self.button_ax, 'Next')
            self.next_button.on_clicked(self.on_next_button_clicked)

        self.prev_image = None
        self.ax_range = None

        self.zigbee = ZigbeeCentralInterface(
            port=port,
            baud_rate=baud_rate,
            num_nodes=self.num_nodes,
            logger=self.get_logger(),
        )

        self.initialize_zigbee()

        # In GUI mode, generate first graph immediately and wait for button clicks
        # In non-GUI mode, use timer with input() blocking
        if self.gui_mode:
            self.process_next_graph()
        else:
            self._timer_thread = threading.Thread(
                target=self._timer_loop,
                daemon=True
            )
            self._timer_thread.start()

            # Non-GUI mode for manual stepping not implemented yet.

    def initialize_zigbee(self):
        self.zigbee.start()
        self.zigbee.run_handshake()

        self.addr_map = self.zigbee.addr_map

    def send_node_info(self):
        """
        Send per-node neighborhood + per-node feature vector only.
        Designed to stay under XBee's ~255 byte payload; logs payload size so
        you can tune feature_dim.
        """
        assert self.data is not None and self.data.x is not None, "Graph data or features not loaded."

        for node_name in self.addr_map:
            idx = int(node_name.split("_")[-1])  # Assuming node_name format is "node_X" # TODO

            # Only use neighbors that are in our active node set
            all_neighbors = list(self.G.neighbors(idx))
            neighbors = [f"{self.node_prefix}{nbr}" for nbr in all_neighbors]

            x_i = self.data.x[idx]  # shape: (F,)
            x_list = x_i.tolist()

            msg = GraphMessage(
                neighbors=neighbors,
                features=x_list,
            )

            payload_bytes = len(msg.to_bytes())
            self.get_logger().debug(
                f"GRAPH payload size -> node_name={node_name} bytes={payload_bytes} "
                f"(features={len(x_list)}, neighbors={len(neighbors)})"
            )
            if payload_bytes > 255:
                self.get_logger().warning(
                    f"GRAPH payload for node_name={node_name} is {payload_bytes} bytes (>255). "
                    "Reduce feature_dim or pruning."
                )

            self.zigbee.send_to_node(node_name, Topic.GRAPH, msg, add_random_delay=False)


    def process_next_graph(self):
        """Generate/load and publish the next graph."""
        if self.graph_mode == "generate":
            self.generate_graph()
        elif self.graph_mode == "load":
            self.load_next_graph()
        else:
            self.get_logger().error(f"Unknown graph mode: {self.graph_mode}")
            return

        self.publish_graph_image() # changed for xbee

        self.send_node_info()

        # Update GUI display if in GUI mode
        if self.gui_mode:
            # Reposition button to fixed pixel location after figure resize
            self.reposition_button()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def graph_timer_cb(self):
        """Timer callback for non-GUI mode."""
        self.process_next_graph()
        input("Ready for next graph?")

    def on_next_button_clicked(self, event):
        """Callback when Next button is clicked in GUI mode."""
        self.process_next_graph()

    def reposition_button(self):
        """Reposition button to fixed pixel location after figure resize."""
        # Get current figure size in pixels
        fig_width_px = self.fig.get_figwidth() * self.fig.dpi
        fig_height_px = self.fig.get_figheight() * self.fig.dpi

        # Button dimensions in pixels
        button_width_px = 100
        button_height_px = 40
        button_x_px = 10  # 10 pixels from left
        button_y_px = 10  # 10 pixels from bottom

        # Convert to figure coordinates (0-1)
        button_left = button_x_px / fig_width_px
        button_bottom = button_y_px / fig_height_px
        button_width = button_width_px / fig_width_px
        button_height = button_height_px / fig_height_px

        # Update button position
        self.button_ax.set_position([button_left, button_bottom, button_width, button_height])

    def generate_graph(self):
        graph_ok = False
        while not graph_ok:
            if random.random() < nx.density(self.G):
                # When number of possible edges is large, we are more likely to
                # enter this branch and we should remove an edge.
                edge_to_remove = random.choice(list(self.G.edges))
                self.G.remove_edge(*edge_to_remove)
                graph_ok = nx.is_connected(self.G)
                if not graph_ok:
                    self.G.add_edge(*edge_to_remove)

            else:
                # When number of possible edges is small, we are more likely to
                # enter this branch and we should add an edge.
                non_edges = list(nx.non_edges(self.G))
                if non_edges:  # Check if there are any non-edges to add
                    self.G.add_edge(*random.choice(non_edges))
                graph_ok = True

    def _timer_loop(self):
        while True:
            time.sleep(0.5)
            self.graph_timer_cb()

    def load_next_graph(self):
        """Load the next graph from the dataset."""
        self.current_graph_index = random.randint(self.dataset_range[0], self.dataset_range[1])
        self.data = self.dataset[self.current_graph_index]  # type: ignore
        # Create graph based on positions and communication radius
        self.G = tg_utils.to_networkx(self.data, to_undirected=True)

    def publish_graph_image(self):
        """Draw/update the graph in the matplotlib window (no ROS publishing)."""

        assert self.data is not None
        assert self.data.x is not None and isinstance(self.data.y, Tensor)

        # Clear the previous plot and draw the updated graph
        self.ax.clear()
        self.ax.set_title(f"Graph index: {self.current_graph_index}")
        self.ax.format_coord = lambda x, y: ''

        cmap = LinearSegmentedColormap.from_list('simple', ['white', 'red'])

        # Use loaded positions if in load mode, otherwise use circular layout
        if self.graph_mode == "load" and self.node_positions:
            # Convert string keys to integers and use loaded positions
            pos = {int(node_id): position for node_id, position in self.node_positions.items()}
        else:
            # Use a fixed layout for consistent positioning
            pos = nx.circular_layout(self.G)

        offset_x = max(p[0] for p in pos.values()) - min(p[0] for p in pos.values()) + 0.5

        # NOTE: i is used later for figure size, so keep it initialized.
        i = 0
        for i in range(self.data.y.shape[1]):
            if self.data.y[0][i] == -1:
                break

            new_pos = {k: v.copy() for k, v in pos.items()}
            for node in new_pos:
                new_pos[node][0] += offset_x

            nx.draw(
                self.G,
                pos=new_pos,
                ax=self.ax,
                with_labels=True,
                edgecolors="black",
                node_size=500,
                font_size=16,
                font_weight="bold",
                node_color=self.data.y[:, i].cpu().numpy(),
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

        # Resize figure (optional; you had this before)
        self.fig.set_size_inches(max(2.5 * max(i, 1), 6), 4)

        # Keep the "Next" button placed correctly
        if self.gui_mode:
            self.reposition_button()
            self.button_ax.set_visible(True)

        # >>> CHANGED: Just draw/flush; no ROS, no buffer, no image conversions
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.draw()


def main(args):
    # Configure matplotlib backend before any figures are created
    if args.gui:
        # Use TkAgg backend for GUI mode (or Qt5Agg if available)
        try:
            matplotlib.use('TkAgg')
        except ImportError:
            try:
                matplotlib.use('Qt5Agg')
            except ImportError:
                logging.getLogger("central").warning(
                    "No interactive matplotlib backend available (install python3-tk or pyqt5). "
                    "Falling back to non-GUI."
                )
                args.gui = False
                matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')

    # rclpy.init()

    graph_generator = GraphGenerator(
        graph_mode=args.mode,
        gui_mode=args.gui,
        baud_rate=args.baud,
        port=args.port,
    )

    try:
        if args.gui:
            # Interactive updates during runtime (your publish_graph_image uses draw/flush)
            plt.ion()
            plt.show(block=False)

            # Keep the GUI + RX callbacks alive until window closed or Ctrl+C
            while plt.fignum_exists(graph_generator.fig.number):
                plt.pause(0.05)  # lets matplotlib process events

        else:
            # Non-GUI: do whatever your non-gui loop will be (placeholder)
            # You can still periodically call process_next_graph() here if desired.
            while True:
                time.sleep(0.5)

    except KeyboardInterrupt:
        logging.getLogger("central").info("Interrupted by user (Ctrl+C).")

    finally:
        # --- Zigbee shutdown ---
        try:
            if hasattr(graph_generator, "zigbee") and graph_generator.zigbee is not None:
                if graph_generator.zigbee.device.is_open():
                    graph_generator.zigbee.device.close()
                    logging.getLogger("central").info("Central XBee device closed.")
        except Exception as e:
                    logging.getLogger("central").warning(f"Failed to close XBee device: {e}")

        # --- Plot handling ---
        # IMPORTANT: do NOT plt.close("all") if you want to show the final figure.
        if args.gui:
            plt.ioff()
            plt.show()   # final blocking show (keeps window open)
        else:
            plt.close("all")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--mode",
        type=str,
        default="load",
        choices=["load", "generate"],
        help='Mode for graph generation: "load" to load from file, "generate" to create random graphs.',
    )
    args.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="If set, display graphs in an interactive GUI window with a Next button.",
    )
    args.add_argument(
        "--continuous", action="store_true", default=False, help="If set, the graph will change continuously over time."
    )
    args.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of steps to interpolate between graphs when in load mode and continuous is set.",
    )
    args.add_argument("--port", default="/dev/ttyUSB0")
    args.add_argument("--baud", type=int, default=9600)
    args.add_argument("--num-nodes", type=int, default=None,help="Number of nodes in the graph.")

    parsed_args = args.parse_args()
    main(parsed_args)
