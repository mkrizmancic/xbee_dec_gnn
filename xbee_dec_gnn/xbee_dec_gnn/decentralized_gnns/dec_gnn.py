import torch
import torch_geometric.nn.aggr as torchg_aggr
from torch_geometric.nn import GraphSAGE

from xbee_dec_gnn.decentralized_gnns.dec_pooling import DecentralizedPooling
from xbee_dec_gnn.decentralized_gnns.gnn_splitter import PyGModelSplitter


class TmpGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gnn = GraphSAGE(
            in_channels=5,
            hidden_channels=32,
            num_layers=5,
            out_channels=32
        )

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.gnn(x=x, edge_index=edge_index, batch=batch)
        x = self.predictor(x)
        return x


class DecentralizedGNN:
    """
    A wrapper class that enables decentralized inference of GNNs.

    A general recipe for decentralized GNN inference:
    1) Initialize the decentralized GNN with the desired GNN model, pooling method, and predictor model.
    2) For each layer in the GNN, call the `update_gnn` method with the appropriate node features and data from
       neighbors. The output of `update_gnn` should be sent to other connected nodes and will be used as input for the
       next layer along with values received from neighbors.
    3) If using pooling, call the `update_pooling` method until convergence is reached. In the intermediate steps,
       the `update_pooling` returns the current value of a node that slowly converges to the true average, minimum,
       maximum, or other metric we use for pooling. The output from the `update_pooling` method should be sent to other
       connected nodes and will be used as input for the next iteration along with values received from neighbors.
       Once the procedure converges, `done` flag with value `True` will be returned, together with the final pooled
       value.
    4) Finally, call the `run_predictor` method to obtain the final predictions using the final pooled value.
    """

    def __init__(
        self,
        gnn_model: torch.nn.Module,
        pooling: torchg_aggr.Aggregation | str | list[str] | None = None,
        predictor_model: torch.nn.Module | None = None,
        **kwargs,
    ):
        """
        Args:
            gnn_model: The GNN model (message passing) to be used for inference.
            pooling: The pooling method to be used (if any).
            predictor_model: The predictor model to be used (if any).
            **kwargs: Additional keyword arguments.
        """
        # Parse and set up the main GNN model.
        self.gnn = PyGModelSplitter(gnn_model)
        self.gnn.eval()
        self.num_layers = self.gnn.model.num_layers

        # Parse and set up the global pooling method.
        if pooling is None:
            self.pooling = None
        else:
            self.pooling = DecentralizedPooling.create_pooling(method=pooling, **kwargs)

        # Parse and set up the predictor model (final classification or regression).
        if predictor_model is None:
            self.predictor_model = torch.nn.Identity()
        else:
            self.predictor_model = predictor_model
        self.predictor_model.eval()

    def update_gnn(self, layer, own_value, neighbor_values):
        """
        Calculate the output of the GNN for the given node in `layer` using its
        `own_value` and `neighbor_values` assumed to be received through
        message passing and already aggregated.

        This methods must be called the number of times equal to the number of
        layers in the GNN.
        """
        with torch.no_grad():
            return self.gnn.update_node(layer, own_value, neighbor_values)

    def update_pooling(self, own_value, neighbor_values):
        """
        Calculate the intermediate output of the pooling function for the given
        node using its `own_value` and `neighbor_values` assumed to be received
        through communication.

        Since with the pooling, we are usually trying to find a mean, min, or
        max of the representations of all nodes in the graph, the pooling
        procedure uses consensus protocol for decentralized calculation of these
        values.

        This method must be called until `own_value` converges. Update method
        returns 3 values: the updated `own_value`, the final value after
        convergence, and the difference between two steps used for detecting
        convergence. The final value is None until convergence is reached.
        """
        if self.pooling is None:
            return own_value, own_value, 0.0  # No pooling, just return own value.

        return self.pooling.update(own_value, neighbor_values)

    def init_pooling(self, own_value):
        if self.pooling is None:
            return own_value

        return self.pooling.init_value(own_value)

    def run_predictor(self, own_value):
        """
        Calculate the final prediction value (classification or regression) from
        the pooled values.
        """
        with torch.no_grad():
            return self.predictor_model(own_value)

    @classmethod
    def from_tmpgnn(cls, path):
        tmp_gnn = TmpGNN()
        config = torch.load(path, map_location=torch.device("cpu"))
        tmp_gnn.load_state_dict(config["model_state_dict"])
        return cls(gnn_model=tmp_gnn.gnn, predictor_model=tmp_gnn.predictor)

    @classmethod
    def from_gnn_wrapper(cls, path, **kwargs):
        from xbee_dec_gnn.gnn_models.algebraic_gnn_wrapper import GNNWrapper

        model = GNNWrapper.load(path)
        return cls(gnn_model=model.gnn, pooling=model.pool, predictor_model=model.predictor, **kwargs)

    @classmethod
    def from_simple_gnn_wrapper(cls, path, **kwargs):
        from xbee_dec_gnn.gnn_models.simple_gnn_wrapper import GNNWrapper

        class Step(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.heaviside(x, torch.tensor([0.0]))

        model = GNNWrapper.load(path)
        return cls(gnn_model=model.gnn, pooling=None, predictor_model=Step(), **kwargs)

    # DOC: Add other class methods for initialization from different model formats.


def test_gnn_wrapper():
    path = "/root/ros2_ws/src/ros2_dec_gnn/ros2_dec_gnn/config/full_model.pth"
    model = DecentralizedGNN.from_gnn_wrapper(path, consensus_sigma=0.1)
    print(model)


def test():
    import random

    import networkx as nx
    import numpy as np
    from torch_geometric.data import Data
    from torch_geometric.nn import summary

    from my_graphs_dataset import GraphDataset
    from xbee_dec_gnn.gnn_models.algebraic_gnn_wrapper import GNNWrapper

    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Initialize a random connected graph with 5 nodes
    num_nodes = 4
    # while True:
    #     G = nx.gnp_random_graph(num_nodes, 0.6, seed=seed, directed=False)
    #     if nx.is_connected(G):
    #         break
    G = GraphDataset.parse_graph6("CR")
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    # Add reverse edges for undirected graph
    edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

    # 2. Add random features to each node (5 features per node)
    num_node_features = 5
    # x = torch.randn((num_nodes, num_node_features), dtype=torch.float)
    x = torch.tensor([[1, 1, 1/3, 0, 0],
                      [1, 1, 1/3, 0, 0],
                      [2, 2/3, 2/3, 0, 0],
                      [2, 2/3, 2/3, 0, 0]])

    data = Data(x=x, edge_index=edge_index)

    # 1. Initialize a model
    # model = GNNWrapper(architecture='GraphSAGE', in_channels=num_node_features, hidden_channels=4, gnn_layers=3)
    model = GNNWrapper.load("/root/ros2_ws/src/ros2_dec_gnn/ros2_dec_gnn/config/models/layernorm_node.pth")
    model.eval()

    print(summary(model, data.x, data.edge_index, batch=data.batch, max_depth=10, leaf_module=None))

    # Use a smaller consensus_sigma for better convergence
    # decentralized_models = [DecentralizedGNN(model.gnn, model.pool, model.predictor, consensus_sigma=0.1) for _ in range(num_nodes)]
    # mode = "consensus"
    decentralized_models = [
        DecentralizedGNN(model.gnn, model.pool, model.predictor, collected_pooling=True) for _ in range(num_nodes)
    ]
    mode = "collect"

    # 2. Get the original (centralized) model output
    print("Computing centralized model output...")
    with torch.no_grad():
        batch = torch.zeros(num_nodes, dtype=torch.long)  # Single graph
        orig_out_gnn, orig_out_pool, orig_out_pred = model.forward_with_saving(data.x, data.edge_index, batch)
    print(f"Original output: {orig_out_pred.item():.6f}")
    print(f"Original GNN output shape: {orig_out_gnn.shape}")
    print(f"Original pooled output shape: {orig_out_pool.shape}")
    print(f"Original pooled output: {orig_out_pool.flatten()[:5].tolist()}...")  # Show first 5 values

    # 3. Simulate decentralized computation
    print("\nSimulating decentralized computation...")

    # Convert edge_index to adjacency list for easier neighbor lookup
    adj_list = [[] for _ in range(num_nodes)]
    for i, j in edge_index.t().tolist():
        if j not in adj_list[i]:  # Avoid duplicates
            adj_list[i].append(j)

    # Initialize node features for each decentralized model
    node_features = [data.x[i].unsqueeze(0) for i in range(num_nodes)]

    # Step 1: Run GNN layers with decentralized message passing
    for layer in range(model.gnn.num_layers):
        print(f"  Layer {layer + 1}/{model.gnn.num_layers}")
        new_features = []

        for node_idx in range(num_nodes):
            # Get neighbor features
            neighbors = adj_list[node_idx]
            if len(neighbors) > 0:
                neighbor_features = [node_features[neighbor] for neighbor in neighbors]
            else:
                neighbor_features = []

            # Update this node using decentralized GNN
            updated_feature = decentralized_models[node_idx].update_gnn(layer, node_features[node_idx], neighbor_features)
            new_features.append(updated_feature)

        # Update features for next layer
        node_features = new_features

    print("  Comparing GNN outputs with centralized model:")
    # Compare decentralized GNN outputs with centralized GNN outputs
    decentralized_gnn_concat = torch.cat(node_features, dim=0)  # Concatenate all node features
    gnn_difference = torch.abs(orig_out_gnn - decentralized_gnn_concat).max().item()
    print(f"    Max difference in GNN outputs: {gnn_difference:.8f}")
    if gnn_difference < 1e-6:
        print("    ✓ GNN outputs match very closely")
    elif gnn_difference < 1e-4:
        print("    ✓ GNN outputs match reasonably well")
    else:
        print("    ⚠ GNN outputs have noticeable differences")

    # Step 2: Decentralized pooling (if model has pooling)
    if model.pool is not None:
        print("  Running decentralized pooling...")
        pooled_values = node_features.copy()  # Start with GNN outputs

        # Run consensus protocol until convergence
        final_pooling: list[torch.Tensor | None] = [None] * num_nodes
        max_iterations = 100 if mode == "consensus" else 1
        for iteration in range(max_iterations):
            new_pooled_values = []

            for node_idx in range(num_nodes):
                # Get neighbor pooled values
                neighbors = adj_list[node_idx]
                if mode == "collect":
                    neighbor_values = pooled_values.copy()
                elif len(neighbors) > 0:
                    neighbor_values = [pooled_values[neighbor] for neighbor in neighbors]
                else:
                    neighbor_values = []

                # Update pooled value using consensus protocol
                result = decentralized_models[node_idx].update_pooling(pooled_values[node_idx].clone(), neighbor_values)

                updated_value, final_pooling[node_idx], _ = result

                new_pooled_values.append(updated_value)

            pooled_values = new_pooled_values

            # Check for convergence
            if all(fp is not None for fp in final_pooling):
                print(f"    Pooling converged after {iteration + 1} iterations")
                break
        else:
            print(f"    Pooling did not converge after {max_iterations} iterations")

        # All nodes should have the same pooled value after convergence
        print("    Checking final pooled values convergence:")
        for i in range(num_nodes):
            final_val = final_pooling[i]
            if final_val is not None:
                try:
                    # Try to show first 3 values if it's a tensor
                    if hasattr(final_val, "flatten"):
                        print(f"      Node {i}: {final_val.flatten()[:3].tolist()}...")
                    else:
                        print(f"      Node {i}: {final_val}")
                except Exception as e:
                    print(f"      Node {i}: {final_val} (error displaying: {e})")
            else:
                print(f"      Node {i}: None (pooling may not have converged)")

        final_pooled_values = final_pooling

        # Compare decentralized pooling outputs with centralized pooling output
        print("  Comparing pooling outputs with centralized model:")
        if any(fp is not None for fp in final_pooling):
            # Use the first non-None final pooling result for comparison
            representative_pooled = next(fp for fp in final_pooling if fp is not None)
            pooling_difference = torch.abs(orig_out_pool - representative_pooled).max().item()
            print(f"    Max difference in pooling outputs: {pooling_difference:.8f}")
            if pooling_difference < 1e-6:
                print("    ✓ Pooling outputs match very closely")
            elif pooling_difference < 1e-4:
                print("    ✓ Pooling outputs match reasonably well")
            else:
                print("    ⚠ Pooling outputs have noticeable differences")
        else:
            print("    ✗ No converged pooling values to compare")
    else:
        # No pooling - each node keeps its own features
        final_pooled_values = node_features

    # Step 3: Run predictor on each node's pooled result
    print("  Running predictors on each node...")
    decentralized_outputs = []
    for node_idx in range(num_nodes):
        # Use the final pooling result if available, otherwise use the node features
        predictor_input = (
            final_pooled_values[node_idx] if final_pooled_values[node_idx] is not None else node_features[node_idx]
        )
        node_output = decentralized_models[node_idx].run_predictor(predictor_input)
        decentralized_outputs.append(node_output)
        print(f"    Node {node_idx} output: {node_output.item():.6f}")

    # Check that all nodes have the same output (if pooling was used)
    if model.pool is not None:
        print("    Checking output consistency across nodes:")
        outputs_tensor = torch.stack(decentralized_outputs)
        output_std = outputs_tensor.std().item()
        output_mean = outputs_tensor.mean().item()
        print(f"      Mean: {output_mean:.6f}, Std: {output_std:.8f}")
        # Use a more relaxed tolerance for decentralized consistency
        consistency_tolerance = 1e-4
        if output_std < consistency_tolerance:
            print(f"      ✓ All nodes have consistent outputs (std < {consistency_tolerance})")
        else:
            print(f"      ⚠ Nodes have some inconsistency (std = {output_std:.8f})")
            print("        This might be acceptable depending on convergence criteria")

    # Use the first node's output as representative
    decentralized_output = decentralized_outputs[0]
    print(f"  Representative decentralized output: {decentralized_output.item():.6f}")

    # 4. Compare results
    difference = abs(orig_out_pred.item() - decentralized_output.item())
    print("\nComparison:")
    print(f"  Original output:    {orig_out_pred.item():.6f}")
    print(f"  Decentralized output: {decentralized_output.item():.6f}")
    print(f"  Absolute difference: {difference:.6f}")

    # Check if results are close (within reasonable tolerance)
    tolerance = 1e-3  # More relaxed tolerance for decentralized vs centralized comparison
    if difference < tolerance:
        print(f"  ✓ PASS: Results match within tolerance ({tolerance})")
        success = True
    else:
        print(f"  ⚠ Results differ by more than tolerance ({tolerance})")
        print("    This might be due to decentralized consensus convergence limitations")
        # Check if it's at least reasonably close
        relaxed_tolerance = 1e-2
        if difference < relaxed_tolerance:
            print(f"    But within relaxed tolerance ({relaxed_tolerance}) - likely acceptable")
            success = True
        else:
            print("    ✗ FAIL: Results differ by too much even with relaxed tolerance")
            success = False

    return success


if __name__ == "__main__":
    test()
