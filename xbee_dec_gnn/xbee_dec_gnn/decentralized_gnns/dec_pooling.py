import copy

import shortuuid
import torch
import torch_geometric.nn.aggr as torchg_aggr
from torch_geometric.nn.resolver import aggregation_resolver


class DecentralizedPooling:
    def __init__(self, **kwargs) -> None:
        self.convergence_min = kwargs.get("convergence_min", 2)

    @classmethod
    def create_pooling(cls, method: torchg_aggr.Aggregation | str | list[str], **kwargs):
        pooling_protocol = kwargs.pop("pooling_protocol", "consensus")  # "consensus" or "flooding"
        if pooling_protocol == "flooding":
            return DecFloodingPooling(method, **kwargs)
        else:
            return DecConsensusPooling(method, **kwargs)

    def init_value(self, xi):
        self.reset()
        return xi

    def reset(self):
        pass


class DecFloodingPooling(DecentralizedPooling):
    def __init__(self, method: torchg_aggr.Aggregation | str | list[str], **kwargs) -> None:
        super().__init__(**kwargs)

        self.num_nodes = kwargs.get("num_nodes")
        if self.num_nodes is None:
            raise ValueError("When using 'flooding' pooling protocol, 'num_nodes' must be specified in kwargs.")

        if isinstance(method, str):
            self.pooling = aggregation_resolver(method)
        elif isinstance(method, list):
            self.pooling = torchg_aggr.MultiAggregation(method)  # type: ignore
        else:
            self.pooling = method

    def init_value(self, xi):
        self.reset()
        return {shortuuid.uuid(): xi}

    def update(
        self, xi: dict[str, torch.Tensor], xj_list: list[dict[str, torch.Tensor]]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, float]:
        assert self.num_nodes is not None, "'num_nodes' must be specified for flooding pooling protocol."

        # Integrate other values from neighbors.
        for xj in xj_list:
            for hash_key, tensor_value in xj.items():
                # We don't yet have the value of this node, add it.
                if hash_key not in xi:
                    xi[hash_key] = tensor_value
                # We already have the value, but it is different. This shouldn't happen.
                elif not torch.equal(xi[hash_key], tensor_value):
                    raise ValueError(
                        f"Conflicting values received for key '{hash_key}': '{xi[hash_key]}' vs '{tensor_value}'."
                    )

        # Compute the pooled value.
        pooled_value = None
        if len(xi) == self.num_nodes:
            input_value = torch.cat(list(xi.values()), dim=0)
            pooled_value = self.pooling(input_value)

        return xi, pooled_value, 0.0  # Difference is not tracked here.


class DecConsensusPooling(DecentralizedPooling):
    supported_methods_and_dim = {"mean": 1, "max": 1, "min": 1, "std": 2}

    def __init__(self, method: torchg_aggr.Aggregation | str | list[str], **kwargs) -> None:
        super().__init__(**kwargs)

        # Check the method type and convert if necessary.
        if isinstance(method, str):
            method = [method]
        elif isinstance(method, torchg_aggr.MultiAggregation):
            method = [type(m).__name__.split("Aggregation")[0].lower() for m in method.aggrs]
        elif isinstance(method, torchg_aggr.Aggregation):
            method = [type(method).__name__.split("Aggregation")[0].lower()]

        # Set up the required parameters.
        for m in method:
            if m not in self.supported_methods_and_dim:
                raise ValueError(f"Unsupported pooling method: {m}")
            if m in ["mean", "std"]:
                assert "consensus_sigma" in kwargs, f"Missing required kwarg for {m}: consensus_sigma"
                self.consensus_sigma = kwargs["consensus_sigma"]  # The update step size for average consensus.
            elif m == "softmax":
                assert "num_nodes" in kwargs, f"Missing required kwarg for {m}: num_nodes"
                self.num_nodes = kwargs["num_nodes"]
                self._softmax_original_xi = None

        # Store the arguments and initialize class variables.
        self.method = method

        self.first_iter = True
        self.convergence_counter = 0

    def reset(self):
        self.first_iter = True
        self.convergence_counter = 0
        if hasattr(self, "_softmax_original_xi"):
            self._softmax_original_xi = None

    def update(self, xi: torch.Tensor, xj_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None, float]:
        """
        Update the current step of the pooling procedure.

        The output of pooling is calculated based on nodes own value `xi` and
        received values from neighbors `xj_list`.

        Since with the pooling, we are usually trying to find a mean, min, or
        max of the representations of all nodes in the graph, the pooling
        procedure uses consensus protocol for decentralized calculation of these
        values.

        This method returns 3 values: the updated `own_value`, the final value
        after convergence, and the difference between two steps used for
        detecting convergence. The final value is None until convergence is
        reached.
        """
        # In the first iteration, the function receives raw values for pooling.
        # If we are using multiple pooling methods in parallel, we need to
        # repeat the input tensor.
        if self.first_iter:
            total_dim = sum(self.supported_methods_and_dim[m] for m in self.method)
            xi = xi.repeat(total_dim, 1)
            # xj_list is a list of N tensors of shape (R, C),
            for i in range(len(xj_list)):
                xj_list[i] = xj_list[i].repeat(total_dim, 1)

        # grouped_xj will be a tensor of shape (R, N, C).
        # grouped_xj[i] will be a tensor of shape (N, C) containing the i-th rows.
        grouped_xj = torch.stack(xj_list, dim=0).transpose(0, 1)

        m = 0
        for method in self.method:
            # Get the dimension of the current statistic. E.g., 'std' requires keeping track of two values.
            m_dim = self.supported_methods_and_dim[method]

            # Pass the corresponding group of rows to each update function.
            xi[m : m + m_dim] = getattr(self, f"_update_{method}")(xi[m : m + m_dim], grouped_xj[m : m + m_dim])

            # Increase method index
            m += m_dim

        # Keep track of calculated values to detect when they converge.
        # TODO: How to know that everybody is finished?
        diff = 0.0
        if not self.first_iter:
            self.convergence_counter = self.convergence_counter + 1 if torch.allclose(xi, self.old_xi) else 0
            diff = float(torch.norm(xi - self.old_xi, p=2).item())
        self.old_xi = xi.clone()
        self.first_iter = False

        # When consensus is reached, return the final value as well.
        x_pooled = None
        if self.convergence_counter >= self.convergence_min:
            x_pooled = self.return_final(xi)

        return xi, x_pooled, diff

    def _update_mean(self, xi, xj):
        """
        Iteratively compute the mean using average consensus update rule.

        x_i(k+1) = x_i(k) + sigma * sum[x_j(k) - x_i(k)]
        After some iterations, x_i(k) = x_j(k) for all i,j and x_i(k) = mean(x).
        """
        xj = xj.squeeze()
        return xi + self.consensus_sigma * torch.sum(xj - xi, dim=0, keepdim=True)

    def _update_std(self, xi, xj):
        """
        Iteratively compute the standard deviation
        std(x) = sqrt(mean(x^2) - mean(x)^2).

        There are many ways to calculate the standard deviation, but this is how
        torch_geometric.aggr.VarAggregation defines it.

        mean(z) is calculated using average consensus:
            z_i(k+1) = z_i(k) + sigma * sum[z_j(k) - z_i(k)]
        After some iterations, z_i(k) = z_j(k) for all i,j and z_i(k) = mean(z).

        Args:
            xi: The input tensor for the current node (running average). xi[0] has squared values.
            xj: The input tensor for the neighboring nodes.
        """
        # If it's the first iteration, we need to square the inputs.
        if self.first_iter:
            xi[0] = torch.square(xi[0])
            xj[0] = torch.square(xj[0])

        xi[[0]] = self._update_mean(xi[[0]], xj[[0]])
        xi[[1]] = self._update_mean(xi[[1]], xj[[1]])
        return xi

    def _update_max(self, xi, xj):
        xj = xj.squeeze(0)
        xj = xj.max(dim=0, keepdim=True)[0]  # Get only the max value for each element of hidden representation.
        return torch.maximum(xi, xj)

    def _update_min(self, xi, xj):
        xj = xj.squeeze(0)
        xj = xj.min(dim=0, keepdim=True)[0]  # Get only the min value for each element of hidden representation.
        return torch.minimum(xi, xj)

    def _update_softmax(self, xi, xj):
        """
        Iteratively compute the softmax pooling:
        softmax(X) = sum_i{exp(t*x_i) / sum_j[exp(t*x_j)] * x_i}
        softmax(X) = N * mean{exp(t*x_i) / (N * mean[exp(t*x_j)]) * x_i}

        mean(z) is calculated using average consensus:
            z_i(k+1) = z_i(k) + sigma * sum[z_j(k) - z_i(k)]
        After some iterations, z_i(k) = z_j(k) for all i,j and z_i(k) = mean(z).
        """
        # If it's the first iteration, we need to exponentiate the inputs and store the original input.
        if self.first_iter:
            xi = torch.exp(xi)
            xj = torch.exp(xj)
            self._softmax_original_xi = xi.clone()

        return self._update_mean(xi, xj)

    def return_final(self, xi):
        """
        Return the final pooled value after convergence.

        For some aggregation methods, we need to do some final calculations on
        the converged values to get the desired statistic.
        Pooling results from different methods are concatenated into a single
        vector.s
        """
        x_pooled = torch.empty(len(self.method), xi.shape[1])
        m = 0
        for p, method in enumerate(self.method):
            # Get the dimension of the current statistic. E.g., 'std' requires keeping track of two values.
            m_dim = self.supported_methods_and_dim[method]

            # Pass the corresponding group of rows to each update function.
            x_pooled[p] = getattr(self, f"_return_{method}")(xi[m : m + m_dim])

            # Increase method index
            m += m_dim

        # TODO: Infer how to combine the results from the provided Aggregation object (cat, max, etc.)
        x_pooled = torch.reshape(x_pooled, (1, -1))

        return x_pooled

    def _return_mean(self, xi):
        return xi

    def _return_std(self, xi):
        return torch.sqrt(xi[0] - torch.square(xi[1]))

    def _return_max(self, xi):
        return xi

    def _return_min(self, xi):
        return xi

    def _return_softmax(self, xi):
        return self._softmax_original_xi / (self.num_nodes * xi)


def test_consensus_pooling():
    from torch_geometric.seed import seed_everything

    seed_everything(42)

    num_nodes = 5
    hc_dim = 4
    all_done = [False for i in range(num_nodes)]

    methods_list = list(DecConsensusPooling.supported_methods_and_dim.keys())
    methods_dims = sum(DecConsensusPooling.supported_methods_and_dim.values())

    # methods_list = ["mean", "std"]
    # methods_dims = sum(DecentralizedConsensusPooling.supported_methods_and_dim[method] for method in methods_list)

    # orig_values = torch.randn((num_nodes, hc_dim))
    orig_values = torch.mul(torch.ones((num_nodes, hc_dim)), torch.tensor([[6], [1], [1], [1], [1]]))
    old_values = orig_values.unsqueeze(1).clone()
    new_values = torch.empty((num_nodes, methods_dims, hc_dim))
    final_values = torch.empty((num_nodes, len(methods_list), hc_dim))
    expected_values = torch.empty((len(methods_list), hc_dim))

    funcs = {
        "mean": lambda x: x.mean(dim=0),
        "max": lambda x: x.max(dim=0)[0],
        "min": lambda x: x.min(dim=0)[0],
        "std": lambda x: x.std(dim=0, unbiased=False),
        "softmax": lambda x: torch.softmax(x, dim=0),
    }

    for i, method in enumerate(methods_list):
        expected_values[i] = funcs[method](orig_values).unsqueeze(0)

    pooling_exec = [
        DecConsensusPooling(
            method=methods_list,
            consensus_sigma=0.25,  # 1 / (num_nodes-1),
        )
        for _ in range(num_nodes)
    ]

    iteration = 0
    while not all(all_done):
        iteration += 1
        for i in range(num_nodes):
            neighbor_values = [old_values[(i - 1) % num_nodes], old_values[(i + 1) % num_nodes]]
            new_values[i], final, _ = pooling_exec[i].update(old_values[i].clone(), neighbor_values)
            if final is not None:
                final_values[i] = torch.reshape(final, (len(methods_list), hc_dim))
                all_done[i] = True
        if any(all_done):
            print(iteration, all_done)
        old_values = new_values.clone()

    print(f"Pooling completed in {iteration} iterations.")
    all_neighbors_close = [torch.allclose(new_values[i], new_values[i - 1], atol=1e-4) for i in range(1, num_nodes)]
    print(f"{all_neighbors_close=}")

    for i in range(num_nodes):
        print(f"Node_{i} good: {torch.allclose(final_values[i], expected_values, atol=1e-4)}")


def test_flooding_pooling():
    from torch_geometric.seed import seed_everything

    seed_everything(42)

    num_nodes = 5
    hc_dim = 4
    all_done = [False for i in range(num_nodes)]

    methods_list = list(DecConsensusPooling.supported_methods_and_dim.keys())
    methods_dims = sum(DecConsensusPooling.supported_methods_and_dim.values())

    # methods_list = ["mean", "std"]
    # methods_dims = sum(DecentralizedConsensusPooling.supported_methods_and_dim[method] for method in methods_list)

    # orig_values = torch.randn((num_nodes, hc_dim))
    orig_values = torch.mul(torch.ones((num_nodes, hc_dim)), torch.tensor([[6], [1], [1], [1], [1]]))
    orig_values = orig_values.unsqueeze(1).clone()
    old_values = dict()
    new_values = {i: dict() for i in range(num_nodes)}
    final_values = torch.empty((num_nodes, len(methods_list), hc_dim))
    expected_values = torch.empty((len(methods_list), hc_dim))

    funcs = {
        "mean": lambda x: x.mean(dim=0),
        "max": lambda x: x.max(dim=0)[0],
        "min": lambda x: x.min(dim=0)[0],
        "std": lambda x: x.std(dim=0, unbiased=False),
        "softmax": lambda x: torch.softmax(x, dim=0),
    }

    for i, method in enumerate(methods_list):
        expected_values[i] = funcs[method](orig_values).unsqueeze(0)

    pooling_exec = []
    for i in range(num_nodes):
        pooling_exec.append(DecFloodingPooling(method=methods_list, num_nodes=num_nodes))
        old_values[i] = pooling_exec[-1].init_value(orig_values[i])

    iteration = 0
    while not all(all_done):
        iteration += 1
        for i in range(num_nodes):
            neighbor_values = [old_values[(i - 1) % num_nodes], old_values[(i + 1) % num_nodes]]
            new_values[i], final, _ = pooling_exec[i].update(old_values[i], neighbor_values)
            if final is not None:
                final_values[i] = torch.reshape(final, (len(methods_list), hc_dim))
                all_done[i] = True
        if any(all_done):
            print(iteration, all_done)
        old_values = copy.deepcopy(new_values)

    print(f"Pooling completed in {iteration} iterations.")
    all_neighbors_close = [torch.allclose(final_values[i], final_values[i - 1], atol=1e-4) for i in range(1, num_nodes)]
    print(f"{all_neighbors_close=}")

    for i in range(num_nodes):
        print(f"Node_{i} good: {torch.allclose(final_values[i], expected_values, atol=1e-4)}")


if __name__ == "__main__":
    print("TESTING DECENTRALIZED CONSENSUS POOLING")
    test_consensus_pooling()

    print()
    print("TESTING DECENTRALIZED FLOODING POOLING")
    test_flooding_pooling()


# TODO: Implment dynamic consensus
# We would need to come up with a new message class that would hold the incoming
# value, ID of the sender, and the sequence number of the message. The update
# method could be modified to parse either a list of tensors (as it is now) or
# this new message class. The decentralized pooling class would keep track of the
# necessary variables to implement the dynamic consensus algorithm like the
# matrix of sequence numbers and deltas. Only _update_mean would need to be
# modified to implement the dynamic consensus update rule. The rest of the code
# either uses _update_mean or is unaffected. For example, min and max can be
# calculated from the received message without the agent ID or sequence number.
# Maybe the best way would be to subclass it.
# In the ROS node, we would need to modify the receive_pooling method to
# compute the new value immediately upon receiving a message instead of storing
# it and waiting for all neighbors to send their messages. We could make two
# versions of the callback function and select which one to use from a flag.
