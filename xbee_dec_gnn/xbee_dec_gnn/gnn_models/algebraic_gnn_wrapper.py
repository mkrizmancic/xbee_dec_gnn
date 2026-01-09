import importlib

import torch
import torch_geometric.nn.aggr as torchg_aggr


def get_global_pooling(name, **kwargs):
    options = {
        "mean": torchg_aggr.MeanAggregation(),
        "max": torchg_aggr.MaxAggregation(),
        "add": torchg_aggr.SumAggregation(),
        "min": torchg_aggr.MinAggregation(),
        "median": torchg_aggr.MedianAggregation(),
        "var": torchg_aggr.VarAggregation(),
        "std": torchg_aggr.StdAggregation(),
        "softmax": torchg_aggr.SoftmaxAggregation(learn=True),
        "s2s": torchg_aggr.Set2Set,
        "minmax": torchg_aggr.MultiAggregation(aggrs=["min", "max"]),
        "multi": torchg_aggr.MultiAggregation(aggrs=["min", "max", "mean", "std"]),
        "multi++": torchg_aggr.MultiAggregation,
        "PNA": torchg_aggr.DegreeScalerAggregation,
        # "powermean": PowerMeanAggregation(learn=True),  # Results in NaNs and error
        # "mlp": MLPAggregation,  # NOT a permutation-invariant operator
        # "sort": SortAggregation,  # Requires sorting node representations
    }

    hc = kwargs["hidden_channels"]
    if name is None:
        return None, kwargs["hidden_channels"]

    pool = options[name]

    if name == "s2s":
        pool = pool(in_channels=hc, processing_steps=4)
        hc = 2 * hc
    elif name == "PNA":
        pool = pool(
            aggr=["mean", "min", "max", "std"], scaler=["identity", "amplification", "attenuation"], deg=kwargs["deg"]
        )
        hc = len(pool.aggr.aggrs) * len(pool.scaler) * hc
    elif name in ["multi", "minmax"]:
        hc = len(pool.aggrs) * hc
    elif name == "multi++":
        pool = pool(
            aggrs=[
                torchg_aggr.Set2Set(in_channels=hc, processing_steps=4),
                torchg_aggr.SoftmaxAggregation(learn=True),
                torchg_aggr.MinAggregation(),
            ]
        )
        hc = (2 + 1 + 1) * hc

    return pool, hc


class GNNWrapper(torch.nn.Module):
    def __init__(
        self,
        architecture: str,
        in_channels: int,
        hidden_channels: int,
        gnn_layers: int,
        **kwargs,
    ):
        super().__init__()

        # Store args and kwargs used to initialize the model.
        self.config = {
            "architecture": architecture,
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "gnn_layers": gnn_layers,
        }
        self.config.update(kwargs)

        # Sanitize and load kwargs.
        if kwargs.get("jk") == "none":
            kwargs["jk"] = None

        self.mlp_layers = kwargs.pop("mlp_layers", 1)
        pool = kwargs.pop("pool", "mean")
        pool_kwargs = kwargs.pop("pool_kwargs", {})

        self.save_embeddings_freq = kwargs.pop("save_embeddings_freq", float("inf"))
        self.embeddings = {}

        # Split the normalization layer string.
        norm = kwargs.get("norm")
        norm_kwargs = dict()
        if norm is not None and "_" in norm:
            norm_type, norm_param = norm.split("_", 1)
            if norm_type == "layer":
                norm_kwargs["mode"] = norm_param
            else:
                raise ValueError(f"Unknown normalization type: {norm}.")
            kwargs["norm"] = norm_type
            kwargs["norm_kwargs"] = norm_kwargs

        # Build the model: 1) pre-scaler, 2) GNN, 3) pooling, 4) final regression predictor
        self.pre_scaler = None
        if kwargs.pop("pre_scaler", False):
            # If pre-scaling is enabled, the input channels are scaled to hidden channels
            self.pre_scaler = torch.nn.Linear(in_channels, hidden_channels)
            in_channels = hidden_channels

        gnn_model = getattr(importlib.import_module("torch_geometric.nn"), architecture)
        self.gnn = gnn_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=gnn_layers,
            **kwargs,
        )

        self.pool, hc = get_global_pooling(pool, hidden_channels=hidden_channels, **pool_kwargs)

        mlp_layer_list = []
        for i in range(self.mlp_layers):
            if i < self.mlp_layers - 1:
                mlp_layer_list.append(torch.nn.Linear(hc, hidden_channels))
                mlp_layer_list.append(torch.nn.ReLU())
                hc = hidden_channels
            else:
                mlp_layer_list.append(torch.nn.Linear(hidden_channels, 1))
        self.predictor = torch.nn.Sequential(*mlp_layer_list)

        # Store other class variables.
        self.gnn_is_mlp = architecture == "MLP"

    def forward(self, x, edge_index, batch, epoch=-1):
        if self.pre_scaler is not None:
            x = self.pre_scaler(x)  # Pre-scaling the input features
        if self.gnn_is_mlp:
            x = self.gnn(x=x, batch=batch)
        else:
            x = self.gnn(x=x, edge_index=edge_index, batch=batch)

        x = self.pool(x, batch) if self.pool is not None else x

        if epoch == 1 or epoch % self.save_embeddings_freq == 0:
            if epoch not in self.embeddings:
                self.embeddings[epoch] = x.detach().cpu()
            else:
                self.embeddings[epoch] = torch.cat((self.embeddings[epoch], x.detach().cpu()), dim=0)

        x = self.predictor(x)

        return x

    def forward_with_saving(self, x, edge_index, batch):
        if self.pre_scaler is not None:
            x = self.pre_scaler(x)  # Pre-scaling the input features
        if self.gnn_is_mlp:
            x = self.gnn(x=x, batch=batch)
        else:
            x = self.gnn(x=x, edge_index=edge_index, batch=batch)
        x_gnn = x.clone()

        x = self.pool(x, batch) if self.pool is not None else x
        x_pool = x.clone()

        x = self.predictor(x)

        return x_gnn, x_pool, x

    def save(self, path):
        model_dict = {
            "config": self.config,
            "model": self.state_dict(),
        }
        torch.save(model_dict, path)

    @classmethod
    def load(cls, path):
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        model_dict = torch.load(path, map_location=map_location)
        model = cls(**model_dict["config"])
        model.load_state_dict(model_dict["model"])
        return model
