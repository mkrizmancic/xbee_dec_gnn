import importlib

import torch


class GNNWrapper(torch.nn.Module):
    def __init__(self, architecture, in_channels, hidden_channels, num_layers, out_channels=1, **kwargs):
        super().__init__()

        # Store args and kwargs used to initialize the model.
        self.config = {
            "architecture": architecture,
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "num_layers": num_layers,
        }
        self.config.update(kwargs)

        # Sanitize and load kwargs.
        if kwargs.get("jk") == "none":
            kwargs["jk"] = None

        gnn_model = getattr(importlib.import_module("torch_geometric.nn"), architecture)
        self.gnn = gnn_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            **kwargs,
        )

        # Store other class variables.
        self.gnn_is_mlp = architecture == "MLP"

    def forward(self, x, edge_index, batch=None):
        if self.gnn_is_mlp:
            x = self.gnn(x=x, batch=batch)
        else:
            x = self.gnn(x=x, edge_index=edge_index, batch=batch)
        return x

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