import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.models

from torch.nn import ModuleList
from torch_geometric.nn import (
    Linear,
    GATv2Conv,
)

from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)


def init_readout(pool_type):
    pool_fn = None
    if pool_type == "max":
        pool_fn = lambda x, ei, ea, b: global_max_pool(x, b)
    elif pool_type == "mean":
        pool_fn = lambda x, ei, ea, b: global_mean_pool(x, b)
    elif pool_type == "sum":
        pool_fn = lambda x, ei, ea, b: global_add_pool(x, b)
    else:
        raise ValueError("Invalid pool_type '{}'.".format(pool_type))
    return pool_fn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class GIN(nn.Module):
    def __init__(
        self,
        in_features,
        layer_size,
        num_classes,
        dropout,
        num_layers,
        pool_type,
        **kwargs
    ):
        super(GIN, self).__init__()
        self.num_classes = num_classes

        self.gin = torch_geometric.nn.models.GIN(
            in_channels=in_features,
            hidden_channels=layer_size,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.pool_fn = init_readout(pool_type)
        self.fc1 = nn.Linear(layer_size, layer_size)
        self.fc2 = nn.Linear(layer_size, num_classes)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, data):
        x, edge_index, edge_attr, batch_index = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        z = self.gin(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_index)
        z = self.pool_fn(z, edge_index, edge_attr, batch_index)
        z = self.fc1(z)
        y = self.fc2(z)

        return F.log_softmax(y, dim=1)


class GAT(nn.Module):
    def __init__(
        self,
        in_features,
        layer_size,
        num_classes,
        edge_dim,
        num_layers,
        heads,
        dropout,
        pool_type,
        **kwargs
    ):
        super().__init__()
        self.dropout = dropout

        assert num_layers >= 2
        layer_cfg = [in_features]
        layer_cfg += [layer_size for _ in range(num_layers)]

        if not isinstance(heads, list):
            heads = [heads for _ in range(num_layers)]

        assert len(layer_cfg) == len(heads) + 1, "{} != {}".format(
            len(layer_cfg), len(heads) + 1
        )
        assert len(heads) == num_layers

        self.layers = ModuleList()
        in_size = layer_cfg[0]
        for i in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_size,
                    layer_cfg[i + 1],
                    edge_dim=edge_dim,
                    heads=heads[i],
                )
            )
            in_size = layer_cfg[i + 1] * heads[i]

        self.num_classes = num_classes
        self.pool_fn = init_readout(pool_type)

        self.fc1 = Linear(in_size, 64)
        self.fc2 = Linear(64, num_classes)
        self.layers.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, data):
        x, edge_index, edge_attr, batch_index = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            if self.dropout is not None and i < len(self.layers):
                x = F.dropout(x, p=self.dropout)

        x = self.pool_fn(x, edge_index, edge_attr, batch_index)
        z = self.fc1(x)
        y = self.fc2(z)

        return F.log_softmax(y, dim=1)


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_features,
        layer_size,
        num_classes,
        dropout,
        num_layers,
        pool_type,
        **kwargs
    ):
        super(GraphSAGE, self).__init__()
        self.num_classes = num_classes

        self.graph_sage = torch_geometric.nn.models.GraphSAGE(
            in_channels=in_features,
            hidden_channels=layer_size,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.pool_fn = init_readout(pool_type)
        self.fc1 = nn.Linear(layer_size, layer_size)
        self.fc2 = nn.Linear(layer_size, num_classes)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, data):
        x, edge_index, edge_attr, batch_index = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        z = self.graph_sage(
            x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_index
        )
        z = self.pool_fn(z, edge_index, edge_attr, batch_index)
        z = self.fc1(z)
        y = self.fc2(z)

        return F.log_softmax(y, dim=1)


def init_model(name, dataset, **kwargs) -> torch.nn.Module:
    sample = next(iter(dataset))
    kwargs["in_features"] = sample.x.shape[1]
    kwargs["num_classes"] = dataset.num_classes
    kwargs["edge_dim"] = sample.edge_attr.shape[1]

    if name.casefold() == "gin".casefold():
        return GIN(**kwargs)
    elif name.casefold() == "gat".casefold():
        return GAT(**kwargs)
    elif name.casefold() == "graphsage".casefold():
        return GraphSAGE(**kwargs)
    else:
        raise NotImplementedError(
            "You need to implement model '{}' first.".format(name)
        )
