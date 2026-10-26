# Modified from
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/tu_dataset.py
# Date: 23.02.2025, Commit: d1f305b

import os

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs, read_tu_data


class TUDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_node_attr: bool = True,
        use_edge_attr: bool = True,
        force_reload=False,
    ):
        self.ds_name = name
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        out = fs.torch_load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again."
            )
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)
        if self._data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self._data.x = self._data.x[:, num_node_attributes:]
        if self._data.edge_attr is not None and not use_edge_attr:
            num_edge_attrs = self.num_edge_attributes
            self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.ds_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.ds_name)

    @property
    def raw_file_names(self):
        return [
            "{}_{}".format(self.ds_name, n)
            for n in [
                "A",
                "edge_attributes",
                "node_attributes",
                "graph_indicator",
                "graph_labels",
            ]
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def num_node_labels(self) -> int:
        return self.sizes["num_node_labels"]

    @property
    def num_node_attributes(self) -> int:
        return self.sizes["num_node_attributes"]

    @property
    def num_edge_labels(self) -> int:
        return self.sizes["num_edge_labels"]

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes["num_edge_attributes"]

    def download(self):
        pass

    def process(self) -> None:
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.ds_name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        fs.torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )
