import os
import logging
import random
import mlflow
import tempfile
import torch
import numpy as np
import pandas as pd
import networkx as nx

from collections import Counter
from fgutils.torch import TUDataset
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from sklearn.model_selection import train_test_split


def log_predictions(y_test, test_pred, ids):
    test_cls = np.argmax(test_pred, axis=1)
    assert len(ids) == len(y_test), f"Mismatch of len(ids) and len(y_test). {len(ids)} != {len(y_test)}"
    assert len(ids) == len(test_cls), f"Mismatch of len(ids) and len(test_cls). {len(ids)} != {len(test_cls)}"
    assert len(ids) == len(test_pred[:,1]), f"Mismatch of len(ids) and len(test_pred[:,1]). {len(ids)} != {len(test_pred[:,1])}"
    result_df = pd.DataFrame(
        {"id": ids, "target_class": y_test, "predicted_class": test_cls, "score": test_pred[:, 1]}
    )
    with tempfile.TemporaryDirectory() as dir:
        file_path = os.path.join(dir, "predictions.csv")
        result_df.to_csv(file_path, index=False)
        mlflow.log_artifact(file_path)


class RemoveIsomorphicSamples:
    def __init__(self, iterations=5):
        self._filter_hash_dict = {}
        self.iterations = iterations

    def __call__(self, data: Data):
        G = to_networkx(
            data, node_attrs=["x"], edge_attrs=["edge_attr"], to_undirected=True
        )
        G_hash = nx.weisfeiler_lehman_graph_hash(
            G, edge_attr="edge_attr", node_attr="x", iterations=self.iterations
        )
        if G_hash in self._filter_hash_dict:
            return False
        else:
            self._filter_hash_dict[G_hash] = G
            return True


class RemoveZeroSamples:
    def __init__(self):
        pass

    def __call__(self, data: Data):
        if data.x.shape[0] == 0:
            logging.warning(
                "Remove sample (id={}) with no nodes.".format(data.id.item())
            )
            return False
        if data.edge_index.shape[1] == 0:
            logging.warning(
                "Remove sample (id={}) with no edges.".format(data.id.item())
            )
            return False
        return True


class MultiFilter:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, data):
        for f in self.filters:
            keep = f(data)
            if keep is False:
                return False
        return True


def load_dataset(cfg, filter=None):
    logging.info("Loading dataset {}.".format(cfg.name))
    if filter is None:
        filter = MultiFilter([RemoveZeroSamples(), RemoveIsomorphicSamples()])
    dataset = TUDataset(cfg.path, cfg.name, pre_filter=filter)
    logging.info("Using filter {} results in {} samples.".format(filter, len(dataset)))

    return dataset


def subset_dataset(dataset, size):
    targets = dataset._data.y

    if size >= len(dataset):
        logging.info(f"Skip subset sampling to size {size} because dataset has only {len(dataset)} instances.")
        return dataset

    _, subset_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=size,
        shuffle=True,
        stratify=targets,
    )
    subset = Subset(dataset, subset_idx)
    targets = targets[subset.indices]
    logging.info(f"Use subset. Reduce dataset size from {len(dataset)} to {len(subset)} (ratio={len(subset)/len(dataset):.2f}).")
    return subset


def undersample_dataset(dataset, fraction: float, minority_cnt=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    labels = np.array([e.y.item() for e in dataset])
    cls_counts = Counter(labels)
    if len(cls_counts) != 2:
        raise RuntimeError(
            "Undersampling TUDataset is only implemented for binary datasets."
        )

    majority_cls = max(cls_counts, key=cls_counts.get)  # type: ignore
    minority_indices = np.where(labels != majority_cls)[0]
    majority_indices = np.where(labels == majority_cls)[0]

    if minority_cnt is not None:
        logging.info(f"Use fixed minority count {minority_cnt} instead of {min(cls_counts.values())}.")
        minority_n = minority_cnt
        minority_indices = np.random.choice(
            minority_indices, size=minority_n, replace=False
        )
    else:
        minority_n = min(cls_counts.values())

    majority_n = int(np.round(minority_n / fraction))
    majority_n = min(majority_n, len(majority_indices))
    undersampled_majority_indices = np.random.choice(
        majority_indices, size=majority_n, replace=False
    )
    indices = undersampled_majority_indices.tolist() + minority_indices.tolist()
    random.shuffle(indices)

    return Subset(dataset, indices)


def resize_dataset(cfg, dataset):
    subset_size = cfg.get("subset", None)
    undersampling_ratio = cfg.get("undersampling_ratio", None)
    if subset_size is not None and undersampling_ratio is not None:
        raise RuntimeError("Subset and undersampling ratio cannot be used together.")
    if subset_size is not None:
        dataset = subset_dataset(dataset, subset_size)
    if undersampling_ratio is not None:
        dataset = undersample_dataset(dataset, undersampling_ratio, minority_cnt=cfg.get("minority_cnt", None))
    return dataset


def compute_class_weights_from_loader(dataloader, num_classes=None):
    all_labels = []

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            labels = batch[1]  # standard (inputs, labels) format
        elif hasattr(batch, "y"):
            labels = batch.y  # PyG Data object with .y
        else:
            raise ValueError("Batch format not recognized.")

        all_labels.extend(labels.cpu().numpy().tolist())

    label_counts = Counter(all_labels)

    if num_classes is None:
        num_classes = max(label_counts.keys()) + 1

    # Compute weights: inverse of frequency
    total = sum(label_counts.values())
    class_weights = []
    for cls in range(num_classes):
        count = label_counts.get(cls, 0)
        weight = total / (count if count > 0 else 1)  # avoid div by zero
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()

    return class_weights
