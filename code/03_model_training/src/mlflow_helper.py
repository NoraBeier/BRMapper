import os
import json
import uuid
import logging
import mlflow
import numpy as np

from contextlib import contextmanager
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource

# Tracking URI can be overridden via the MLFLOW_TRACKING_URI environment variable.
MLFLOW_TRACKING_URL = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


@contextmanager
def _intact_logging_config():
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    try:
        yield
    finally:
        for lg in logging.Logger.manager.loggerDict.values():
            if isinstance(lg, logging.Logger):
                lg.disabled = False
        root.handlers[:] = handlers
        root.setLevel(level)


def get_or_create_experiment(experiment_name):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    with _intact_logging_config():
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        return mlflow.create_experiment(experiment_name)


def mlflow_data_artifact(data, filename, write_callback):
    tmp_dir = os.path.join(".tmp", uuid.uuid4().hex)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    config_artifact_path = os.path.join(tmp_dir, filename)
    write_callback(data, config_artifact_path)
    logging.info(
        "Upload artifact {} to {}".format(filename, mlflow.active_run().info.run_name)
    )
    mlflow.log_artifact(config_artifact_path)

    os.remove(config_artifact_path)
    os.rmdir(tmp_dir)


def mlflow_dict_artifact(data, filename):
    def _callback(data, path):
        with open(path, "w") as f:
            json.dump(data, f)

    mlflow_data_artifact(data, filename, write_callback=_callback)


class TUDSSource(DatasetSource):
    def __init__(self, path):
        self.path = path

    @staticmethod
    def _get_source_type():
        return "TUDataset"

    def to_dict(self):
        return {"path": self.path}

    @classmethod
    def from_dict(cls, source_dict):
        return cls(**source_dict)


class TUDatasetMeta(Dataset):
    def __init__(self, tudataset):
        super().__init__(
            TUDSSource(tudataset.raw_dir), name=tudataset.ds_name, digest="0000"
        )
        self.dataset = tudataset
        self.__profile = None

    def init_profile(self):
        cls_cnts = [0, 0]
        nodes = []
        edges = []

        for entry in self.dataset:
            cls_cnts[entry.y.item()] += 1
            edges.append(int(entry.edge_index.size(1) / 2))
            nodes.append(entry.x.size(0))

        self.__profile = {
            "Positive Samples": int(cls_cnts[1]),
            "Negative Samples": int(cls_cnts[0]),
            "Graphs": [int(c) for c in cls_cnts],
            "Pos./Neg. Ratio": float(cls_cnts[1] / np.sum(cls_cnts)),
            "Avg. Nodes": float(np.mean(nodes)),
            "Avg. Edges": float(np.mean(edges)),
        }

    @property
    def profile(self):
        if self.__profile is None:
            self.init_profile()
        return self.__profile

    def to_dict(self):
        result = super().to_dict()
        result.update({"profile": json.dumps(self.profile)})
        return result
