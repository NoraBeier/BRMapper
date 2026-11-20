import os
import time
import tempfile
import hydra
import mlflow
import logging
import numpy as np
import pandas as pd

from collections import Counter
from omegaconf import DictConfig, OmegaConf
from sklearn.utils import Bunch
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.mlflow_helper import (
    TUDatasetMeta,
    get_or_create_experiment,
    mlflow_dict_artifact,
)
from src.svm import to_svm_dataset, init_kernel, test_model
from src.utils import load_dataset, resize_dataset, log_predictions

MLFLOW_TAGS = {
    "project": "Biermapper",
    "model_family": "SVM",
}
MLFLOW_EXP_NAME = "Biermapper SVM Train and Test"


def _get_tags(cfg):
    tags = MLFLOW_TAGS.copy()
    tags["model"] = "{} SVM".format(cfg.model.kernel.type)
    tags["kernel"] = cfg.model.kernel.type
    tags["state"] = "train/test"
    if "name" in cfg:
        mlflow.set_tag("name", cfg.name)
    return tags


def _initial_log(train_dataset_meta, test_dataset_meta, cfg):
    mlflow.log_input(train_dataset_meta, context="training")
    mlflow.log_input(test_dataset_meta, context="testing")
    mlflow.set_tags(_get_tags(cfg))
    mlflow.log_param("undersampling_ratio", cfg.undersampling_ratio)
    mlflow_dict_artifact(OmegaConf.to_container(cfg, resolve=True), "config.json")


def convert_dataset(dataset):
    dataset = to_svm_dataset(dataset)
    assert isinstance(dataset, Bunch)
    return dataset.data, dataset.target


def train_test_run(cfg, train_dataset, X_test, y_test, test_ids, run_name):
    train_dataset = resize_dataset(cfg, train_dataset)

    X_train, y_train = convert_dataset(train_dataset)
    cls_counts = Counter(y_train)
    logging.info(
        "Use undersampled train dataset. | pos={} neg={} (total={})".format(
            cls_counts[1], cls_counts[0], len(y_train)
        )
    )

    logging.info("Setup graph kernel and SVM")
    kernel = init_kernel(cfg.model.kernel)
    svm = CalibratedClassifierCV(
        SVC(
            kernel="precomputed",
            class_weight="balanced",
            **cfg.model.get("svm", {}),
        ),
        ensemble=False
    )
    if cfg.scale_C:
        svm.estimator.C *= 1.0 / np.sqrt(len(X_train))
    mlflow.log_params(svm.get_params())
    logging.info("Compute gramm matrix")
    t1 = time.time()
    K_train = kernel.fit_transform(X_train)
    logging.info(f"Train SVM from kernel matrix (size={K_train.shape})")
    t2 = time.time()
    svm.fit(K_train, y_train)
    t3 = time.time()
    train_result = test_model(kernel, svm, X_train, y_train, progress=True)
    train_result["running_time"] = t3 - t1
    mlflow.log_metric("runningtime_gramm_matrix_computation", t2 - t1)
    mlflow.log_metrics({"train_{}".format(k): v for k, v in train_result.items()})

    logging.info(f"Test SVM on {len(y_test)} samples.")
    t4 = time.time()
    test_result, test_pred = test_model(
        kernel, svm, X_test, y_test, return_prediction=True, progress=True
    )
    test_result["running_time"] = time.time() - t4
    logging.info(f"Matrix computation: {t2 - t1:.3f}s | Train SVM: {t3 - t2:.3f}s | Train infer: {t4 - t3:.3f}s | Test infer: {time.time() - t4:.3f}s")
    mlflow.log_metrics({"test_{}".format(k): v for k, v in test_result.items()})
    logging.info(
        "Training: BAcc={:.4f} F1={:.4f} | Testing: BAcc={:.4f} F1={:.4f}".format(
            train_result["BalancedAccuracy"],
            train_result["F1"],
            test_result["BalancedAccuracy"],
            test_result["F1"],
        )
    )
    log_predictions(y_test, test_pred, test_ids)
    return test_result, train_result


def _log_results(result_df, prefix):
    result_mean = result_df.mean().to_dict()
    result_std = result_df.std().to_dict()
    result_mean = {"{}_{}_mean".format(prefix, k): v for k, v in result_mean.items()}
    result_std = {"{}_{}_std".format(prefix, k): v for k, v in result_std.items()}
    mlflow.log_metrics(result_mean)
    mlflow.log_metrics(result_std)


@hydra.main(version_base=None, config_path="cfg", config_name="train_test_svm.yaml")
def run(cfg: DictConfig) -> None:
    train_dataset = load_dataset(cfg.ds_train)
    test_dataset = load_dataset(cfg.ds_test)
    logging.info(
        "Loaded datasets: Trainingset={} Testset={}".format(
            len(train_dataset), len(test_dataset)
        )
    )
    train_ds_meta = TUDatasetMeta(train_dataset)
    test_ds_meta = TUDatasetMeta(test_dataset)
    logging.info("Convert TUDatasets to SVM datasets")
    X_test, y_test = convert_dataset(test_dataset)
    test_ids = [e.id.item() for e in test_dataset]

    exp_id = get_or_create_experiment(MLFLOW_EXP_NAME)
    run_name = "{}".format(cfg.model.kernel.type)
    if cfg.undersampling_ratio is not None:
        run_name += " a={:.2f}".format(cfg.undersampling_ratio)

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, nested=True):
        _initial_log(train_ds_meta, test_ds_meta, cfg)
        if cfg.runs == 1:
            train_test_run(cfg, train_dataset, X_test, y_test, test_ids, run_name)
        else:
            test_results = []
            train_results = []
            for run_i in range(cfg.runs):
                logging.info("Start train/test run {}/{}".format(run_i + 1, cfg.runs))
                with mlflow.start_run(
                    experiment_id=exp_id,
                    run_name="Model {}".format(run_i + 1),
                    nested=True,
                ):
                    _initial_log(train_ds_meta, test_ds_meta, cfg)
                    test_result, train_result = train_test_run(
                        cfg,
                        train_dataset,
                        X_test,
                        y_test,
                        test_ids,
                        run_name + "_run_{}".format(run_i + 1)
                    )
                    test_results.append(test_result)
                    train_results.append(train_result)
            test_result_df = pd.DataFrame(test_results)
            train_result_df = pd.DataFrame(train_results)
            with tempfile.TemporaryDirectory() as dir:
                file_path = os.path.join(
                    dir, "results_{}.csv".format(run_name.replace(" ", "_"))
                )
                test_result_df.to_csv(file_path, index=False)
                mlflow.log_artifact(file_path)

            _log_results(test_result_df, "test")
            _log_results(train_result_df, "train")


if __name__ == "__main__":
    run()
