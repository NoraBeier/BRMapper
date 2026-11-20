import torch
import hydra
import mlflow
import lightning as L

from lightning.pytorch.loggers import MLFlowLogger
from torch.nn import NLLLoss
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig, OmegaConf

from src.mlflow_helper import (
    get_or_create_experiment,
    mlflow_dict_artifact,
    TUDatasetMeta,
    MLFLOW_TRACKING_URL,
)
from src.metric import ClassificationMetrics
from src.utils import load_dataset, compute_class_weights_from_loader, resize_dataset, log_predictions
from src.models import init_model


MLFLOW_TAGS = {
    "project": "Biermapper",
    "model_family": "GNN",
}
MLFLOW_EXP_NAME = "Biermapper GNN Train and Test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)


def _get_tags(cfg):
    tags = MLFLOW_TAGS.copy()
    tags["model"] = cfg.model.type
    tags["state"] = "build"
    if "name" in cfg:
        mlflow.set_tag("name", cfg.name)
    return tags


def _mlflow_init_run(train_dataset_meta, test_dataset_meta, cfg):
    mlflow.log_input(train_dataset_meta, context="training")
    mlflow.log_input(test_dataset_meta, context="testing")
    mlflow.set_tags(_get_tags(cfg))
    mlflow.log_param("undersampling_ratio", cfg.get("undersampling_ratio", None))
    mlflow_dict_artifact(OmegaConf.to_container(cfg, resolve=True), "config.json")


class BinaryGraphClassifier(L.LightningModule):
    def __init__(self, model, loss_fn, batch_size, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.batch_size = batch_size

        self.train_metrics = ClassificationMetrics()
        self.test_metrics = ClassificationMetrics()

        self._test_step_outputs = []
        self._test_outputs = {}

    @property
    def test_outputs(self):
        return self._test_outputs

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.train_metrics.on_epoch_start()

    def on_test_epoch_start(self) -> None:
        self.test_metrics.on_epoch_start()

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss_fn(pred, batch.y)
        self.train_metrics.update(pred, batch.y)
        return loss

    def test_step(self, batch, batch_idx):
        pred = self.model(batch)
        self.test_metrics.update(pred, batch.y)
        self._test_step_outputs.append({"target": batch.y, "prediction": pred})

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict({f"train_{k}": v for k, v in metrics.items()})
        self.train_metrics.reset()

    def on_test_epoch_end(self) -> None:
        outputs = self._test_step_outputs
        targets = torch.cat([o["target"] for o in outputs])
        predictions = torch.cat([o["prediction"] for o in outputs])
        self._test_outputs = {"target": targets, "prediction": predictions}

        metrics = self.test_metrics.compute()
        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


@hydra.main(version_base=None, config_path="cfg", config_name="train_test_gnn.yaml")
def main(cfg: DictConfig):
    train_ds = load_dataset(cfg.ds_train)
    test_ds = load_dataset(cfg.ds_test)
    train_ds_meta = TUDatasetMeta(train_ds)
    test_ds_meta = TUDatasetMeta(test_ds)
    test_loader = DataLoader(test_ds, batch_size=cfg.model.batch_size)
    test_ids = [e.id.item() for e in test_ds]

    experiment_id = get_or_create_experiment(MLFLOW_EXP_NAME)

    run_name = "{}".format(cfg.model.type)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        _mlflow_init_run(train_ds_meta, test_ds_meta, cfg)
        for run_i in range(cfg.runs):
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name="Model {}".format(run_i + 1),
                nested=True,
            ) as run:
                _mlflow_init_run(train_ds_meta, test_ds_meta, cfg)

                _train_ds = resize_dataset(cfg, train_ds)
                train_loader = DataLoader(_train_ds, batch_size=cfg.model.batch_size, shuffle=True)
                class_weight = compute_class_weights_from_loader(train_loader)

                logger = MLFlowLogger(
                    experiment_name=MLFLOW_EXP_NAME,
                    tracking_uri=MLFLOW_TRACKING_URL,
                    tags=_get_tags(cfg),
                    run_id=run.info.run_id,
                )
                loss_fn = NLLLoss(class_weight.to(torch.float32).to(device))
                gnn_model = init_model(cfg.model.type, train_ds, **cfg.model)
                model = BinaryGraphClassifier(
                    gnn_model, loss_fn, cfg.model.batch_size, lr=cfg.model.lr
                )
                trainer = L.Trainer(
                    max_epochs=cfg.model.epochs,
                    logger=logger,
                    log_every_n_steps=10,
                )
                trainer.fit(model=model, train_dataloaders=train_loader)
                trainer.test(model, test_loader)

                target_cls = model.test_outputs["target"].cpu().numpy().tolist()
                predictions = model.test_outputs["prediction"].cpu().numpy()

                log_predictions(target_cls, predictions, test_ids)


if __name__ == "__main__":
    main()
