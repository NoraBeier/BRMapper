import time
import numpy as np
import torch

from torchmetrics import Metric
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    average_precision_score,
    matthews_corrcoef,
)


def get_metric_dict(y_true, y_pred):
    y_cls = np.argmax(y_pred, axis=1)
    y_score = y_pred[:, 1]
    CM = confusion_matrix(y_true, y_cls)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    acc = accuracy_score(y_true, y_cls)
    balanced_acc = balanced_accuracy_score(y_true, y_cls)
    precision = precision_score(y_true, y_cls, zero_division=0)
    recall = recall_score(y_true, y_cls, zero_division=0)
    f1 = f1_score(y_true, y_cls, zero_division=0)
    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    specificity = TN / (FP + TN)
    mcc = matthews_corrcoef(y_true, y_cls)

    return {
        "Accuracy": acc,
        "BalancedAccuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUROC": auroc,
        "AveragePrecision": ap,
        "Specificity": specificity,
        "MatthewsCorrCoef": mcc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Samples": len(y_true)
    }


class ClassificationMetrics(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self._start_time: float | None = None

    def on_epoch_start(self):
        if self._start_time is None:
            self._start_time = time.perf_counter()

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor = None):
        self.preds.append(preds)
        self.targets.append(targets)
        if loss is not None:
            self.loss_sum += loss.detach()
            self.loss_count += 1

    def compute(self):
        y_pred = torch.cat(self.preds).cpu().detach().numpy()
        y_true = torch.cat(self.targets).cpu().detach().numpy()
        metrics = get_metric_dict(y_true, y_pred)
        if self.loss_count > 0:
            metrics["Loss"] = (self.loss_sum / self.loss_count).item()
        metrics["RunningTime"] = time.perf_counter() - self._start_time
        return metrics
