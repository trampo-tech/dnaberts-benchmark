from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def positive_class_probability(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        return 1.0 / (1.0 + np.exp(-logits))
    if logits.shape[-1] == 1:
        return 1.0 / (1.0 + np.exp(-logits[:, 0]))

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs[:, 1]


def numeric_metrics(metrics: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def compute_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    acc_metric,
    f1_metric,
    prec_metric,
    rec_metric,
    num_labels: int = 2,
) -> dict[str, float]:
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    avg = "binary" if num_labels <= 2 else "weighted"
    metrics = {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average=avg)["f1"],
        "precision": prec_metric.compute(predictions=preds, references=labels, average=avg)["precision"],
        "recall": rec_metric.compute(predictions=preds, references=labels, average=avg)["recall"],
    }

    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        if logits.ndim == 1 or logits.shape[-1] <= 2:
            probs = positive_class_probability(np.asarray(logits))
            metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        else:
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(shifted)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            metrics["roc_auc"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
            )
    else:
        metrics["roc_auc"] = 0.0

    return metrics
