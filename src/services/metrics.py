from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score


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
        if  isinstance(value, (bool, int, float)):
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
    average: str = "macro",
) -> dict[str, float]:
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average=average)["f1"],
        "precision": prec_metric.compute(predictions=preds, references=labels, average=average)["precision"],
        "recall": rec_metric.compute(predictions=preds, references=labels, average=average)["recall"],
        "mcc": float(matthews_corrcoef(labels, preds)),
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


def compute_multilabel_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if probs.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch for AUROC computation: probs={probs.shape}, labels={labels.shape}"
        )

    if probs.ndim > 2:
        probs = probs.reshape(-1, probs.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])

    if probs.ndim != 2:
        raise ValueError(
            f"Expected 2D or 3D arrays for multilabel AUROC, got ndim={probs.ndim}"
        )

    metrics: dict[str, float] = {}
    macro_scores: list[float] = []
    for track_idx in range(probs.shape[1]):
        track_labels = labels[:, track_idx]
        unique_labels = np.unique(track_labels)
        if unique_labels.size < 2:
            continue

        score = float(roc_auc_score(track_labels, probs[:, track_idx]))
        metrics[f"auroc_{track_idx}"] = score
        macro_scores.append(score)

    metrics["auroc_macro"] = float(np.mean(macro_scores)) if macro_scores else 0.0
    return metrics
