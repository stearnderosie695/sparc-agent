from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def class_sensitivity(labels: np.ndarray, predictions: np.ndarray, target: int) -> float:
    binary_true = (labels == target).astype(int)
    binary_pred = (predictions == target).astype(int)
    return float(recall_score(binary_true, binary_pred, zero_division=0))


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    gold = target.astype(bool)
    denom = pred.sum() + gold.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gold).sum() / denom)


def expected_calibration_error(
    labels: np.ndarray, confidences: np.ndarray, predictions: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    correct = (labels == predictions).astype(float)
    total = len(labels)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        accuracy = correct[mask].mean()
        confidence = confidences[mask].mean()
        error += (mask.sum() / total) * abs(accuracy - confidence)
    return float(error)
