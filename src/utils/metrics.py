"""Metric helpers for fraud classification experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Result of threshold selection on validation data."""

    threshold: float
    threshold_table: pd.DataFrame


def get_predicted_labels(scores: pd.Series | np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous scores into binary predictions."""

    return (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute classification metrics from continuous scores and a threshold."""

    y_true_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    y_pred = get_predicted_labels(score_array, threshold=threshold)

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "n_obs": int(len(y_true_array)),
        "n_positive": int(y_true_array.sum()),
        "positive_rate": float(y_true_array.mean()) if len(y_true_array) else 0.0,
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true_array, score_array))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, score_array))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def build_threshold_table(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a threshold sweep table for precision/recall/F1 analysis."""

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = compute_classification_metrics(y_true, scores, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n_predicted_positive": int(metrics["tp"]) + int(metrics["fp"]),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    threshold_table: pd.DataFrame,
    criterion: str = "f1",
) -> ThresholdSelectionResult:
    """Select the best threshold according to one metric column."""

    if threshold_table.empty:
        raise ValueError("Threshold table is empty.")
    if criterion not in threshold_table.columns:
        raise ValueError(f"Criterion '{criterion}' not found in threshold table.")
    best_row = threshold_table.sort_values([criterion, "recall", "precision"], ascending=[False, False, False]).iloc[0]
    return ThresholdSelectionResult(threshold=float(best_row["threshold"]), threshold_table=threshold_table.copy())

