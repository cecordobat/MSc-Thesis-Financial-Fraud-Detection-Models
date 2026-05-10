"""Model evaluation helpers for notebook-driven experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.metrics import (
    ThresholdSelectionResult,
    build_threshold_table,
    compute_classification_metrics,
    get_predicted_labels,
    select_threshold,
)


def get_model_scores(model: object, x_frame: pd.DataFrame) -> pd.Series:
    """Return a continuous fraud score from a fitted estimator."""

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_frame)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(x_frame)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=float)))
    else:
        scores = np.asarray(model.predict(x_frame), dtype=float)
    return pd.Series(scores, index=x_frame.index, name="score")


def get_positive_class_scores(model: object, x_frame: pd.DataFrame) -> pd.Series:
    """Backward-compatible alias used by notebook code."""

    return get_model_scores(model, x_frame)


def evaluate_validation_thresholds(
    model: object,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    criterion: str = "f1",
) -> ThresholdSelectionResult:
    """Sweep thresholds on validation data and return the selected threshold."""

    scores = get_model_scores(model, x_validation)
    threshold_table = build_threshold_table(y_validation, scores)
    return select_threshold(threshold_table, criterion=criterion)


def evaluate_classifier(
    model: object,
    x_frame: pd.DataFrame,
    y_true: pd.Series,
    split_name: str,
    model_name: str,
    subset_name: str,
    threshold: float,
) -> tuple[dict[str, object], pd.Series]:
    """Evaluate a fitted classifier on one dataset split."""

    scores = get_model_scores(model, x_frame)
    metrics = compute_classification_metrics(y_true, scores, threshold=threshold)
    metrics.update(
        {
            "model_name": model_name,
            "subset_name": subset_name,
            "feature_subset": subset_name,
            "split": split_name,
        }
    )
    return metrics, scores


def build_metrics_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Convert a list of metric dictionaries into a dataframe."""

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def save_metrics_table(
    frame: pd.DataFrame,
    csv_path: str | Path,
    excel_path: str | Path | None = None,
) -> None:
    """Persist a metrics table to CSV and optionally Excel."""

    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_target, index=False)
    if excel_path is not None:
        excel_target = Path(excel_path)
        excel_target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_excel(excel_target, index=False)


def save_threshold_table(
    frame: pd.DataFrame,
    csv_path: str | Path,
    excel_path: str | Path | None = None,
) -> None:
    """Persist a threshold optimization table."""

    save_metrics_table(frame, csv_path=csv_path, excel_path=excel_path)


__all__ = [
    "build_metrics_frame",
    "evaluate_classifier",
    "evaluate_validation_thresholds",
    "get_model_scores",
    "get_positive_class_scores",
    "get_predicted_labels",
    "save_metrics_table",
    "save_threshold_table",
    "select_threshold",
]
