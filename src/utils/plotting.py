"""Plotting helpers for model evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.utils.metrics import get_predicted_labels


def _ensure_parent(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    confusion_values: np.ndarray | pd.Series,
    labels: tuple[str, str] | pd.Series | np.ndarray = ("No Fraud", "Fraud"),
    title: str | None = None,
    output_path: str | Path | None = None,
    threshold: float | None = None,
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""

    if threshold is not None and not (
        isinstance(labels, tuple) and len(labels) == 2 and all(isinstance(item, str) for item in labels)
    ):
        y_true = np.asarray(confusion_values, dtype=int)
        scores = np.asarray(labels, dtype=float)
        y_pred = get_predicted_labels(scores, threshold=threshold)
        confusion_values = np.array(
            [
                [((y_true == 0) & (y_pred == 0)).sum(), ((y_true == 0) & (y_pred == 1)).sum()],
                [((y_true == 1) & (y_pred == 0)).sum(), ((y_true == 1) & (y_pred == 1)).sum()],
            ]
        )
        labels = ("No Fraud", "Fraud")

    fig, ax = plt.subplots(figsize=(7, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_values, display_labels=list(labels))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(_ensure_parent(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    title: str | None = None,
    output_path: str | Path | None = None,
    model_name: str | None = None,
) -> None:
    """Plot and optionally save a ROC curve."""

    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax)
    final_title = title or (f"ROC Curve - {model_name}" if model_name else None)
    if final_title:
        ax.set_title(final_title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(_ensure_parent(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    title: str | None = None,
    output_path: str | Path | None = None,
    model_name: str | None = None,
) -> None:
    """Plot and optionally save a precision-recall curve."""

    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax)
    final_title = title or (f"Precision-Recall Curve - {model_name}" if model_name else None)
    if final_title:
        ax.set_title(final_title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(_ensure_parent(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importance_frame: pd.DataFrame,
    top_n: int = 20,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> None:
    """Plot and optionally save a horizontal feature importance chart."""

    plot_frame = importance_frame.copy().sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_frame))))
    sns.barplot(data=plot_frame, x="importance", y="feature", ax=ax, color="#1f77b4")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(_ensure_parent(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_probability_distribution(
    scores: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    threshold: float,
    title: str | None = None,
    output_path: str | Path | None = None,
    bins: int = 60,
) -> None:
    """Plot and optionally save the predicted-probability distribution by class."""

    scores_array = np.asarray(scores, dtype=float)
    y_true_array = np.asarray(y_true, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(scores_array[y_true_array == 0], bins=bins, alpha=0.6, label="No Fraud", edgecolor="black")
    ax.hist(scores_array[y_true_array == 1], bins=bins, alpha=0.6, label="Fraud", edgecolor="black")
    ax.axvline(float(threshold), color="red", linestyle="--", linewidth=2.0, label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(_ensure_parent(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
