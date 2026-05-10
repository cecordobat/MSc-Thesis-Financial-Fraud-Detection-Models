"""Notebook-4 final model selection workflow extracted into Python code."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone

from src.models.train_models import RANDOM_STATE, TemporalSplitConfig, load_temporal_split_dataset
from src.utils.metrics import build_threshold_table, compute_classification_metrics, get_predicted_labels, select_threshold
from src.utils.notebook_common import append_suffix, save_table_with_optional_excel
from src.utils.plotting import plot_precision_recall_curve, plot_roc_curve


@dataclass(frozen=True)
class FinalSelectionConfig:
    """Configuration for notebook 4."""

    modeling_data_path: Path
    model_results_path: Path
    feature_ranking_path: Path
    tables_dir: Path
    figures_dir: Path
    models_dir: Path
    outputs_dir: Path
    temporal_split: TemporalSplitConfig
    save_outputs: bool = True
    output_suffix: str = ""
    supported_models: set[str] = field(default_factory=set)
    retrain_sample_limits: dict[str, int | None] = field(default_factory=dict)
    test_sample_limits: dict[str, int | None] = field(default_factory=dict)


def make_default_final_selection_config(
    project_root: Path,
    sample_run: bool = False,
    save_outputs: bool = True,
    output_suffix: str = "",
) -> FinalSelectionConfig:
    """Build the default notebook-4 configuration."""

    outputs_dir = project_root / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    models_dir = outputs_dir / "models"
    for directory in (outputs_dir, tables_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    retrain_limits = {
        "xgboost_calibrated": 150_000,
        "xgboost": 250_000,
        "mlp_classifier_light": 180_000,
        "mlp_classifier": 180_000,
        "bagging_tree": 200_000,
        "gradient_boosting": 180_000,
        "hist_gradient_boosting": 250_000,
        "random_forest": 220_000,
        "extra_trees": 220_000,
        "decision_tree": 300_000,
        "adaboost": 180_000,
        "logistic_regression": 400_000,
        "gaussian_nb": 300_000,
        "sgd_classifier": 400_000,
        "linear_svc": 250_000,
    }
    if sample_run:
        retrain_limits = {key: min(value or 50_000, 50_000) if value is not None else 50_000 for key, value in retrain_limits.items()}

    return FinalSelectionConfig(
        modeling_data_path=project_root / "data" / "processed" / "transactions_modeling.parquet",
        model_results_path=append_suffix(tables_dir / "model_experiment_results.csv", output_suffix),
        feature_ranking_path=append_suffix(tables_dir / "feature_ranking.csv", output_suffix),
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        outputs_dir=outputs_dir,
        temporal_split=TemporalSplitConfig(
            train_start="1991-01",
            train_end="2017-12",
            validation_start="2018-01",
            validation_end="2018-12",
            test_start="2019-01",
            test_end="2019-10",
            excluded_periods=("2019-11", "2019-12", "2020-01", "2020-02"),
        ),
        save_outputs=save_outputs,
        output_suffix=output_suffix,
        supported_models={
            "xgboost_calibrated",
            "xgboost",
            "mlp_classifier_light",
            "mlp_classifier",
            "bagging_tree",
            "gradient_boosting",
            "hist_gradient_boosting",
            "random_forest",
            "extra_trees",
            "decision_tree",
            "adaboost",
            "logistic_regression",
            "gaussian_nb",
            "sgd_classifier",
            "linear_svc",
        },
        retrain_sample_limits=retrain_limits,
        test_sample_limits={"svc_rbf": 60_000},
    )


def _table_path(config: FinalSelectionConfig, filename: str) -> Path:
    return append_suffix(config.tables_dir / filename, config.output_suffix)


def _figure_path(config: FinalSelectionConfig, filename: str) -> Path:
    return append_suffix(config.figures_dir / filename, config.output_suffix)


def _model_path(config: FinalSelectionConfig, filename: str) -> Path:
    return append_suffix(config.models_dir / filename, config.output_suffix)


def _output_path(config: FinalSelectionConfig, filename: str) -> Path:
    return append_suffix(config.outputs_dir / filename, config.output_suffix)


def load_model_safe(model_path: str) -> Any:
    """Safely load a joblib model artifact."""

    try:
        return joblib.load(Path(model_path))
    except Exception:
        return None


def reconstruct_model_pipeline(artifact: Any) -> Any:
    """Extract the trainable estimator from a saved artifact."""

    if isinstance(artifact, dict):
        if "model" in artifact:
            return artifact["model"]
        if "classifier" in artifact:
            return artifact["classifier"]
    return artifact


def predict_scores(estimator: Any, x_frame: pd.DataFrame) -> np.ndarray:
    """Return continuous scores for a fitted estimator."""

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_frame)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw_scores = estimator.decision_function(x_frame)
        return 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=float)))
    return np.asarray(estimator.predict(x_frame), dtype=float)


def _clone_estimator(estimator: Any) -> Any:
    """Clone a fitted estimator for retraining."""

    try:
        return clone(estimator)
    except Exception:
        return deepcopy(estimator)


def _get_requested_retrain_rows(config: FinalSelectionConfig, model_info: dict[str, Any]) -> int | None:
    requested = model_info.get("max_train_rows_requested")
    if pd.notna(requested):
        return int(requested)
    return config.retrain_sample_limits.get(model_info["model_name"])


def _load_split_frame(
    config: FinalSelectionConfig,
    split_name: str,
    feature_columns: list[str],
    target_col: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    requested_columns = list(dict.fromkeys(feature_columns + [target_col]))
    return load_temporal_split_dataset(
        path=config.modeling_data_path,
        split_config=config.temporal_split,
        split_name=split_name,
        columns=requested_columns,
        max_rows=max_rows,
        target_col=target_col,
        keep_all_positives=True,
        sort_by=None,
    )


def _load_train_validation_frame(
    config: FinalSelectionConfig,
    feature_columns: list[str],
    target_col: str,
    split_row_counts: dict[str, int],
    max_rows: int | None = None,
) -> pd.DataFrame:
    train_rows = split_row_counts["train"]
    validation_rows = split_row_counts["validation"]
    total_rows = train_rows + validation_rows

    if max_rows is None:
        train_cap = None
        validation_cap = None
    else:
        train_cap = max(1, int(round(max_rows * train_rows / total_rows)))
        validation_cap = max(1, max_rows - train_cap)

    train_frame = _load_split_frame(config, "train", feature_columns, target_col, max_rows=train_cap)
    validation_frame = _load_split_frame(config, "validation", feature_columns, target_col, max_rows=validation_cap)
    return pd.concat([train_frame, validation_frame], axis=0, ignore_index=True)


def _select_top_models(config: FinalSelectionConfig, results_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the top retrainable validation models."""

    validation_frame = results_frame[results_frame["split"] == "validation"].copy()
    validation_frame["artifact_suffix"] = validation_frame["model_artifact_path"].astype(str).str.lower().str.extract(r"(\.[a-z0-9]+)$")[0]
    validation_frame["retrain_supported"] = (
        validation_frame["model_name"].isin(config.supported_models) & validation_frame["artifact_suffix"].eq(".joblib")
    )
    validation_frame["model_config"] = (
        validation_frame["model_name"] + "_" + validation_frame["feature_subset"] + "_" + validation_frame["balancing_strategy"]
    )
    validation_frame = validation_frame.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    unsupported_high_rank = validation_frame[~validation_frame["retrain_supported"]].head(10)
    top_3 = (
        validation_frame[validation_frame["retrain_supported"]][
            [
                "model_name",
                "feature_subset",
                "balancing_strategy",
                "pr_auc",
                "recall",
                "f1",
                "precision",
                "roc_auc",
                "model_artifact_path",
                "number_of_features_used",
                "train_time_seconds",
                "max_train_rows_requested",
                "max_validation_rows_requested",
            ]
        ]
        .drop_duplicates(subset=["model_name", "feature_subset", "balancing_strategy"])
        .head(3)
        .reset_index(drop=True)
    )
    if top_3.empty:
        raise RuntimeError("No hay modelos compatibles para reentrenamiento final.")
    return top_3, unsupported_high_rank


def _retrain_top_models(
    config: FinalSelectionConfig,
    top_models: pd.DataFrame,
    split_row_counts: dict[str, int],
    target_col: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retrain the selected models on train+validation and evaluate on test."""

    retrained_models: dict[str, Any] = {}
    retrained_results: list[dict[str, Any]] = []

    for rank, model_info in enumerate(top_models.to_dict(orient="records"), start=1):
        artifact = load_model_safe(model_info["model_artifact_path"])
        if artifact is None:
            continue

        estimator = reconstruct_model_pipeline(artifact)
        feature_columns = list(artifact["feature_columns"])
        threshold_from_validation = float(artifact.get("threshold", 0.5))
        model_key = f"{model_info['model_name']}_{model_info['feature_subset']}_{model_info['balancing_strategy']}"

        train_validation_frame = _load_train_validation_frame(
            config,
            feature_columns,
            target_col,
            split_row_counts=split_row_counts,
            max_rows=_get_requested_retrain_rows(config, model_info),
        )
        test_frame = _load_split_frame(
            config,
            "test",
            feature_columns,
            target_col,
            max_rows=config.test_sample_limits.get(model_info["model_name"]),
        )

        x_train = train_validation_frame[feature_columns]
        y_train = train_validation_frame[target_col].copy()
        x_test = test_frame[feature_columns]
        y_test = test_frame[target_col].copy()

        fitted_model = _clone_estimator(estimator)
        fitted_model.fit(x_train, y_train)
        test_scores = predict_scores(fitted_model, x_test)
        test_metrics = compute_classification_metrics(y_test, test_scores, threshold=threshold_from_validation)

        result_row = {
            "rank": rank,
            "model_key": model_key,
            "model_name": model_info["model_name"],
            "feature_subset": model_info["feature_subset"],
            "balancing_strategy": model_info["balancing_strategy"],
            "n_features": len(feature_columns),
            "train_rows_used": int(len(train_validation_frame)),
            "test_rows_used": int(len(test_frame)),
            "threshold_from_validation": threshold_from_validation,
            "pr_auc_test": float(test_metrics["pr_auc"]),
            "roc_auc_test": float(test_metrics["roc_auc"]),
            "recall_test": float(test_metrics["recall"]),
            "precision_test": float(test_metrics["precision"]),
            "f1_test": float(test_metrics["f1"]),
            "balanced_accuracy_test": float(test_metrics["balanced_accuracy"]),
            "model_artifact_path": model_info["model_artifact_path"],
        }
        retrained_results.append(result_row)
        retrained_models[model_key] = {
            "model": fitted_model,
            "features": feature_columns,
            "y_test": y_test,
            "y_test_pred_proba": np.asarray(test_scores, dtype=float),
            "source_artifact": artifact,
        }

    if not retrained_results:
        raise RuntimeError("No fue posible reentrenar ningun modelo finalista.")
    return retrained_models, retrained_results


def _save_probability_distribution(scores: np.ndarray, y_true: pd.Series, threshold: float, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(scores[y_true == 0], bins=60, alpha=0.6, label="No Fraud", edgecolor="black")
    ax.hist(scores[y_true == 1], bins=60, alpha=0.6, label="Fraud", edgecolor="black")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2.0, label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_heatmap(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    cm = pd.crosstab(pd.Series(y_true, name="True"), pd.Series(y_pred, name="Pred"), dropna=False).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm.to_numpy(),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No Fraud", "Fraud"],
        yticklabels=["No Fraud", "Fraud"],
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_final_model_selection(config: FinalSelectionConfig) -> dict[str, Any]:
    """Run the notebook-4 final model-selection workflow."""

    if not config.modeling_data_path.exists():
        raise FileNotFoundError(f"No existe el dataset de modelado: {config.modeling_data_path}")
    if not config.model_results_path.exists():
        raise FileNotFoundError(f"No existe la tabla de resultados: {config.model_results_path}")
    if not config.feature_ranking_path.exists():
        raise FileNotFoundError(f"No existe el ranking de features: {config.feature_ranking_path}")

    df_results = pd.read_csv(config.model_results_path)
    df_ranking = pd.read_csv(config.feature_ranking_path)
    split_summary = pd.read_csv(_table_path(config, "split_summary.csv")) if _table_path(config, "split_summary.csv").exists() else None
    if split_summary is None:
        raise FileNotFoundError("No existe split_summary.csv para reconstruir el split temporal.")
    split_row_counts = {row["split"]: int(row["n_rows"]) for row in split_summary.to_dict(orient="records")}
    target_col = "is_fraud"

    top_3, unsupported_high_rank = _select_top_models(config, df_results)
    retrained_models, retrained_results = _retrain_top_models(config, top_3, split_row_counts, target_col)
    df_retrained_results = pd.DataFrame(retrained_results).sort_values("pr_auc_test", ascending=False).reset_index(drop=True)

    best_model_row = df_retrained_results.iloc[0]
    best_model_key = best_model_row["model_key"]
    best_model_bundle = retrained_models[best_model_key]
    best_model = best_model_bundle["model"]
    best_features = best_model_bundle["features"]
    best_y_pred_proba = best_model_bundle["y_test_pred_proba"]
    y_test = best_model_bundle["y_test"]

    validation_frame = _load_split_frame(config, "validation", best_features, target_col, max_rows=None)
    x_validation = validation_frame[best_features]
    y_validation = validation_frame[target_col].copy()
    validation_scores = predict_scores(best_model, x_validation)
    threshold_result = select_threshold(build_threshold_table(y_validation, validation_scores), criterion="f1")
    best_threshold = float(threshold_result.threshold)
    df_thresholds = threshold_result.threshold_table.copy()

    y_test_pred = get_predicted_labels(best_y_pred_proba, threshold=best_threshold)
    final_metrics = compute_classification_metrics(y_test, best_y_pred_proba, threshold=best_threshold)
    final_precision = float(final_metrics["precision"])
    final_recall = float(final_metrics["recall"])
    final_f1 = float(final_metrics["f1"])
    final_pr_auc = float(final_metrics["pr_auc"])
    final_roc_auc = float(final_metrics["roc_auc"])
    tn = int(final_metrics["tn"])
    fp = int(final_metrics["fp"])
    fn = int(final_metrics["fn"])
    tp = int(final_metrics["tp"])

    df_final_features = df_ranking[df_ranking["feature"].isin(best_features)].copy()
    if "rank_average" in df_final_features.columns:
        df_final_features = df_final_features.sort_values("rank_average")

    final_summary = pd.DataFrame(
        [
            {"Métrica": "PR-AUC", "Valor": f"{final_pr_auc:.6f}", "Interpretación": "Área bajo la curva Precisión-Recall"},
            {"Métrica": "ROC-AUC", "Valor": f"{final_roc_auc:.6f}", "Interpretación": "Área bajo la curva ROC"},
            {
                "Métrica": "Precision",
                "Valor": f"{final_precision:.6f}",
                "Interpretación": f"De los {tp + fp:,} casos predichos como fraude, {tp:,} fueron reales",
            },
            {
                "Métrica": "Recall",
                "Valor": f"{final_recall:.6f}",
                "Interpretación": f"Detectamos {tp:,} de {tp + fn:,} fraudes reales ({final_recall * 100:.2f}%)",
            },
            {"Métrica": "F1-Score", "Valor": f"{final_f1:.6f}", "Interpretación": "Media armónica de Precision y Recall"},
            {
                "Métrica": "Accuracy",
                "Valor": f"{(tp + tn) / (tp + tn + fp + fn):.6f}",
                "Interpretación": "Proporción de predicciones correctas",
            },
        ]
    )

    summary_text = f"""
MODELO SELECCIONADO
- Nombre: {best_model_row['model_name']}
- Subset: {best_model_row['feature_subset']}
- Balanceo: {best_model_row['balancing_strategy']}
- Threshold final: {best_threshold:.3f}

RENDIMIENTO EN TEST
- PR-AUC: {final_pr_auc:.4f}
- ROC-AUC: {final_roc_auc:.4f}
- Recall: {final_recall:.4f}
- Precision: {final_precision:.4f}
- F1-Score: {final_f1:.4f}

MATRIZ DE CONFUSIÓN
- TN: {tn:,}
- FP: {fp:,}
- FN: {fn:,}
- TP: {tp:,}
""".strip()

    if config.save_outputs:
        save_table_with_optional_excel(
            df_retrained_results,
            _table_path(config, "final_model_comparison.csv"),
            _table_path(config, "final_model_comparison.xlsx"),
        )
        save_table_with_optional_excel(
            df_thresholds,
            _table_path(config, "threshold_optimization_validation.csv"),
            _table_path(config, "threshold_optimization_validation.xlsx"),
        )
        save_table_with_optional_excel(
            df_final_features,
            _table_path(config, "final_selected_features.csv"),
            _table_path(config, "final_selected_features.xlsx"),
        )
        save_table_with_optional_excel(
            final_summary,
            _table_path(config, "final_metrics_summary.csv"),
            _table_path(config, "final_metrics_summary.xlsx"),
        )

        best_model_path = _model_path(config, "best_model_final.joblib")
        joblib.dump(
            {
                "model": best_model,
                "features": best_features,
                "threshold_final": best_threshold,
                "selection_row": best_model_row.to_dict(),
            },
            best_model_path,
        )
        metadata_path = _model_path(config, "best_model_metadata.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "model_name": best_model_row["model_name"],
                    "feature_subset": best_model_row["feature_subset"],
                    "balancing_strategy": best_model_row["balancing_strategy"],
                    "n_features": len(best_features),
                    "features": best_features,
                    "threshold_final": best_threshold,
                    "training_data": "train + validation",
                    "pr_auc_test": final_pr_auc,
                    "roc_auc_test": final_roc_auc,
                    "recall_test": final_recall,
                    "precision_test": final_precision,
                    "f1_test": final_f1,
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        _save_confusion_heatmap(
            y_test,
            y_test_pred,
            _figure_path(config, "confusion_matrix_best_model.png"),
            f"Matriz de Confusión - {best_model_key}\n(Threshold={best_threshold:.2f})",
        )
        plot_precision_recall_curve(
            y_test,
            best_y_pred_proba,
            model_name=best_model_key,
            output_path=_figure_path(config, "precision_recall_curve_best_model.png"),
        )
        plot_roc_curve(
            y_test,
            best_y_pred_proba,
            model_name=best_model_key,
            output_path=_figure_path(config, "roc_curve_best_model.png"),
        )
        _save_probability_distribution(
            best_y_pred_proba,
            y_test,
            best_threshold,
            _figure_path(config, "probability_distribution_best_model.png"),
            f"Distribution of Predicted Probabilities - {best_model_key}",
        )
        _output_path(config, "NOTEBOOK4_SUMMARY.txt").write_text(summary_text, encoding="utf-8")

    generated_files = {
        "tables": [
            _table_path(config, "final_model_comparison.csv"),
            _table_path(config, "threshold_optimization_validation.csv"),
            _table_path(config, "final_selected_features.csv"),
            _table_path(config, "final_metrics_summary.csv"),
        ],
        "models": [
            _model_path(config, "best_model_final.joblib"),
            _model_path(config, "best_model_metadata.json"),
        ],
        "figures": [
            _figure_path(config, "confusion_matrix_best_model.png"),
            _figure_path(config, "precision_recall_curve_best_model.png"),
            _figure_path(config, "roc_curve_best_model.png"),
            _figure_path(config, "probability_distribution_best_model.png"),
        ],
    }

    return {
        "unsupported_high_rank": unsupported_high_rank,
        "top_3": top_3,
        "df_retrained_results": df_retrained_results,
        "best_model_row": best_model_row,
        "best_threshold": best_threshold,
        "df_thresholds": df_thresholds,
        "df_final_features": df_final_features,
        "final_summary": final_summary,
        "summary_text": summary_text,
        "generated_files": generated_files,
    }
