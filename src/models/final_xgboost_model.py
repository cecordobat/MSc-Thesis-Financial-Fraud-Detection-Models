"""Helpers to reconstruct the final XGBoost thesis model selected in notebook 4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.models.final_model_selection import predict_scores
from src.models.train_models import (
    RANDOM_STATE,
    TemporalSplitConfig,
    detect_identifier_columns,
    get_modeling_dataset_columns,
    load_temporal_split_dataset,
    summarize_temporal_split_from_parquet,
)
from src.utils.metrics import compute_classification_metrics


@dataclass(frozen=True)
class FinalXGBoostConfig:
    """Paths and methodological settings for notebook 5."""

    project_root: Path
    modeling_data_path: Path
    tables_dir: Path
    figures_dir: Path
    models_dir: Path
    selected_features_path: Path
    final_model_comparison_path: Path
    threshold_optimization_path: Path
    best_model_metadata_path: Path
    best_model_final_artifact_path: Path
    source_model_artifact_path: Path
    final_metrics_summary_path: Path
    temporal_split: TemporalSplitConfig
    target_col: str = "is_fraud"


def make_default_final_xgboost_config(project_root: Path) -> FinalXGBoostConfig:
    """Build the default configuration matching notebook 4 artifacts."""

    outputs_dir = project_root / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    models_dir = outputs_dir / "models"
    for directory in (tables_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return FinalXGBoostConfig(
        project_root=project_root,
        modeling_data_path=project_root / "data" / "processed" / "transactions_modeling.parquet",
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        selected_features_path=tables_dir / "final_selected_features.csv",
        final_model_comparison_path=tables_dir / "final_model_comparison.csv",
        threshold_optimization_path=tables_dir / "threshold_optimization_validation.csv",
        best_model_metadata_path=models_dir / "best_model_metadata.json",
        best_model_final_artifact_path=models_dir / "best_model_final.joblib",
        source_model_artifact_path=models_dir / "xgboost_top_100_scale_pos_weight_manual.joblib",
        final_metrics_summary_path=tables_dir / "final_metrics_summary.csv",
        temporal_split=TemporalSplitConfig(
            train_start="1991-01",
            train_end="2017-12",
            validation_start="2018-01",
            validation_end="2018-12",
            test_start="2019-01",
            test_end="2019-10",
            excluded_periods=("2019-11", "2019-12", "2020-01", "2020-02"),
        ),
    )


def _required_paths(config: FinalXGBoostConfig) -> dict[str, Path]:
    return {
        "modeling_dataset": config.modeling_data_path,
        "selected_features": config.selected_features_path,
        "final_model_comparison": config.final_model_comparison_path,
        "threshold_optimization": config.threshold_optimization_path,
        "best_model_metadata": config.best_model_metadata_path,
        "best_model_final_artifact": config.best_model_final_artifact_path,
        "source_model_artifact": config.source_model_artifact_path,
        "final_metrics_summary": config.final_metrics_summary_path,
    }


def validate_required_files(config: FinalXGBoostConfig) -> pd.DataFrame:
    """Return a validation table for files needed by notebook 5."""

    rows = []
    for name, path in _required_paths(config).items():
        rows.append(
            {
                "artifact_name": name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1024**2, 4) if path.exists() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_xgboost_hyperparameters(estimator: Any) -> dict[str, Any]:
    keys = [
        "objective",
        "eval_metric",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "gamma",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
        "random_state",
        "n_jobs",
        "tree_method",
        "verbosity",
    ]
    params = estimator.get_params()
    return {key: params.get(key) for key in keys}


def load_final_xgboost_inputs(config: FinalXGBoostConfig) -> dict[str, Any]:
    """Load and reconcile notebook-4 artifacts for the final XGBoost reconstruction."""

    missing = [name for name, path in _required_paths(config).items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos requeridos para notebook 5: {missing}")

    metadata = _load_json(config.best_model_metadata_path)
    best_model_final_artifact = joblib.load(config.best_model_final_artifact_path)
    source_artifact = joblib.load(config.source_model_artifact_path)
    final_model_comparison = pd.read_csv(config.final_model_comparison_path)
    selected_features_table = pd.read_csv(config.selected_features_path)
    threshold_table = pd.read_csv(config.threshold_optimization_path)
    final_metrics_summary = pd.read_csv(config.final_metrics_summary_path)

    winning_row = final_model_comparison[
        (final_model_comparison["model_name"] == "xgboost")
        & (final_model_comparison["feature_subset"] == "top_100")
        & (final_model_comparison["balancing_strategy"] == "scale_pos_weight_manual")
    ]
    if winning_row.empty:
        raise ValueError("No se encontró la fila final de XGBoost top_100 scale_pos_weight_manual.")
    winning_row = winning_row.iloc[0].to_dict()

    estimator = source_artifact["model"].named_steps["estimator"]
    hyperparameters = _extract_xgboost_hyperparameters(estimator)
    threshold_row = threshold_table.loc[threshold_table["f1"].idxmax()].to_dict()

    metadata_features = list(metadata["features"])
    ranked_top_100 = selected_features_table[selected_features_table["selected_top_100"] == True]["feature"].tolist()
    if set(metadata_features) != set(ranked_top_100):
        raise ValueError("La lista de features del metadata no coincide con final_selected_features.csv.")

    if metadata["model_name"] != "xgboost" or metadata["feature_subset"] != "top_100":
        raise ValueError("best_model_metadata.json no corresponde al XGBoost final esperado.")

    return {
        "metadata": metadata,
        "best_model_final_artifact": best_model_final_artifact,
        "source_artifact": source_artifact,
        "winning_row": winning_row,
        "selected_features_table": selected_features_table,
        "threshold_table": threshold_table,
        "best_threshold_row": threshold_row,
        "final_metrics_summary": final_metrics_summary,
        "selected_features": metadata_features,
        "hyperparameters": hyperparameters,
        "train_validation_row_cap": int(winning_row["train_rows_used"]),
        "test_rows_used": int(winning_row["test_rows_used"]),
        "threshold_final": float(metadata["threshold_final"]),
        "threshold_from_source_artifact": float(source_artifact["threshold"]),
        "random_state": int(hyperparameters["random_state"]) if hyperparameters["random_state"] is not None else RANDOM_STATE,
        "numeric_features": list(source_artifact.get("numeric_features", [])),
        "categorical_features": list(source_artifact.get("categorical_features", [])),
    }


def validate_selected_features(
    dataset_columns: list[str],
    selected_features: list[str],
    target_col: str = "is_fraud",
) -> dict[str, Any]:
    """Validate that the selected features are usable and leakage-safe."""

    missing_features = sorted(set(selected_features) - set(dataset_columns))
    duplicated_features = sorted(pd.Index(selected_features)[pd.Index(selected_features).duplicated()].unique().tolist())
    identifier_hits = detect_identifier_columns(selected_features)
    prohibited = set(identifier_hits) | {target_col}
    direct_target_hits = [feature for feature in selected_features if feature == target_col or feature.lower() == target_col.lower()]

    return {
        "n_selected_features": len(selected_features),
        "missing_features": missing_features,
        "duplicated_features": duplicated_features,
        "identifier_hits": identifier_hits,
        "direct_target_hits": direct_target_hits,
        "is_valid": not missing_features and not duplicated_features and not prohibited.intersection(selected_features),
    }


def build_split_summary(config: FinalXGBoostConfig) -> pd.DataFrame:
    """Recompute the split summary directly from parquet."""

    return summarize_temporal_split_from_parquet(
        path=config.modeling_data_path,
        split_config=config.temporal_split,
        target_col=config.target_col,
    )


def load_final_split(
    config: FinalXGBoostConfig,
    split_name: str,
    feature_columns: list[str],
    max_rows: int | None = None,
    sort_by: str | None = "datetime",
) -> pd.DataFrame:
    """Load one split keeping all positives when sampling is requested."""

    requested_columns = list(dict.fromkeys(feature_columns + [config.target_col]))
    if sort_by is not None and sort_by not in requested_columns:
        requested_columns.append(sort_by)

    return load_temporal_split_dataset(
        path=config.modeling_data_path,
        split_config=config.temporal_split,
        split_name=split_name,
        columns=requested_columns,
        max_rows=max_rows,
        target_col=config.target_col,
        keep_all_positives=True if max_rows is not None else False,
        sort_by=sort_by,
    )


def load_train_validation_frame(
    config: FinalXGBoostConfig,
    feature_columns: list[str],
    split_summary: pd.DataFrame,
    max_rows: int | None = None,
    sort_by: str | None = "datetime",
) -> pd.DataFrame:
    """Load train+validation using the same proportional caps as notebook 4."""

    caps = compute_train_validation_caps(split_summary, max_rows=max_rows)
    train_cap = caps["train_cap"]
    validation_cap = caps["validation_cap"]

    train_frame = load_final_split(config, "train", feature_columns, max_rows=train_cap, sort_by=sort_by)
    validation_frame = load_final_split(config, "validation", feature_columns, max_rows=validation_cap, sort_by=sort_by)
    return pd.concat([train_frame, validation_frame], axis=0, ignore_index=True)


def compute_train_validation_caps(split_summary: pd.DataFrame, max_rows: int | None) -> dict[str, int | None]:
    """Compute the proportional train/validation caps used in notebook 4."""

    row_counts = {row["split"]: int(row["n_rows"]) for row in split_summary.to_dict(orient="records")}
    train_rows = row_counts["train"]
    validation_rows = row_counts["validation"]
    total_rows = train_rows + validation_rows

    if max_rows is None:
        train_cap = None
        validation_cap = None
    else:
        train_cap = max(1, int(round(max_rows * train_rows / total_rows)))
        validation_cap = max(1, int(max_rows - train_cap))

    return {
        "train_cap": train_cap,
        "validation_cap": validation_cap,
        "train_rows_full": train_rows,
        "validation_rows_full": validation_rows,
        "train_validation_rows_full": total_rows,
    }


def train_final_xgboost_model(
    source_artifact: dict[str, Any],
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "is_fraud",
) -> dict[str, Any]:
    """Clone the selected pipeline, apply manual scale_pos_weight and fit it."""

    x_train = train_frame.loc[:, feature_columns].copy()
    y_train = train_frame.loc[:, target_col].copy()

    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float((y_train == 0).sum()), 1.0)
    effective_scale_pos_weight = negatives / positives

    model = clone(source_artifact["model"])
    if hasattr(model, "set_params"):
        model.set_params(estimator__scale_pos_weight=effective_scale_pos_weight)

    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time_seconds = time.perf_counter() - fit_start

    return {
        "model": model,
        "x_train": x_train,
        "y_train": y_train,
        "train_time_seconds": train_time_seconds,
        "effective_scale_pos_weight": float(effective_scale_pos_weight),
        "train_rows_used": int(len(train_frame)),
        "train_positive_rows": int(y_train.sum()),
    }


def predict_final_model(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Generate probabilities for one split and measure prediction time."""

    x_frame = frame.loc[:, feature_columns].copy()
    predict_start = time.perf_counter()
    scores = predict_scores(model, x_frame)
    predict_time_seconds = time.perf_counter() - predict_start
    return {
        "x_frame": x_frame,
        "scores": np.asarray(scores, dtype=float),
        "predict_time_seconds": predict_time_seconds,
    }


def build_final_metrics_table(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    threshold: float,
    model_name: str,
    feature_subset: str,
    balancing_strategy: str,
    number_of_features_used: int,
    train_time_seconds: float,
    predict_time_seconds: float,
) -> pd.DataFrame:
    """Build the final one-row metrics table for the thesis model."""

    metrics = compute_classification_metrics(y_true, scores, threshold=threshold)
    accuracy = float((int(metrics["tp"]) + int(metrics["tn"])) / int(metrics["n_obs"])) if int(metrics["n_obs"]) else 0.0
    row = {
        "model_name": model_name,
        "feature_subset": feature_subset,
        "balancing_strategy": balancing_strategy,
        "number_of_features_used": int(number_of_features_used),
        "threshold_used": float(threshold),
        "accuracy": accuracy,
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
        "n_obs": int(metrics["n_obs"]),
        "n_positive": int(metrics["n_positive"]),
        "positive_rate": float(metrics["positive_rate"]),
        "train_time_seconds": float(train_time_seconds),
        "predict_time_seconds": float(predict_time_seconds),
    }
    return pd.DataFrame([row])


def build_configuration_table(
    config: FinalXGBoostConfig,
    inputs: dict[str, Any],
    effective_scale_pos_weight: float,
) -> pd.DataFrame:
    """Build a one-row configuration table for export."""

    hyperparameters_json = json.dumps(inputs["hyperparameters"], ensure_ascii=False)
    return pd.DataFrame(
        [
            {
                "model_name": inputs["metadata"]["model_name"],
                "feature_subset": inputs["metadata"]["feature_subset"],
                "balancing_strategy": inputs["metadata"]["balancing_strategy"],
                "random_state": inputs["random_state"],
                "threshold_final": inputs["threshold_final"],
                "threshold_from_source_artifact": inputs["threshold_from_source_artifact"],
                "train_validation_row_cap": inputs["train_validation_row_cap"],
                "training_data_rule": inputs["metadata"]["training_data"],
                "target_col": config.target_col,
                "dataset_path": str(config.modeling_data_path),
                "source_model_artifact_path": str(config.source_model_artifact_path),
                "best_model_metadata_path": str(config.best_model_metadata_path),
                "selected_features_path": str(config.selected_features_path),
                "threshold_optimization_path": str(config.threshold_optimization_path),
                "effective_scale_pos_weight": float(effective_scale_pos_weight),
                "hyperparameters_json": hyperparameters_json,
            }
        ]
    )


def build_feature_importance_frame(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Extract XGBoost feature importances aligned to the selected feature list."""

    estimator = model.named_steps["estimator"] if hasattr(model, "named_steps") else model
    importances = np.asarray(getattr(estimator, "feature_importances_", np.zeros(len(feature_columns))), dtype=float)
    if len(importances) != len(feature_columns):
        raise ValueError("El número de importancias no coincide con la lista de features.")
    return pd.DataFrame({"feature": feature_columns, "importance": importances}).sort_values("importance", ascending=False).reset_index(drop=True)


def build_final_model_artifact(
    model: Any,
    feature_columns: list[str],
    threshold_final: float,
    hyperparameters: dict[str, Any],
    random_state: int,
    final_metrics_row: dict[str, Any],
    configuration_row: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the final joblib payload for the thesis model."""

    return {
        "model": model,
        "features": list(feature_columns),
        "threshold_final": float(threshold_final),
        "hyperparameters": dict(hyperparameters),
        "random_state": int(random_state),
        "execution_timestamp": datetime.now().isoformat(),
        "final_metrics": dict(final_metrics_row),
        "configuration": dict(configuration_row),
    }


def get_dataset_schema(path: Path) -> list[str]:
    """Expose the modeling dataset schema to notebooks."""

    return get_modeling_dataset_columns(path)
