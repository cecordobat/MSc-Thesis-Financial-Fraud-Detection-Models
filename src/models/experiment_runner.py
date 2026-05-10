"""Notebook-3 modeling workflow extracted into reusable Python code."""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, OneClassSVM, SVC
from sklearn.tree import DecisionTreeClassifier

from src.models.evaluate_models import (
    build_metrics_frame,
    evaluate_classifier,
    evaluate_validation_thresholds,
    get_positive_class_scores,
    save_metrics_table,
    save_threshold_table,
)
from src.models.feature_selection import (
    build_feature_ranking_table,
    create_feature_subsets,
    filter_constant_features,
    filter_correlated_features,
    rank_features_logistic_coefficients,
    rank_features_mutual_information,
    rank_features_permutation_importance,
    rank_features_tree_importance,
    save_feature_ranking,
)
from src.models.train_baseline import build_decision_tree_baseline, build_dummy_baseline, build_logistic_baseline
from src.models.train_models import (
    RANDOM_STATE,
    TemporalSplitConfig,
    build_candidate_feature_columns,
    build_hist_gradient_boosting_pipeline,
    build_preprocessor,
    build_random_forest_pipeline,
    build_random_undersample_logistic_pipeline,
    build_smote_logistic_pipeline,
    detect_identifier_columns,
    detect_target_column,
    get_modeling_dataset_columns,
    load_modeling_dataset_preview,
    load_temporal_split_dataset,
    make_estimator_pipeline,
    split_features_target,
    summarize_temporal_split_from_parquet,
    summarize_temporal_split_monthly_from_parquet,
    validate_modeling_dataset,
)
from src.utils.metrics import build_threshold_table, compute_classification_metrics, get_predicted_labels, select_threshold
from src.utils.notebook_common import append_suffix, save_table_with_optional_excel
from src.utils.plotting import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve


AVAILABLE_LIBRARIES = {
    "xgboost": importlib.util.find_spec("xgboost") is not None,
    "openpyxl": importlib.util.find_spec("openpyxl") is not None,
}

if AVAILABLE_LIBRARIES["xgboost"]:
    from xgboost import XGBClassifier


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for notebook 3."""

    dataset_path: Path
    tables_dir: Path
    figures_dir: Path
    models_dir: Path
    temporal_split: TemporalSplitConfig
    threshold_criterion: str = "f1"
    ranking_subset_sizes: tuple[int, ...] = (25, 50, 75, 100)
    dataset_preview_rows: int = 5
    in_memory_train_rows: int = 500_000
    in_memory_validation_rows: int = 200_000
    ranking_max_train_rows: int = 300_000
    ranking_max_validation_rows: int = 150_000
    feature_metadata_max_rows: int = 250_000
    correlation_sample_size: int = 200_000
    mi_sample_size: int = 200_000
    logistic_ranking_sample_size: int = 150_000
    tree_ranking_sample_size: int = 150_000
    permutation_max_train_rows: int = 200_000
    permutation_max_validation_rows: int = 120_000
    run_optional_heavy_models: bool = True
    run_anomaly_models: bool = True
    run_optional_deep_models: bool = True
    save_outputs: bool = True
    output_suffix: str = ""
    model_sample_limits: dict[str, dict[str, int | None]] = field(default_factory=dict)


def make_default_experiment_config(
    project_root: Path,
    sample_run: bool = False,
    save_outputs: bool = True,
    output_suffix: str = "",
) -> ExperimentConfig:
    """Build the default notebook-3 configuration."""

    outputs_dir = project_root / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    models_dir = outputs_dir / "models"
    for directory in (outputs_dir, tables_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base_limits = {
        "dummy_classifier": {"max_train_rows": None, "max_validation_rows": None},
        "logistic_regression": {"max_train_rows": None, "max_validation_rows": None},
        "logistic_regression_undersample": {"max_train_rows": 500_000, "max_validation_rows": None},
        "logistic_regression_smote": {"max_train_rows": 300_000, "max_validation_rows": 200_000},
        "decision_tree": {"max_train_rows": None, "max_validation_rows": None},
        "random_forest": {"max_train_rows": None, "max_validation_rows": None},
        "extra_trees": {"max_train_rows": None, "max_validation_rows": None},
        "hist_gradient_boosting": {"max_train_rows": None, "max_validation_rows": None},
        "gradient_boosting": {"max_train_rows": 250_000, "max_validation_rows": 150_000},
        "adaboost": {"max_train_rows": 250_000, "max_validation_rows": 150_000},
        "bagging_tree": {"max_train_rows": 250_000, "max_validation_rows": 150_000},
        "xgboost": {"max_train_rows": 500_000, "max_validation_rows": 200_000},
        "sgd_classifier": {"max_train_rows": None, "max_validation_rows": None},
        "linear_svc": {"max_train_rows": None, "max_validation_rows": None},
        "svc_rbf": {"max_train_rows": 60_000, "max_validation_rows": 40_000},
        "knn": {"max_train_rows": 120_000, "max_validation_rows": 60_000},
        "gaussian_nb": {"max_train_rows": None, "max_validation_rows": None},
        "bernoulli_nb": {"max_train_rows": None, "max_validation_rows": None},
        "complement_nb": {"max_train_rows": None, "max_validation_rows": None},
        "mlp_classifier": {"max_train_rows": 350_000, "max_validation_rows": 200_000},
        "isolation_forest": {"max_train_rows": 250_000, "max_validation_rows": 150_000},
        "local_outlier_factor": {"max_train_rows": 60_000, "max_validation_rows": 40_000},
        "one_class_svm": {"max_train_rows": 50_000, "max_validation_rows": 30_000},
        "xgboost_calibrated": {"max_train_rows": 180_000, "max_validation_rows": 120_000},
        "mlp_classifier_light": {"max_train_rows": 220_000, "max_validation_rows": 120_000},
    }

    if sample_run:
        base_limits |= {
            "dummy_classifier": {"max_train_rows": 40_000, "max_validation_rows": 20_000},
            "logistic_regression": {"max_train_rows": 50_000, "max_validation_rows": 20_000},
            "logistic_regression_undersample": {"max_train_rows": 40_000, "max_validation_rows": 20_000},
            "logistic_regression_smote": {"max_train_rows": 30_000, "max_validation_rows": 15_000},
            "decision_tree": {"max_train_rows": 60_000, "max_validation_rows": 20_000},
            "random_forest": {"max_train_rows": 60_000, "max_validation_rows": 20_000},
            "extra_trees": {"max_train_rows": 60_000, "max_validation_rows": 20_000},
            "hist_gradient_boosting": {"max_train_rows": 60_000, "max_validation_rows": 20_000},
            "gradient_boosting": {"max_train_rows": 40_000, "max_validation_rows": 15_000},
            "adaboost": {"max_train_rows": 40_000, "max_validation_rows": 15_000},
            "bagging_tree": {"max_train_rows": 40_000, "max_validation_rows": 15_000},
            "xgboost": {"max_train_rows": 50_000, "max_validation_rows": 20_000},
            "sgd_classifier": {"max_train_rows": 50_000, "max_validation_rows": 20_000},
            "linear_svc": {"max_train_rows": 50_000, "max_validation_rows": 20_000},
            "gaussian_nb": {"max_train_rows": 50_000, "max_validation_rows": 20_000},
            "mlp_classifier": {"max_train_rows": 40_000, "max_validation_rows": 15_000},
            "xgboost_calibrated": {"max_train_rows": 40_000, "max_validation_rows": 15_000},
            "mlp_classifier_light": {"max_train_rows": 40_000, "max_validation_rows": 15_000},
        }

    return ExperimentConfig(
        dataset_path=project_root / "data" / "processed" / "transactions_modeling.parquet",
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        temporal_split=TemporalSplitConfig(
            train_start="1991-01",
            train_end="2017-12",
            validation_start="2018-01",
            validation_end="2018-12",
            test_start="2019-01",
            test_end="2019-10",
            excluded_periods=("2019-11", "2019-12", "2020-01", "2020-02"),
        ),
        ranking_subset_sizes=(25, 50, 75, 100),
        in_memory_train_rows=80_000 if sample_run else 500_000,
        in_memory_validation_rows=30_000 if sample_run else 200_000,
        ranking_max_train_rows=60_000 if sample_run else 300_000,
        ranking_max_validation_rows=25_000 if sample_run else 150_000,
        feature_metadata_max_rows=80_000 if sample_run else 250_000,
        correlation_sample_size=50_000 if sample_run else 200_000,
        mi_sample_size=50_000 if sample_run else 200_000,
        logistic_ranking_sample_size=40_000 if sample_run else 150_000,
        tree_ranking_sample_size=40_000 if sample_run else 150_000,
        permutation_max_train_rows=40_000 if sample_run else 200_000,
        permutation_max_validation_rows=20_000 if sample_run else 120_000,
        run_optional_heavy_models=not sample_run,
        run_anomaly_models=not sample_run,
        run_optional_deep_models=not sample_run,
        save_outputs=save_outputs,
        output_suffix=output_suffix,
        model_sample_limits=base_limits,
    )


def _table_path(config: ExperimentConfig, filename: str) -> Path:
    return append_suffix(config.tables_dir / filename, config.output_suffix)


def _figure_path(config: ExperimentConfig, filename: str) -> Path:
    return append_suffix(config.figures_dir / filename, config.output_suffix)


def _model_path(config: ExperimentConfig, filename: str) -> Path:
    return append_suffix(config.models_dir / filename, config.output_suffix)


def deduplicate_keep_order(values: Sequence[str]) -> list[str]:
    """Remove duplicates while preserving order."""

    return list(dict.fromkeys(values))


def sample_frame_preserving_positives(
    frame: pd.DataFrame,
    target_column: str,
    max_rows: int | None,
    keep_all_positives: bool = True,
) -> pd.DataFrame:
    """Sample deterministically while preserving positives when requested."""

    if max_rows is None or len(frame) <= max_rows:
        return frame.copy()

    if not keep_all_positives:
        return frame.sample(n=max_rows, random_state=RANDOM_STATE).sort_index().copy()

    positives = frame.loc[frame[target_column] == 1]
    negatives = frame.loc[frame[target_column] == 0]
    if len(positives) >= max_rows:
        return positives.sample(n=max_rows, random_state=RANDOM_STATE).sort_index().copy()

    negative_quota = min(max_rows - len(positives), len(negatives))
    sampled_negatives = negatives.sample(n=negative_quota, random_state=RANDOM_STATE)
    return pd.concat([positives, sampled_negatives], axis=0).sort_index().copy()


def sample_negative_training_frame(frame: pd.DataFrame, target_column: str, max_rows: int | None) -> pd.DataFrame:
    """Sample only negative rows for anomaly-detection training."""

    negative_frame = frame.loc[frame[target_column] == 0]
    if max_rows is None or len(negative_frame) <= max_rows:
        return negative_frame.copy()
    return negative_frame.sample(n=max_rows, random_state=RANDOM_STATE).sort_index().copy()


def build_model_filename(model_name: str, subset_name: str, balancing_strategy: str) -> str:
    """Create a filesystem-safe artifact name."""

    raw_name = f"{model_name}_{subset_name}_{balancing_strategy}".lower()
    safe_name = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw_name)
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    return f"{safe_name}.joblib"


def plot_metric_comparison(results_frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Create the same metric comparison chart used by notebook 3."""

    if results_frame.empty or metric not in results_frame.columns:
        return

    plot_frame = results_frame.copy()
    plot_frame["model_label"] = (
        plot_frame["model_name"] + " | " + plot_frame["feature_subset"] + " | " + plot_frame["balancing_strategy"]
    )
    plot_frame = plot_frame.sort_values(metric, ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(plot_frame))))
    sns.barplot(data=plot_frame, x=metric, y="model_label", ax=ax, color="#1f77b4")
    ax.set_title(f"Validation {metric} comparison")
    ax.set_xlabel(metric)
    ax.set_ylabel("model | subset | balancing")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_generic_pipeline(
    estimator: object,
    numeric_subset: Sequence[str],
    categorical_subset: Sequence[str],
    scale_numeric: bool,
) -> object:
    """Build a generic preprocessing pipeline."""

    preprocessor = build_preprocessor(
        numeric_features=numeric_subset,
        categorical_features=categorical_subset,
        scale_numeric=scale_numeric,
    )
    return make_estimator_pipeline(preprocessor, estimator)


def get_anomaly_scores(model: object, x_frame: pd.DataFrame | np.ndarray) -> pd.Series:
    """Convert anomaly-model outputs into fraud-likelihood scores."""

    if hasattr(model, "score_samples"):
        scores = -model.score_samples(x_frame)
    elif hasattr(model, "decision_function"):
        scores = -model.decision_function(x_frame)
    else:
        predictions = model.predict(x_frame)
        scores = (pd.Series(predictions) == -1).astype(float)
        return scores.rename("score")
    return pd.Series(scores, name="score")


def _save_threshold_artifact(config: ExperimentConfig, model_key: str, threshold_table: pd.DataFrame) -> None:
    """Persist threshold optimization tables."""

    if not config.save_outputs:
        return
    save_threshold_table(
        threshold_table,
        csv_path=_table_path(config, f"{model_key}_thresholds.csv"),
        excel_path=_table_path(config, f"{model_key}_thresholds.xlsx"),
    )


def _artifact_payload(
    model: object,
    feature_subset: list[str],
    numeric_subset: list[str],
    categorical_subset: list[str],
    threshold_used: float,
    spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "feature_columns": feature_subset,
        "numeric_features": numeric_subset,
        "categorical_features": categorical_subset,
        "threshold": float(threshold_used),
        "model_name": spec["model_name"],
        "balancing_strategy": spec["balancing_strategy"],
        "feature_subset": spec["current_subset_name"],
        "experiment_family": spec["model_family"],
        "max_train_rows_requested": spec.get("max_train_rows"),
        "max_validation_rows_requested": spec.get("max_validation_rows"),
    }


def fit_and_evaluate_supervised_experiment(
    spec: dict[str, Any],
    feature_subset_name: str,
    feature_subset: list[str],
    split_data: dict[str, pd.DataFrame],
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Fit one supervised experiment and return a flat metrics record."""

    train_sample = sample_frame_preserving_positives(
        split_data["train"], target_column=target_column, max_rows=spec.get("max_train_rows"), keep_all_positives=True
    )
    validation_sample = sample_frame_preserving_positives(
        split_data["validation"],
        target_column=target_column,
        max_rows=spec.get("max_validation_rows"),
        keep_all_positives=True,
    )

    numeric_subset = [column for column in feature_subset if column in numeric_features]
    categorical_subset = [column for column in feature_subset if column in categorical_features]

    x_train, y_train = split_features_target(train_sample, target_col=target_column, feature_cols=feature_subset)
    x_validation, y_validation = split_features_target(validation_sample, target_col=target_column, feature_cols=feature_subset)

    model = spec["builder"](numeric_subset, categorical_subset)
    if spec["model_name"] == "xgboost" and hasattr(model, "set_params"):
        positives = max(float(y_train.sum()), 1.0)
        negatives = max(float((y_train == 0).sum()), 1.0)
        model.set_params(estimator__scale_pos_weight=negatives / positives)

    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    threshold_result = evaluate_validation_thresholds(model, x_validation, y_validation, criterion=config.threshold_criterion)
    threshold_used = float(threshold_result.threshold)
    metric_record, scores = evaluate_classifier(
        model,
        x_frame=x_validation,
        y_true=y_validation,
        split_name="validation",
        model_name=spec["model_name"],
        subset_name=feature_subset_name,
        threshold=threshold_used,
    )
    predict_time_seconds = time.perf_counter() - predict_start

    y_pred = get_predicted_labels(scores, threshold=threshold_used)
    classification_report_json = json.dumps(
        classification_report(y_validation, y_pred, output_dict=True, zero_division=0), ensure_ascii=False
    )

    spec = {**spec, "current_subset_name": feature_subset_name}
    artifact_path = _model_path(config, build_model_filename(spec["model_name"], feature_subset_name, spec["balancing_strategy"]))
    if config.save_outputs:
        joblib.dump(_artifact_payload(model, feature_subset, numeric_subset, categorical_subset, threshold_used, spec), artifact_path)
        model_key = artifact_path.stem
        _save_threshold_artifact(config, model_key, threshold_result.threshold_table)

    metric_record.update(
        {
            "model_family": spec["model_family"],
            "balancing_strategy": spec["balancing_strategy"],
            "feature_subset": feature_subset_name,
            "number_of_features_used": len(feature_subset),
            "train_rows_used": int(len(train_sample)),
            "validation_rows_used": int(len(validation_sample)),
            "train_time_seconds": float(train_time_seconds),
            "predict_time_seconds": float(predict_time_seconds),
            "threshold_used": threshold_used,
            "classification_report_json": classification_report_json,
            "model_artifact_path": str(artifact_path),
            "max_train_rows_requested": spec.get("max_train_rows"),
            "max_validation_rows_requested": spec.get("max_validation_rows"),
        }
    )
    return metric_record


def fit_and_evaluate_anomaly_experiment(
    spec: dict[str, Any],
    feature_subset_name: str,
    feature_subset: list[str],
    split_data: dict[str, pd.DataFrame],
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Fit one anomaly-detection experiment and return a flat metrics record."""

    train_negative_sample = sample_negative_training_frame(
        split_data["train"], target_column=target_column, max_rows=spec.get("max_train_rows")
    )
    validation_sample = sample_frame_preserving_positives(
        split_data["validation"],
        target_column=target_column,
        max_rows=spec.get("max_validation_rows"),
        keep_all_positives=True,
    )

    numeric_subset = [column for column in feature_subset if column in numeric_features]
    categorical_subset = [column for column in feature_subset if column in categorical_features]
    x_train = train_negative_sample.loc[:, feature_subset].copy()
    x_validation = validation_sample.loc[:, feature_subset].copy()
    y_validation = validation_sample[target_column].copy()

    preprocessor = build_preprocessor(
        numeric_features=numeric_subset,
        categorical_features=categorical_subset,
        scale_numeric=spec.get("scale_numeric", True),
    )
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_validation_transformed = preprocessor.transform(x_validation)

    model = spec["builder"]()
    fit_start = time.perf_counter()
    model.fit(x_train_transformed)
    train_time_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    scores = get_anomaly_scores(model, x_validation_transformed)
    threshold_table = build_threshold_table(y_validation, scores)
    threshold_result = select_threshold(threshold_table, criterion=config.threshold_criterion)
    threshold_used = float(threshold_result.threshold)
    metrics = compute_classification_metrics(y_validation, scores, threshold=threshold_used)
    predict_time_seconds = time.perf_counter() - predict_start

    metrics.update(
        {
            "model_name": spec["model_name"],
            "model_family": spec["model_family"],
            "feature_subset": feature_subset_name,
            "feature_subset_name": feature_subset_name,
            "balancing_strategy": spec["balancing_strategy"],
            "split": "validation",
            "number_of_features_used": len(feature_subset),
            "train_rows_used": int(len(train_negative_sample)),
            "validation_rows_used": int(len(validation_sample)),
            "train_time_seconds": float(train_time_seconds),
            "predict_time_seconds": float(predict_time_seconds),
            "threshold_used": threshold_used,
            "classification_report_json": "",
            "max_train_rows_requested": spec.get("max_train_rows"),
            "max_validation_rows_requested": spec.get("max_validation_rows"),
        }
    )

    artifact_path = _model_path(config, build_model_filename(spec["model_name"], feature_subset_name, spec["balancing_strategy"]))
    if config.save_outputs:
        joblib.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "feature_columns": feature_subset,
                "numeric_features": numeric_subset,
                "categorical_features": categorical_subset,
                "threshold": threshold_used,
                "model_name": spec["model_name"],
                "balancing_strategy": spec["balancing_strategy"],
                "feature_subset": feature_subset_name,
                "experiment_family": "anomaly",
                "max_train_rows_requested": spec.get("max_train_rows"),
                "max_validation_rows_requested": spec.get("max_validation_rows"),
            },
            artifact_path,
        )
        _save_threshold_artifact(config, artifact_path.stem, threshold_result.threshold_table)

    metrics["model_artifact_path"] = str(artifact_path)
    return metrics


def _run_feature_ranking(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str], list[str], pd.DataFrame]:
    """Run the notebook-3 feature-ranking pipeline."""

    ranking_train_frame = sample_frame_preserving_positives(
        train_frame,
        target_column=target_col,
        max_rows=config.ranking_max_train_rows,
        keep_all_positives=True,
    )
    ranking_validation_frame = sample_frame_preserving_positives(
        validation_frame,
        target_column=target_col,
        max_rows=config.ranking_max_validation_rows,
        keep_all_positives=True,
    )

    x_train, y_train = split_features_target(ranking_train_frame, target_col=target_col, feature_cols=feature_cols)
    x_validation, y_validation = split_features_target(
        ranking_validation_frame,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    retained_after_constant, constant_metadata = filter_constant_features(x_train, feature_cols)
    retained_after_correlation, correlation_metadata = filter_correlated_features(
        x_train,
        retained_after_constant,
        correlation_threshold=0.98,
        sample_size=config.correlation_sample_size,
    )
    ranking_numeric_features = [
        feature
        for feature in retained_after_correlation
        if feature in x_train.columns and pd.api.types.is_numeric_dtype(x_train[feature])
    ]
    retained_categorical_features = [feature for feature in retained_after_correlation if feature not in ranking_numeric_features]

    feature_metadata = constant_metadata.merge(correlation_metadata, on="feature", how="left")
    feature_metadata["retained_for_ranking"] = feature_metadata["feature"].isin(retained_after_correlation)

    mutual_info = rank_features_mutual_information(
        x_train,
        y_train,
        ranking_numeric_features,
        sample_size=config.mi_sample_size,
    )
    logistic = rank_features_logistic_coefficients(
        x_train,
        y_train,
        ranking_numeric_features,
        sample_size=config.logistic_ranking_sample_size,
    )
    tree = rank_features_tree_importance(
        x_train,
        y_train,
        ranking_numeric_features,
        sample_size=config.tree_ranking_sample_size,
    )

    baseline_for_permutation = build_logistic_baseline(ranking_numeric_features, [])
    baseline_for_permutation.fit(x_train.loc[:, ranking_numeric_features], y_train)
    permutation = rank_features_permutation_importance(
        baseline_for_permutation,
        x_validation,
        y_validation,
        ranking_numeric_features,
        sample_size=min(config.permutation_max_validation_rows, len(x_validation)),
    )

    ranking_frame = build_feature_ranking_table(
        feature_metadata,
        mutual_info=mutual_info,
        logistic=logistic,
        tree=tree,
        permutation=permutation,
        subset_sizes=config.ranking_subset_sizes,
    )
    feature_subsets_numeric = create_feature_subsets(ranking_frame, subset_sizes=config.ranking_subset_sizes)
    feature_subsets = {
        subset_name: deduplicate_keep_order(feature_list + retained_categorical_features)
        for subset_name, feature_list in feature_subsets_numeric.items()
    }

    if config.save_outputs:
        save_feature_ranking(
            ranking_frame,
            csv_path=_table_path(config, "feature_ranking.csv"),
            excel_path=_table_path(config, "feature_ranking.xlsx"),
        )

    return ranking_frame, feature_subsets, ranking_numeric_features, retained_categorical_features, feature_metadata


def _optional_extra_experiments(
    config: ExperimentConfig,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_subsets: dict[str, list[str]],
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run lighter optional experiments that materially match notebook 3."""

    optional_records: list[dict[str, Any]] = []
    optional_comparisons: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []

    if not config.run_optional_deep_models:
        omissions.extend(
            [
                {"model_name": "mlp_classifier_light", "model_family": "optional", "reason": "Experimentos opcionales deshabilitados."},
                {"model_name": "xgboost_calibrated", "model_family": "optional", "reason": "Experimentos opcionales deshabilitados."},
            ]
        )
        return optional_records, optional_comparisons, omissions

    preferred_subset = "top_100" if "top_100" in feature_subsets else ("top_50" if "top_50" in feature_subsets else next(iter(feature_subsets)))
    feature_subset = feature_subsets[preferred_subset]
    numeric_subset = [column for column in feature_subset if column in numeric_features]
    categorical_subset = [column for column in feature_subset if column in categorical_features]

    light_limits = config.model_sample_limits["mlp_classifier_light"]
    light_train = sample_frame_preserving_positives(
        train_frame, target_col, max_rows=light_limits["max_train_rows"], keep_all_positives=True
    )
    light_validation = sample_frame_preserving_positives(
        validation_frame, target_col, max_rows=light_limits["max_validation_rows"], keep_all_positives=True
    )
    x_train_light, y_train_light = split_features_target(light_train, target_col=target_col, feature_cols=feature_subset)
    x_validation_light, y_validation_light = split_features_target(
        light_validation, target_col=target_col, feature_cols=feature_subset
    )

    light_model = build_generic_pipeline(
        MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=40, random_state=RANDOM_STATE),
        numeric_subset,
        categorical_subset,
        scale_numeric=True,
    )
    fit_start = time.perf_counter()
    light_model.fit(x_train_light, y_train_light)
    train_time = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    light_scores = pd.Series(light_model.predict_proba(x_validation_light)[:, 1], index=y_validation_light.index, name="score")
    light_threshold_table = build_threshold_table(y_validation_light, light_scores)
    light_threshold_result = select_threshold(light_threshold_table, criterion=config.threshold_criterion)
    predict_time = time.perf_counter() - predict_start
    light_metrics = compute_classification_metrics(y_validation_light, light_scores, threshold=light_threshold_result.threshold)
    light_metrics.update(
        {
            "model_name": "mlp_classifier_light",
            "model_family": "neural_network_reference",
            "feature_subset": preferred_subset,
            "balancing_strategy": "none",
            "split": "validation",
            "number_of_features_used": len(feature_subset),
            "train_rows_used": int(len(light_train)),
            "validation_rows_used": int(len(light_validation)),
            "train_time_seconds": float(train_time),
            "predict_time_seconds": float(predict_time),
            "threshold_used": float(light_threshold_result.threshold),
        }
    )
    light_path = _model_path(config, build_model_filename("mlp_classifier_light", preferred_subset, "none"))
    if config.save_outputs:
        joblib.dump(
            {
                "model": light_model,
                "feature_columns": feature_subset,
                "numeric_features": numeric_subset,
                "categorical_features": categorical_subset,
                "threshold": float(light_threshold_result.threshold),
                "model_name": "mlp_classifier_light",
                "balancing_strategy": "none",
                "feature_subset": preferred_subset,
                "experiment_family": "neural_network_reference",
                "max_train_rows_requested": light_limits["max_train_rows"],
                "max_validation_rows_requested": light_limits["max_validation_rows"],
            },
            light_path,
        )
        _save_threshold_artifact(config, light_path.stem, light_threshold_result.threshold_table)
    light_metrics["model_artifact_path"] = str(light_path)
    optional_records.append(light_metrics)

    if AVAILABLE_LIBRARIES["xgboost"]:
        xgb_limits = config.model_sample_limits["xgboost_calibrated"]
        xgb_train = sample_frame_preserving_positives(
            train_frame, target_col, max_rows=xgb_limits["max_train_rows"], keep_all_positives=True
        )
        xgb_validation = sample_frame_preserving_positives(
            validation_frame, target_col, max_rows=xgb_limits["max_validation_rows"], keep_all_positives=True
        )
        x_train_xgb, y_train_xgb = split_features_target(xgb_train, target_col=target_col, feature_cols=feature_subset)
        x_validation_xgb, y_validation_xgb = split_features_target(
            xgb_validation, target_col=target_col, feature_cols=feature_subset
        )
        positives = max(float(y_train_xgb.sum()), 1.0)
        negatives = max(float((y_train_xgb == 0).sum()), 1.0)
        xgb_pipeline = build_generic_pipeline(
            XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                scale_pos_weight=negatives / positives,
            ),
            numeric_subset,
            categorical_subset,
            scale_numeric=False,
        )
        calibrated = CalibratedClassifierCV(estimator=xgb_pipeline, method="sigmoid", cv=3)
        fit_start = time.perf_counter()
        calibrated.fit(x_train_xgb, y_train_xgb)
        train_time = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        xgb_scores = pd.Series(calibrated.predict_proba(x_validation_xgb)[:, 1], index=y_validation_xgb.index, name="score")
        xgb_threshold_table = build_threshold_table(y_validation_xgb, xgb_scores)
        xgb_threshold_result = select_threshold(xgb_threshold_table, criterion=config.threshold_criterion)
        predict_time = time.perf_counter() - predict_start
        xgb_metrics = compute_classification_metrics(y_validation_xgb, xgb_scores, threshold=xgb_threshold_result.threshold)
        xgb_metrics.update(
            {
                "model_name": "xgboost_calibrated",
                "model_family": "ensemble_calibrated",
                "feature_subset": preferred_subset,
                "balancing_strategy": "scale_pos_weight",
                "split": "validation",
                "number_of_features_used": len(feature_subset),
                "train_rows_used": int(len(xgb_train)),
                "validation_rows_used": int(len(xgb_validation)),
                "train_time_seconds": float(train_time),
                "predict_time_seconds": float(predict_time),
                "threshold_used": float(xgb_threshold_result.threshold),
            }
        )
        xgb_path = _model_path(config, build_model_filename("xgboost_calibrated", preferred_subset, "scale_pos_weight"))
        if config.save_outputs:
            joblib.dump(
                {
                    "model": calibrated,
                    "feature_columns": feature_subset,
                    "numeric_features": numeric_subset,
                    "categorical_features": categorical_subset,
                    "threshold": float(xgb_threshold_result.threshold),
                    "model_name": "xgboost_calibrated",
                    "balancing_strategy": "scale_pos_weight",
                    "feature_subset": preferred_subset,
                    "experiment_family": "ensemble_calibrated",
                    "max_train_rows_requested": xgb_limits["max_train_rows"],
                    "max_validation_rows_requested": xgb_limits["max_validation_rows"],
                },
                xgb_path,
            )
            _save_threshold_artifact(config, xgb_path.stem, xgb_threshold_result.threshold_table)
        xgb_metrics["model_artifact_path"] = str(xgb_path)
        optional_records.append(xgb_metrics)

        optional_comparisons.append(
            {
                "comparison_group": "optional_reference",
                "feature_subset": preferred_subset,
                "baseline_model": "mlp_classifier_light",
                "treatment_model": "xgboost_calibrated",
                "baseline_pr_auc": light_metrics["pr_auc"],
                "treatment_pr_auc": xgb_metrics["pr_auc"],
                "delta_pr_auc": xgb_metrics["pr_auc"] - light_metrics["pr_auc"],
                "baseline_recall": light_metrics["recall"],
                "treatment_recall": xgb_metrics["recall"],
                "delta_recall": xgb_metrics["recall"] - light_metrics["recall"],
                "baseline_f1": light_metrics["f1"],
                "treatment_f1": xgb_metrics["f1"],
                "delta_f1": xgb_metrics["f1"] - light_metrics["f1"],
            }
        )
    else:
        omissions.append({"model_name": "xgboost_calibrated", "model_family": "optional", "reason": "xgboost no esta instalado."})

    return optional_records, optional_comparisons, omissions


def run_modeling_experiments(config: ExperimentConfig) -> dict[str, Any]:
    """Run the full notebook-3 modeling workflow."""

    if not config.dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset de modelado: {config.dataset_path}. Falta primero el notebook 2."
        )

    dataset_columns = get_modeling_dataset_columns(config.dataset_path)
    dataset_preview = load_modeling_dataset_preview(config.dataset_path, n_rows=config.dataset_preview_rows)
    required_base_columns = [
        "user",
        "card",
        "user_card_id",
        "datetime",
        "year_month",
        "merchant_name",
        "merchant_city",
        "merchant_state",
        "zip",
        "mcc",
        "use_chip",
        "is_fraud",
    ]
    validation_summary_frame = validate_modeling_dataset(dataset_columns, required_base_columns)

    target_col = detect_target_column(dataset_preview)
    identifier_cols = detect_identifier_columns(dataset_preview)
    base_feature_cols = build_candidate_feature_columns(dataset_columns, target_col=target_col, identifier_cols=identifier_cols)

    target_derived_patterns = ("fraud", "target", "label")
    temporal_leakage_patterns = ("global", "overall", "all_time")
    excluded_target_derived_features = sorted(
        [column for column in base_feature_cols if any(pattern in column.lower() for pattern in target_derived_patterns)]
    )
    excluded_global_features = sorted(
        [column for column in base_feature_cols if any(pattern in column.lower() for pattern in temporal_leakage_patterns)]
    )
    excluded_feature_cols = sorted(set(excluded_target_derived_features) | set(excluded_global_features))
    feature_cols = [column for column in base_feature_cols if column not in excluded_feature_cols]

    split_summary = summarize_temporal_split_from_parquet(config.dataset_path, config.temporal_split, target_col=target_col)
    monthly_summary = summarize_temporal_split_monthly_from_parquet(config.dataset_path, config.temporal_split, target_col=target_col)

    modeling_columns_for_memory = list(dict.fromkeys(feature_cols + [target_col]))
    train_frame = load_temporal_split_dataset(
        config.dataset_path,
        config.temporal_split,
        split_name="train",
        columns=modeling_columns_for_memory,
        max_rows=config.in_memory_train_rows,
        target_col=target_col,
        keep_all_positives=True,
        sort_by=None,
    )
    validation_frame = load_temporal_split_dataset(
        config.dataset_path,
        config.temporal_split,
        split_name="validation",
        columns=modeling_columns_for_memory,
        max_rows=config.in_memory_validation_rows,
        target_col=target_col,
        keep_all_positives=True,
        sort_by=None,
    )
    split_frames = {"train": train_frame, "validation": validation_frame}

    ranking_frame, feature_subsets, ranking_numeric_features, retained_categorical_features, feature_metadata = _run_feature_ranking(
        train_frame, validation_frame, feature_cols, target_col, config
    )

    numeric_features = [column for column in feature_cols if column in train_frame.columns and pd.api.types.is_numeric_dtype(train_frame[column])]
    categorical_features = [column for column in feature_cols if column not in numeric_features]

    omitted_models: list[dict[str, Any]] = []
    supervised_model_specs: list[dict[str, Any]] = []
    anomaly_model_specs: list[dict[str, Any]] = []

    def add_omission(model_name: str, reason: str, family: str = "supervised") -> None:
        omitted_models.append({"model_name": model_name, "model_family": family, "reason": reason})

    non_negative_check_sample = sample_frame_preserving_positives(
        train_frame, target_column=target_col, max_rows=min(50_000, len(train_frame)), keep_all_positives=True
    )
    if ranking_numeric_features:
        min_numeric_value = pd.to_numeric(non_negative_check_sample[ranking_numeric_features].stack(), errors="coerce").min()
        has_only_non_negative_features = bool(pd.notna(min_numeric_value) and min_numeric_value >= 0)
        binary_like_features = bool(
            non_negative_check_sample[ranking_numeric_features].apply(lambda series: series.dropna().isin([0, 1]).all()).all()
        )
    else:
        has_only_non_negative_features = False
        binary_like_features = False

    def limit_config(model_key: str) -> dict[str, int | None]:
        return config.model_sample_limits[model_key].copy()

    supervised_model_specs.extend(
        [
            {
                "model_name": "dummy_classifier",
                "model_family": "baseline",
                "balancing_strategy": "none",
                "subset_names": ["all_features"],
                "builder": lambda numeric_subset, categorical_subset: build_dummy_baseline(),
                **limit_config("dummy_classifier"),
            },
            {
                "model_name": "logistic_regression",
                "model_family": "linear",
                "balancing_strategy": "none",
                "subset_names": ["top_25", "top_50", "top_100", "all_features"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    LogisticRegression(class_weight=None, max_iter=1000, random_state=RANDOM_STATE, solver="liblinear"),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=True,
                ),
                **limit_config("logistic_regression"),
            },
            {
                "model_name": "logistic_regression",
                "model_family": "linear",
                "balancing_strategy": "class_weight_balanced",
                "subset_names": ["top_25", "top_50", "top_100", "all_features"],
                "builder": lambda numeric_subset, categorical_subset: build_logistic_baseline(numeric_subset, categorical_subset),
                **limit_config("logistic_regression"),
            },
            {
                "model_name": "logistic_regression",
                "model_family": "linear",
                "balancing_strategy": "random_undersampling",
                "subset_names": ["top_25", "top_50"],
                "builder": lambda numeric_subset, categorical_subset: build_random_undersample_logistic_pipeline(
                    numeric_subset, categorical_subset
                ),
                **limit_config("logistic_regression_undersample"),
            },
            {
                "model_name": "logistic_regression",
                "model_family": "linear",
                "balancing_strategy": "smote",
                "subset_names": ["top_25", "top_50"],
                "builder": lambda numeric_subset, categorical_subset: build_smote_logistic_pipeline(
                    numeric_subset, categorical_subset
                ),
                **limit_config("logistic_regression_smote"),
            },
            {
                "model_name": "decision_tree",
                "model_family": "tree",
                "balancing_strategy": "class_weight_balanced",
                "subset_names": ["top_25", "top_50", "top_100"],
                "builder": lambda numeric_subset, categorical_subset: build_decision_tree_baseline(numeric_subset, categorical_subset),
                **limit_config("decision_tree"),
            },
            {
                "model_name": "random_forest",
                "model_family": "ensemble",
                "balancing_strategy": "class_weight_balanced_subsample",
                "subset_names": ["top_25", "top_50", "top_100"],
                "builder": lambda numeric_subset, categorical_subset: build_random_forest_pipeline(
                    numeric_subset, categorical_subset, n_estimators=300, n_jobs=1
                ),
                **limit_config("random_forest"),
            },
            {
                "model_name": "extra_trees",
                "model_family": "ensemble",
                "balancing_strategy": "class_weight_balanced",
                "subset_names": ["top_25", "top_50", "top_100"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=False,
                ),
                **limit_config("extra_trees"),
            },
            {
                "model_name": "hist_gradient_boosting",
                "model_family": "ensemble",
                "balancing_strategy": "none",
                "subset_names": ["top_25", "top_50", "top_100"],
                "builder": lambda numeric_subset, categorical_subset: build_hist_gradient_boosting_pipeline(
                    numeric_subset, categorical_subset
                ),
                **limit_config("hist_gradient_boosting"),
            },
            {
                "model_name": "gradient_boosting",
                "model_family": "ensemble",
                "balancing_strategy": "none",
                "subset_names": ["top_25", "top_50"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=False,
                ),
                **limit_config("gradient_boosting"),
            },
            {
                "model_name": "adaboost",
                "model_family": "ensemble",
                "balancing_strategy": "none",
                "subset_names": ["top_25", "top_50"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=150),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=False,
                ),
                **limit_config("adaboost"),
            },
            {
                "model_name": "bagging_tree",
                "model_family": "ensemble",
                "balancing_strategy": "none",
                "subset_names": ["top_25", "top_50"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    BaggingClassifier(
                        estimator=DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
                        n_estimators=50,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=False,
                ),
                **limit_config("bagging_tree"),
            },
            {
                "model_name": "gaussian_nb",
                "model_family": "naive_bayes",
                "balancing_strategy": "none",
                "subset_names": ["top_25", "top_50", "top_100"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    GaussianNB(), numeric_subset, categorical_subset, scale_numeric=False
                ),
                **limit_config("gaussian_nb"),
            },
            {
                "model_name": "sgd_classifier",
                "model_family": "linear_margin",
                "balancing_strategy": "class_weight_balanced",
                "subset_names": ["top_25", "top_50", "top_100", "all_features"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        max_iter=1000,
                        early_stopping=True,
                    ),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=True,
                ),
                **limit_config("sgd_classifier"),
            },
            {
                "model_name": "linear_svc",
                "model_family": "linear_margin",
                "balancing_strategy": "class_weight_balanced",
                "subset_names": ["top_25", "top_50", "top_100"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, dual="auto"),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=True,
                ),
                **limit_config("linear_svc"),
            },
            {
                "model_name": "mlp_classifier",
                "model_family": "neural_network",
                "balancing_strategy": "none",
                "subset_names": ["top_50", "top_100"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=50, random_state=RANDOM_STATE),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=True,
                ),
                **limit_config("mlp_classifier"),
            },
        ]
    )

    if AVAILABLE_LIBRARIES["xgboost"]:
        supervised_model_specs.append(
            {
                "model_name": "xgboost",
                "model_family": "ensemble",
                "balancing_strategy": "scale_pos_weight_manual",
                "subset_names": ["top_25", "top_50", "top_100"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                    numeric_subset,
                    categorical_subset,
                    scale_numeric=False,
                ),
                **limit_config("xgboost"),
            }
        )
    else:
        add_omission("xgboost", "xgboost no esta instalado.")

    if binary_like_features:
        supervised_model_specs.append(
            {
                "model_name": "bernoulli_nb",
                "model_family": "naive_bayes",
                "balancing_strategy": "none",
                "subset_names": ["top_25", "top_50"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    BernoulliNB(), numeric_subset, categorical_subset, scale_numeric=False
                ),
                **limit_config("bernoulli_nb"),
            }
        )
    else:
        add_omission("bernoulli_nb", "Las features no son binarias ni cercanas a Bernoulli.")

    if has_only_non_negative_features:
        supervised_model_specs.append(
            {
                "model_name": "complement_nb",
                "model_family": "naive_bayes",
                "balancing_strategy": "none",
                "subset_names": ["top_25", "top_50"],
                "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                    ComplementNB(), numeric_subset, categorical_subset, scale_numeric=False
                ),
                **limit_config("complement_nb"),
            }
        )
    else:
        add_omission(
            "complement_nb",
            "ComplementNB requiere features no negativas y el muestreo detecto valores negativos.",
        )

    if config.run_optional_heavy_models:
        supervised_model_specs.extend(
            [
                {
                    "model_name": "knn",
                    "model_family": "neighbors",
                    "balancing_strategy": "none",
                    "subset_names": ["top_25", "top_50"],
                    "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                        KNeighborsClassifier(n_neighbors=15),
                        numeric_subset,
                        categorical_subset,
                        scale_numeric=True,
                    ),
                    **limit_config("knn"),
                },
                {
                    "model_name": "svc_rbf",
                    "model_family": "kernel_margin",
                    "balancing_strategy": "class_weight_balanced",
                    "subset_names": ["top_25"],
                    "builder": lambda numeric_subset, categorical_subset: build_generic_pipeline(
                        SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=False, random_state=RANDOM_STATE),
                        numeric_subset,
                        categorical_subset,
                        scale_numeric=True,
                    ),
                    **limit_config("svc_rbf"),
                },
            ]
        )
    else:
        add_omission("knn", "KNN omitido por configuracion.")
        add_omission("svc_rbf", "SVC no lineal omitido por configuracion.")

    if config.run_anomaly_models:
        anomaly_model_specs.append(
            {
                "model_name": "isolation_forest",
                "model_family": "anomaly",
                "balancing_strategy": "unsupervised_negative_train_only",
                "subset_names": ["top_25"],
                "builder": lambda: IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_STATE, n_jobs=1),
                "scale_numeric": False,
                **limit_config("isolation_forest"),
            }
        )
        if config.run_optional_heavy_models:
            anomaly_model_specs.extend(
                [
                    {
                        "model_name": "local_outlier_factor",
                        "model_family": "anomaly",
                        "balancing_strategy": "unsupervised_negative_train_only",
                        "subset_names": ["top_25"],
                        "builder": lambda: LocalOutlierFactor(n_neighbors=35, novelty=True),
                        "scale_numeric": True,
                        **limit_config("local_outlier_factor"),
                    },
                    {
                        "model_name": "one_class_svm",
                        "model_family": "anomaly",
                        "balancing_strategy": "unsupervised_negative_train_only",
                        "subset_names": ["top_25"],
                        "builder": lambda: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
                        "scale_numeric": True,
                        **limit_config("one_class_svm"),
                    },
                ]
            )
        else:
            add_omission("local_outlier_factor", "LOF omitido por configuracion.", family="anomaly")
            add_omission("one_class_svm", "OneClassSVM omitido por configuracion.", family="anomaly")
    else:
        add_omission("isolation_forest", "Anomaly models deshabilitados por configuracion.", family="anomaly")

    experiment_records: list[dict[str, Any]] = []
    for spec in supervised_model_specs:
        for subset_name in spec["subset_names"]:
            feature_subset = feature_subsets[subset_name]
            try:
                record = fit_and_evaluate_supervised_experiment(
                    spec=spec,
                    feature_subset_name=subset_name,
                    feature_subset=feature_subset,
                    split_data=split_frames,
                    target_column=target_col,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    config=config,
                )
                experiment_records.append(record)
            except Exception as exc:
                add_omission(
                    spec["model_name"],
                    f"Fallo en entrenamiento/evaluacion para subset={subset_name}, balancing={spec['balancing_strategy']}: {exc}",
                    family=str(spec["model_family"]),
                )

    for spec in anomaly_model_specs:
        for subset_name in spec["subset_names"]:
            feature_subset = feature_subsets[subset_name]
            try:
                record = fit_and_evaluate_anomaly_experiment(
                    spec=spec,
                    feature_subset_name=subset_name,
                    feature_subset=feature_subset,
                    split_data=split_frames,
                    target_column=target_col,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    config=config,
                )
                experiment_records.append(record)
            except Exception as exc:
                add_omission(
                    spec["model_name"],
                    f"Fallo en entrenamiento/evaluacion para subset={subset_name}, balancing={spec['balancing_strategy']}: {exc}",
                    family=str(spec["model_family"]),
                )

    results_frame = build_metrics_frame(experiment_records)
    if not results_frame.empty:
        if "subset_name" in results_frame.columns:
            results_frame["feature_subset"] = results_frame["feature_subset"].fillna(results_frame["subset_name"])
            results_frame = results_frame.drop(columns=["subset_name"])
        results_frame = results_frame.sort_values(
            ["pr_auc", "recall", "f1", "precision"], ascending=[False, False, False, False]
        ).reset_index(drop=True)

    optional_records, optional_comparisons, optional_omissions = _optional_extra_experiments(
        config=config,
        train_frame=train_frame,
        validation_frame=validation_frame,
        feature_subsets=feature_subsets,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
    )
    omitted_models.extend(optional_omissions)
    if optional_records:
        optional_frame = build_metrics_frame(optional_records)
        results_frame = pd.concat([results_frame, optional_frame], axis=0, ignore_index=True)
        results_frame = results_frame.sort_values(
            ["pr_auc", "recall", "f1", "precision"], ascending=[False, False, False, False]
        ).reset_index(drop=True)

    omitted_models_frame = pd.DataFrame(omitted_models)

    if config.save_outputs:
        save_table_with_optional_excel(split_summary, _table_path(config, "split_summary.csv"), _table_path(config, "split_summary.xlsx"))
        save_table_with_optional_excel(
            monthly_summary, _table_path(config, "split_monthly_summary.csv"), _table_path(config, "split_monthly_summary.xlsx")
        )
        if not results_frame.empty:
            save_metrics_table(
                results_frame,
                csv_path=_table_path(config, "model_experiment_results.csv"),
                excel_path=_table_path(config, "model_experiment_results.xlsx"),
            )
        if not omitted_models_frame.empty:
            omitted_models_frame.to_csv(_table_path(config, "omitted_model_experiments.csv"), index=False)
        if optional_comparisons:
            comparison_frame = pd.DataFrame(optional_comparisons)
            save_table_with_optional_excel(
                comparison_frame,
                _table_path(config, "optional_model_comparisons.csv"),
                _table_path(config, "optional_model_comparisons.xlsx"),
            )

    if config.save_outputs and not results_frame.empty:
        plot_metric_comparison(results_frame, "pr_auc", _figure_path(config, "model_pr_auc_comparison.png"))
        plot_metric_comparison(results_frame, "roc_auc", _figure_path(config, "model_roc_auc_comparison.png"))
        plot_metric_comparison(results_frame, "recall", _figure_path(config, "model_recall_comparison.png"))
        plot_metric_comparison(results_frame, "f1", _figure_path(config, "model_f1_comparison.png"))

        best_row = results_frame.iloc[0]
        best_artifact = joblib.load(best_row["model_artifact_path"])
        best_feature_subset = best_artifact["feature_columns"]
        max_validation_rows = best_artifact.get("max_validation_rows_requested")
        validation_sample_for_best = sample_frame_preserving_positives(
            validation_frame, target_column=target_col, max_rows=max_validation_rows, keep_all_positives=True
        )
        x_validation_best, y_validation_best = split_features_target(
            validation_sample_for_best, target_col=target_col, feature_cols=best_feature_subset
        )
        if best_artifact.get("experiment_family") == "anomaly":
            transformed_validation = best_artifact["preprocessor"].transform(x_validation_best)
            best_scores = get_anomaly_scores(best_artifact["model"], transformed_validation)
        else:
            best_scores = get_positive_class_scores(best_artifact["model"], x_validation_best)

        plot_confusion_matrix(
            y_validation_best,
            best_scores,
            threshold=best_artifact["threshold"],
            output_path=_figure_path(config, "best_validation_confusion_matrix.png"),
        )
        plot_roc_curve(
            y_validation_best,
            best_scores,
            model_name=str(best_row["model_name"]),
            output_path=_figure_path(config, "best_validation_roc_curve.png"),
        )
        plot_precision_recall_curve(
            y_validation_best,
            best_scores,
            model_name=str(best_row["model_name"]),
            output_path=_figure_path(config, "best_validation_precision_recall_curve.png"),
        )

    return {
        "dataset_columns": dataset_columns,
        "dataset_preview": dataset_preview,
        "validation_summary_frame": validation_summary_frame,
        "split_summary": split_summary,
        "monthly_summary": monthly_summary,
        "feature_metadata": feature_metadata,
        "ranking_frame": ranking_frame,
        "feature_subsets": feature_subsets,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "results_frame": results_frame,
        "omitted_models_frame": omitted_models_frame,
        "target_col": target_col,
        "identifier_cols": identifier_cols,
        "excluded_feature_cols": excluded_feature_cols,
        "retained_categorical_features": retained_categorical_features,
    }
