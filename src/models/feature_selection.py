"""Feature filtering and ranking helpers for notebook 3."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src.models.train_models import RANDOM_STATE


def _sample_rows_early(
    x_frame: pd.DataFrame,
    y: pd.Series | None = None,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Sample rows before expensive preprocessing to control memory."""

    if sample_size is None or len(x_frame) <= sample_size:
        return x_frame, y
    sampled_idx = x_frame.sample(n=sample_size, random_state=RANDOM_STATE).index
    sampled_x = x_frame.loc[sampled_idx].copy()
    sampled_y = y.loc[sampled_idx].copy() if y is not None else None
    return sampled_x, sampled_y


def _prepare_numeric_frame(x_train: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Select numeric-like columns and impute missing values."""

    frame = x_train.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(frame)
    return pd.DataFrame(transformed, columns=frame.columns, index=frame.index)


def filter_constant_features(
    x_train: pd.DataFrame,
    feature_columns: Sequence[str],
    near_constant_threshold: float = 0.999,
) -> tuple[list[str], pd.DataFrame]:
    """Remove constant and near-constant columns."""

    rows = []
    kept: list[str] = []
    for column in feature_columns:
        series = x_train[column]
        value_counts = series.value_counts(dropna=False, normalize=True)
        top_frequency = float(value_counts.iloc[0]) if not value_counts.empty else 1.0
        is_constant = int(series.nunique(dropna=False) <= 1)
        is_near_constant = int(top_frequency >= near_constant_threshold)
        rows.append(
            {
                "feature": column,
                "n_unique": int(series.nunique(dropna=False)),
                "top_frequency": top_frequency,
                "is_constant": is_constant,
                "is_near_constant": is_near_constant,
            }
        )
        if not is_constant and not is_near_constant:
            kept.append(column)
    return kept, pd.DataFrame(rows)


def filter_correlated_features(
    x_train: pd.DataFrame,
    feature_columns: Sequence[str],
    correlation_threshold: float = 0.98,
    sample_size: int | None = 200_000,
) -> tuple[list[str], pd.DataFrame]:
    """Remove one variable from highly correlated pairs."""

    sampled_x, _ = _sample_rows_early(x_train.loc[:, list(feature_columns)], sample_size=sample_size)
    numeric_frame = _prepare_numeric_frame(sampled_x, sampled_x.columns)
    corr = numeric_frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if (upper[column] > correlation_threshold).any()]

    rows = []
    for column in feature_columns:
        max_corr = float(upper[column].max()) if column in upper.columns and upper[column].notna().any() else 0.0
        rows.append(
            {
                "feature": column,
                "max_abs_correlation": max_corr,
                "drop_for_correlation": int(column in to_drop),
            }
        )
    kept = [column for column in feature_columns if column not in to_drop]
    return kept, pd.DataFrame(rows)


def rank_features_mutual_information(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: Sequence[str],
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Rank features using mutual information on training data only."""

    sampled_x, sampled_y = _sample_rows_early(x_train.loc[:, list(feature_columns)], y_train, sample_size=sample_size)
    numeric_frame = _prepare_numeric_frame(sampled_x, feature_columns)
    scores = mutual_info_classif(numeric_frame, sampled_y, random_state=RANDOM_STATE)
    return pd.DataFrame({"feature": numeric_frame.columns, "mutual_info_score": scores})


def rank_features_logistic_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: Sequence[str],
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Rank features by absolute coefficients of a balanced logistic regression."""

    sampled_x, sampled_y = _sample_rows_early(x_train.loc[:, list(feature_columns)], y_train, sample_size=sample_size)
    numeric_frame = _prepare_numeric_frame(sampled_x, feature_columns)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    model.fit(numeric_frame, sampled_y)
    coefficients = np.abs(model.coef_).ravel()
    return pd.DataFrame({"feature": numeric_frame.columns, "logistic_abs_coef": coefficients})


def rank_features_tree_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: Sequence[str],
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Rank features by tree-based importance."""

    sampled_x, sampled_y = _sample_rows_early(x_train.loc[:, list(feature_columns)], y_train, sample_size=sample_size)
    numeric_frame = _prepare_numeric_frame(sampled_x, feature_columns)
    model = ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(numeric_frame, sampled_y)
    return pd.DataFrame({"feature": numeric_frame.columns, "tree_importance": model.feature_importances_})


def rank_features_permutation_importance(
    estimator: object,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    feature_columns: Sequence[str],
    sample_size: int | None = None,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Rank features via permutation importance on validation data."""

    sampled_x, sampled_y = _sample_rows_early(
        x_validation.loc[:, list(feature_columns)],
        y_validation,
        sample_size=sample_size,
    )
    numeric_frame = _prepare_numeric_frame(sampled_x, feature_columns)
    result = permutation_importance(
        estimator,
        numeric_frame,
        sampled_y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": numeric_frame.columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    )


def build_feature_ranking_table(
    feature_metadata: pd.DataFrame,
    mutual_info: pd.DataFrame | None = None,
    logistic: pd.DataFrame | None = None,
    tree: pd.DataFrame | None = None,
    permutation: pd.DataFrame | None = None,
    subset_sizes: Sequence[int] = (25, 50, 75, 100),
) -> pd.DataFrame:
    """Combine metadata and multiple ranking methods into one final table."""

    ranking = feature_metadata.copy()
    for table in (mutual_info, logistic, tree, permutation):
        if table is not None and not table.empty:
            ranking = ranking.merge(table, on="feature", how="left")

    ranking["rank_mutual_info"] = ranking["mutual_info_score"].rank(ascending=False, method="average")
    ranking["rank_logistic"] = ranking["logistic_abs_coef"].rank(ascending=False, method="average")
    ranking["rank_tree"] = ranking["tree_importance"].rank(ascending=False, method="average")
    if "permutation_importance_mean" in ranking.columns:
        ranking["rank_permutation"] = ranking["permutation_importance_mean"].rank(ascending=False, method="average")
    else:
        ranking["rank_permutation"] = np.nan

    rank_columns = ["rank_mutual_info", "rank_logistic", "rank_tree", "rank_permutation"]
    ranking["rank_average"] = ranking[rank_columns].mean(axis=1, skipna=True)
    ranking = ranking.sort_values(["rank_average", "feature"]).reset_index(drop=True)

    for size in subset_sizes:
        selected = ranking.index < size
        ranking[f"selected_top_{size}"] = selected
    return ranking


def create_feature_subsets(
    ranking_table: pd.DataFrame,
    subset_sizes: Sequence[int] = (25, 50, 75, 100),
) -> dict[str, list[str]]:
    """Create named feature subsets from a ranking table."""

    subsets = {"all_features": ranking_table["feature"].tolist()}
    for size in subset_sizes:
        subsets[f"top_{size}"] = ranking_table.loc[ranking_table[f"selected_top_{size}"], "feature"].tolist()
    return subsets


def save_feature_ranking(
    ranking_table: pd.DataFrame,
    csv_path: str | Path,
    excel_path: str | Path | None = None,
) -> None:
    """Persist a feature ranking table to CSV and optionally Excel."""

    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    ranking_table.to_csv(csv_target, index=False)
    if excel_path is not None:
        excel_target = Path(excel_path)
        excel_target.parent.mkdir(parents=True, exist_ok=True)
        ranking_table.to_excel(excel_target, index=False)

