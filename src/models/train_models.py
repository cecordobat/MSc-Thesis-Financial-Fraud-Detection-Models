"""Reusable data loading and model-construction helpers for notebook 3/4."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
except ModuleNotFoundError:  # pragma: no cover
    ImbPipeline = None

RANDOM_STATE = 42


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Configuration for a temporal train/validation/test split."""

    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    excluded_periods: Sequence[str] = field(default_factory=tuple)


def _to_month_period(value: str | pd.Period) -> pd.Period:
    """Convert a YYYY-MM string into a monthly pandas Period."""

    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(value, freq="M")


def _period_start_timestamp(value: str | pd.Period) -> pd.Timestamp:
    """Convert a monthly period-like value into a month-start timestamp."""

    return _to_month_period(value).to_timestamp(how="start")


def load_modeling_dataset(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load the transaction-level modeling dataset from parquet."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {dataset_path}")
    return pd.read_parquet(dataset_path, columns=list(columns) if columns is not None else None)


def get_modeling_dataset_columns(path: str | Path) -> list[str]:
    """Return parquet column names without loading the full dataset."""

    import pyarrow.parquet as pq

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {dataset_path}")
    return pq.ParquetFile(dataset_path).schema_arrow.names


def load_modeling_dataset_preview(
    path: str | Path,
    columns: Sequence[str] | None = None,
    n_rows: int = 5,
) -> pd.DataFrame:
    """Load only a small preview from the parquet dataset."""

    import pyarrow.parquet as pq

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {dataset_path}")

    parquet_file = pq.ParquetFile(dataset_path)
    batch_iter = parquet_file.iter_batches(
        batch_size=n_rows,
        columns=list(columns) if columns is not None else None,
    )
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        return pd.DataFrame(columns=list(columns) if columns is not None else parquet_file.schema_arrow.names)
    return first_batch.to_pandas()


def _split_name_to_filter(
    split_config: TemporalSplitConfig,
    split_name: str,
    year_month_col: str = "year_month",
):
    """Build a pyarrow.dataset filter expression for one named temporal split."""

    import pyarrow.dataset as ds

    field_expr = ds.field(year_month_col)
    if split_name == "train":
        return (field_expr >= _period_start_timestamp(split_config.train_start)) & (
            field_expr <= _period_start_timestamp(split_config.train_end)
        )
    if split_name == "validation":
        return (field_expr >= _period_start_timestamp(split_config.validation_start)) & (
            field_expr <= _period_start_timestamp(split_config.validation_end)
        )
    if split_name == "test":
        return (field_expr >= _period_start_timestamp(split_config.test_start)) & (
            field_expr <= _period_start_timestamp(split_config.test_end)
        )
    if split_name == "excluded":
        excluded_values = [_period_start_timestamp(period) for period in split_config.excluded_periods]
        return field_expr.isin(excluded_values)
    raise ValueError(f"Unknown split_name: {split_name}")


def _stable_hash_priority(frame: pd.DataFrame) -> np.ndarray:
    """Return a deterministic random-like priority vector for sampling."""

    candidate_columns = [col for col in ("user_card_id", "datetime", "user", "card") if col in frame.columns]
    if candidate_columns:
        key_frame = frame.loc[:, candidate_columns].astype(str)
        hashed = pd.util.hash_pandas_object(key_frame, index=False).to_numpy(dtype=np.uint64)
    else:
        hashed = pd.util.hash_pandas_object(frame.index.to_series(), index=False).to_numpy(dtype=np.uint64)
    return hashed.astype(np.uint64)


def _reduce_negative_reservoir(
    negative_frames: list[pd.DataFrame],
    negative_priorities: list[np.ndarray],
    negative_quota: int,
) -> tuple[list[pd.DataFrame], list[np.ndarray]]:
    """Trim stored negative batches to the required quota using deterministic priorities."""

    if negative_quota <= 0 or not negative_frames:
        return [], []

    all_priorities = np.concatenate(negative_priorities)
    if len(all_priorities) <= negative_quota:
        return negative_frames, negative_priorities

    threshold = np.partition(all_priorities, negative_quota - 1)[negative_quota - 1]
    kept_frames: list[pd.DataFrame] = []
    kept_priorities: list[np.ndarray] = []
    running_total = 0
    for frame, priorities in zip(negative_frames, negative_priorities, strict=False):
        mask = priorities < threshold
        if running_total < negative_quota:
            equal_mask = priorities == threshold
            remaining = negative_quota - (running_total + int(mask.sum()))
            if remaining > 0 and equal_mask.any():
                equal_indices = np.flatnonzero(equal_mask)[:remaining]
                temp_mask = mask.copy()
                temp_mask[equal_indices] = True
                mask = temp_mask
        selected = frame.loc[mask]
        selected_priorities = priorities[mask]
        if not selected.empty:
            kept_frames.append(selected)
            kept_priorities.append(selected_priorities)
            running_total += len(selected)
    return kept_frames, kept_priorities


def load_temporal_split_dataset(
    path: str | Path,
    split_config: TemporalSplitConfig,
    split_name: str,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
    target_col: str | None = None,
    keep_all_positives: bool = False,
    sort_by: Sequence[str] | str | None = "datetime",
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Load one temporal split from parquet, optionally sampling at read time."""

    import pyarrow.dataset as ds

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {dataset_path}")

    requested_columns = list(columns) if columns is not None else get_modeling_dataset_columns(dataset_path)
    dataset_filter = _split_name_to_filter(split_config, split_name)
    dataset = ds.dataset(dataset_path, format="parquet")

    if max_rows is None:
        scanner = dataset.scanner(
            columns=requested_columns,
            filter=dataset_filter,
            batch_size=batch_size,
            use_threads=True,
        )
        frame = scanner.to_table().to_pandas()
        if sort_by is not None:
            frame = frame.sort_values(sort_by).reset_index(drop=True)
        return frame

    if target_col is None:
        raise ValueError("target_col is required when max_rows is provided.")

    scan_columns = list(dict.fromkeys(requested_columns + [target_col]))
    priority_columns = ["user_card_id", "datetime", "user", "card"]
    scan_columns = list(dict.fromkeys(scan_columns + [col for col in priority_columns if col not in scan_columns]))

    count_scanner = dataset.scanner(
        columns=[target_col],
        filter=dataset_filter,
        batch_size=batch_size,
        use_threads=True,
    )
    n_rows_total = 0
    n_positive_total = 0
    for batch in count_scanner.to_batches():
        target_values = batch.column(0).to_numpy(zero_copy_only=False)
        n_rows_total += len(target_values)
        n_positive_total += int(np.nansum(target_values))

    if n_rows_total <= max_rows:
        scanner = dataset.scanner(
            columns=requested_columns,
            filter=dataset_filter,
            batch_size=batch_size,
            use_threads=True,
        )
        frame = scanner.to_table().to_pandas()
        if sort_by is not None:
            frame = frame.sort_values(sort_by).reset_index(drop=True)
        return frame

    if keep_all_positives:
        positive_quota = min(n_positive_total, max_rows)
        negative_quota = max(0, max_rows - positive_quota)
    else:
        positive_quota = 0
        negative_quota = max_rows

    positive_frames: list[pd.DataFrame] = []
    negative_frames: list[pd.DataFrame] = []
    negative_priorities: list[np.ndarray] = []

    scanner = dataset.scanner(
        columns=scan_columns,
        filter=dataset_filter,
        batch_size=batch_size,
        use_threads=True,
    )
    for batch in scanner.to_batches():
        batch_frame = batch.to_pandas()
        if keep_all_positives:
            positive_batch = batch_frame.loc[batch_frame[target_col] == 1, requested_columns]
            if not positive_batch.empty:
                positive_frames.append(positive_batch)
            negative_batch = batch_frame.loc[batch_frame[target_col] == 0, :]
        else:
            positive_batch = pd.DataFrame()
            negative_batch = batch_frame

        if negative_quota > 0 and not negative_batch.empty:
            negative_batch_requested = negative_batch.loc[:, requested_columns]
            priorities = _stable_hash_priority(negative_batch)
            negative_frames.append(negative_batch_requested)
            negative_priorities.append(priorities)
            if sum(len(chunk) for chunk in negative_frames) > max(negative_quota * 2, batch_size):
                negative_frames, negative_priorities = _reduce_negative_reservoir(
                    negative_frames,
                    negative_priorities,
                    negative_quota,
                )

    if keep_all_positives:
        positive_frame = (
            pd.concat(positive_frames, axis=0, ignore_index=True)
            if positive_frames
            else pd.DataFrame(columns=requested_columns)
        )
    else:
        positive_frame = pd.DataFrame(columns=requested_columns)

    negative_frames, negative_priorities = _reduce_negative_reservoir(
        negative_frames,
        negative_priorities,
        negative_quota,
    )
    negative_frame = (
        pd.concat(negative_frames, axis=0, ignore_index=True)
        if negative_frames
        else pd.DataFrame(columns=requested_columns)
    )

    frame = pd.concat([positive_frame, negative_frame], axis=0, ignore_index=True)
    if len(frame) > max_rows:
        if keep_all_positives:
            positives = frame.loc[frame[target_col] == 1]
            negatives = frame.loc[frame[target_col] == 0]
            negative_take = max_rows - min(len(positives), max_rows)
            negatives = negatives.sample(n=max(negative_take, 0), random_state=RANDOM_STATE)
            frame = pd.concat([positives.head(max_rows), negatives], axis=0, ignore_index=True)
        else:
            frame = frame.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    if sort_by is not None:
        frame = frame.sort_values(sort_by).reset_index(drop=True)
    return frame.loc[:, requested_columns]


def summarize_temporal_split_from_parquet(
    path: str | Path,
    split_config: TemporalSplitConfig,
    target_col: str = "is_fraud",
) -> pd.DataFrame:
    """Summarize row counts and fraud counts for each temporal split from parquet."""

    rows: list[dict[str, object]] = []
    for split_name in ("train", "validation", "test", "excluded"):
        frame = load_temporal_split_dataset(
            path=path,
            split_config=split_config,
            split_name=split_name,
            columns=["year_month", target_col],
            max_rows=None,
            target_col=target_col,
            keep_all_positives=False,
            sort_by=None,
        )
        n_rows = int(len(frame))
        n_positive = int(frame[target_col].sum()) if n_rows else 0
        if split_name == "train":
            start_period, end_period = split_config.train_start, split_config.train_end
        elif split_name == "validation":
            start_period, end_period = split_config.validation_start, split_config.validation_end
        elif split_name == "test":
            start_period, end_period = split_config.test_start, split_config.test_end
        else:
            start_period = split_config.excluded_periods[0]
            end_period = split_config.excluded_periods[-1]
        rows.append(
            {
                "split": split_name,
                "n_rows": n_rows,
                "n_positive": n_positive,
                "fraud_rate": (n_positive / n_rows) if n_rows else 0.0,
                "start_period": start_period,
                "end_period": end_period,
            }
        )
    return pd.DataFrame(rows)


def summarize_temporal_split_monthly_from_parquet(
    path: str | Path,
    split_config: TemporalSplitConfig,
    target_col: str = "is_fraud",
) -> pd.DataFrame:
    """Return monthly counts for each split from parquet."""

    monthly_frames: list[pd.DataFrame] = []
    for split_name in ("train", "validation", "test", "excluded"):
        frame = load_temporal_split_dataset(
            path=path,
            split_config=split_config,
            split_name=split_name,
            columns=["year_month", target_col],
            max_rows=None,
            target_col=target_col,
            keep_all_positives=False,
            sort_by=None,
        )
        if frame.empty:
            continue
        month_summary = (
            frame.groupby("year_month", dropna=False)[target_col]
            .agg(n_rows="size", n_positive="sum")
            .reset_index()
        )
        month_summary["fraud_rate"] = month_summary["n_positive"] / month_summary["n_rows"]
        month_summary["split"] = split_name
        monthly_frames.append(month_summary)
    if not monthly_frames:
        return pd.DataFrame(columns=["year_month", "n_rows", "n_positive", "fraud_rate", "split"])
    return pd.concat(monthly_frames, axis=0, ignore_index=True)


def detect_target_column(columns: Sequence[str], preferred: str = "is_fraud") -> str:
    """Detect the target column name from a schema."""

    if preferred in columns:
        return preferred
    candidates = [col for col in columns if col.lower() in {"target", "label", "fraud", "isfraud"}]
    if not candidates:
        raise ValueError("No target column could be detected.")
    return candidates[0]


def detect_identifier_columns(columns: Sequence[str]) -> list[str]:
    """Return likely identifier/metadata columns that should not be used as predictors."""

    known = {
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
    }
    return [col for col in columns if col in known]


def build_candidate_feature_columns(columns: Sequence[str], target_col: str, identifier_columns: Sequence[str]) -> list[str]:
    """Return candidate predictor columns."""

    excluded = set(identifier_columns) | {target_col}
    return [col for col in columns if col not in excluded]


def split_features_target(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into X/y using explicit feature columns."""

    return frame.loc[:, list(feature_cols)].copy(), frame.loc[:, target_col].copy()


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric and categorical inputs."""

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, object, list[str]]] = [
        ("numeric", SklearnPipeline(steps=numeric_steps), list(numeric_features))
    ]

    if categorical_features:
        categorical_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_features)))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator_pipeline(preprocessor: ColumnTransformer, estimator: object) -> SklearnPipeline:
    """Wrap a sklearn estimator in a standard preprocessing pipeline."""

    return SklearnPipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])


def make_resampling_pipeline(preprocessor: ColumnTransformer, sampler: object, estimator: object):
    """Wrap preprocessing, sampling and estimator steps in an imblearn pipeline."""

    if ImbPipeline is None:
        raise ModuleNotFoundError("imblearn is required for resampling pipelines.")
    return ImbPipeline(steps=[("preprocessor", preprocessor), ("sampler", sampler), ("estimator", estimator)])


def build_random_forest_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    n_estimators: int = 300,
    n_jobs: int = 1,
) -> SklearnPipeline:
    """Create a RandomForest pipeline for imbalanced fraud detection."""

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    return make_estimator_pipeline(preprocessor, estimator)


def build_hist_gradient_boosting_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> SklearnPipeline:
    """Create a HistGradientBoosting pipeline."""

    estimator = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    return make_estimator_pipeline(preprocessor, estimator)

