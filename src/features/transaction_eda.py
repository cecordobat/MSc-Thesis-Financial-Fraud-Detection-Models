"""Reusable helpers behind notebook 1 (EDA)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.convert_transactions_to_parquet import convertir_parquet
from src.utils.eda import (
    convert_data_types,
    finalize_transaction_columns,
    get_null_summary,
    get_target_distribution,
    get_unique_and_null_values_summary,
    impute_location_fields,
)
from src.utils.notebook_common import read_parquet_head, read_parquet_row_groups


@dataclass(frozen=True)
class TransactionEDAPaths:
    """Resolved input and output paths for notebook 1."""

    raw_csv: Path
    raw_parquet: Path
    clean_parquet: Path


def get_transaction_eda_paths(project_root: Path) -> TransactionEDAPaths:
    """Return the data paths used by the EDA notebook."""

    processed_dir = project_root / "data" / "processed"
    return TransactionEDAPaths(
        raw_csv=project_root / "data" / "original" / "card_transaction.csv",
        raw_parquet=processed_dir / "card_transaction.parquet",
        clean_parquet=processed_dir / "card_transactions_clean.parquet",
    )


def _clean_raw_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Match the original notebook column-name normalization."""

    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return cleaned


def ensure_raw_parquet(paths: TransactionEDAPaths, chunksize: int = 2_000_000) -> Path:
    """Create the raw parquet cache when it does not exist yet."""

    if not paths.raw_parquet.exists():
        convertir_parquet(str(paths.raw_csv), str(paths.raw_parquet), chunksize=chunksize)
    return paths.raw_parquet


def load_raw_transactions(
    paths: TransactionEDAPaths,
    use_sample: bool = False,
    sample_row_groups: list[int] | None = None,
    sample_rows: int = 50_000,
) -> pd.DataFrame:
    """Load the raw transaction dataset or a sample from it."""

    raw_parquet = ensure_raw_parquet(paths)
    if use_sample:
        if sample_row_groups:
            return read_parquet_row_groups(raw_parquet, row_groups=sample_row_groups)
        return read_parquet_head(raw_parquet, n_rows=sample_rows)
    return pd.read_parquet(raw_parquet)


def build_errors_risk_summary(
    df: pd.DataFrame,
    errors_col: str = "errors?",
    target_col: str = "is_fraud?",
) -> pd.DataFrame:
    """Summarize fraud risk by raw error code."""

    if errors_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(columns=["error_clean", "count", "fraud_count", "fraud_rate"])

    analysis = pd.DataFrame(
        {
            "error_clean": df[errors_col].fillna("NO_ERROR").astype("string").str.strip(),
            "is_fraud_binary": (
                df[target_col].astype("string").str.strip().map({"No": 0, "Yes": 1}).astype("Int8")
            ),
        }
    )

    return (
        analysis.groupby("error_clean", dropna=False)
        .agg(
            count=("is_fraud_binary", "size"),
            fraud_count=("is_fraud_binary", "sum"),
            fraud_rate=("is_fraud_binary", "mean"),
        )
        .reset_index()
        .sort_values(["fraud_rate", "count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def prepare_transaction_eda_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the data preparation used by notebook 1."""

    prepared = _clean_raw_column_names(df)
    prepared = convert_data_types(prepared)
    prepared = impute_location_fields(prepared)

    prepared["user_card_id"] = prepared["user"].astype("string") + "_" + prepared["card"].astype("string")
    prepared = finalize_transaction_columns(prepared)

    if "is_fraud?" in prepared.columns:
        prepared = prepared.rename(columns={"is_fraud?": "is_fraud"})
    prepared["is_fraud"] = (
        prepared["is_fraud"].astype("string").str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).astype("Int8")
    )

    prepared["amount_is_negative"] = (prepared["amount"] < 0).astype("Int8")
    prepared["amount_abs"] = prepared["amount"].abs()
    prepared["hour"] = prepared["time"].astype("string").str.slice(0, 2).astype("int64")
    prepared["datetime"] = pd.to_datetime(
        prepared["year"].astype(str)
        + "-"
        + prepared["month"].astype(str).str.zfill(2)
        + "-"
        + prepared["day"].astype(str).str.zfill(2)
        + " "
        + prepared["time"].astype(str),
        errors="coerce",
    )
    prepared["day_of_week"] = prepared["datetime"].dt.dayofweek
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype("int8")
    return prepared


def build_duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate-row counts and percentages."""

    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = (duplicate_count / len(df) * 100) if len(df) else 0.0
    return pd.DataFrame(
        {
            "metric": ["duplicated_rows", "duplicated_percentage"],
            "value": [duplicate_count, duplicate_percentage],
        }
    )


def summarize_numeric_by_target(
    df: pd.DataFrame,
    numeric_col: str,
    target_col: str = "is_fraud",
) -> pd.DataFrame:
    """Summarize one numeric variable by the target."""

    return (
        df.groupby(target_col)[numeric_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )


def build_amount_quantiles_by_target(
    df: pd.DataFrame,
    amount_col: str = "amount",
    target_col: str = "is_fraud",
) -> pd.DataFrame:
    """Return amount quantiles by target class."""

    return (
        df.groupby(target_col)[amount_col]
        .quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .reset_index()
        .rename(columns={"level_1": "quantile", amount_col: "amount"})
    )


def fraud_rate_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "is_fraud",
    min_count: int | None = None,
    sort_by_group: bool = True,
) -> pd.DataFrame:
    """Compute count, fraud count and fraud rate by group."""

    summary = (
        df.groupby(group_col, dropna=False)
        .agg(
            count=(target_col, "size"),
            fraud_count=(target_col, "sum"),
            fraud_rate=(target_col, "mean"),
        )
        .reset_index()
    )
    if min_count is not None:
        summary = summary.loc[summary["count"] >= min_count]
    if sort_by_group:
        summary = summary.sort_values(group_col)
    return summary.reset_index(drop=True)


def top_volume_and_risk_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "is_fraud",
    top_n: int = 20,
    min_count_for_risk: int = 1_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top categories by volume and by filtered fraud risk."""

    grouped = fraud_rate_by_group(df, group_col=group_col, target_col=target_col, sort_by_group=False)
    top_volume = grouped.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    top_risk = (
        grouped.loc[grouped["count"] >= min_count_for_risk]
        .sort_values("fraud_rate", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top_volume, top_risk


def build_core_eda_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the most important tables used by notebook 1."""

    outputs = {
        "unique_null_summary": get_unique_and_null_values_summary(df),
        "null_summary": get_null_summary(df),
        "duplicate_summary": build_duplicate_summary(df),
        "target_distribution": get_target_distribution(df, "is_fraud"),
        "amount_by_target": summarize_numeric_by_target(df, "amount"),
        "amount_quantiles_by_target": build_amount_quantiles_by_target(df),
        "fraud_by_year": fraud_rate_by_group(df, "year"),
        "fraud_by_month": fraud_rate_by_group(df, "month"),
        "fraud_by_day": fraud_rate_by_group(df, "day"),
        "fraud_by_hour": fraud_rate_by_group(df, "hour"),
        "fraud_by_weekend": fraud_rate_by_group(df, "is_weekend"),
        "fraud_by_card": fraud_rate_by_group(df, "card"),
    }
    return outputs


def build_numeric_correlation_sample(
    df: pd.DataFrame,
    numeric_columns: list[str],
    max_rows: int = 500_000,
) -> pd.DataFrame:
    """Build the notebook 1 numeric correlation matrix on a sample."""

    sample = df.loc[:, numeric_columns].dropna()
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=42)
    return sample.corr(numeric_only=True)
