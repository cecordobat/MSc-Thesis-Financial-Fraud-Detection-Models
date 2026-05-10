"""Feature-engineering pipeline extracted from notebook 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.notebook_common import append_suffix, read_parquet_head, read_parquet_row_groups


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration for transaction feature engineering."""

    input_path: Path
    output_transaction_features_path: Path
    output_user_card_snapshot_path: Path
    use_sample: bool = False
    sample_row_groups: tuple[int, ...] = (0,)
    sample_rows: int = 50_000
    window_months: tuple[int, ...] = (3, 6, 9, 12)
    rare_frequency_threshold: float = 0.0001
    save_outputs: bool = True
    output_suffix: str = ""


def read_transactions(
    path: Path,
    use_sample: bool = False,
    row_groups: Sequence[int] | None = None,
    sample_rows: int = 50_000,
) -> pd.DataFrame:
    """Load the full clean parquet or only a sample."""

    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de entrada: {path}")

    if not use_sample:
        return pd.read_parquet(path)
    if row_groups:
        return read_parquet_row_groups(path, row_groups=row_groups)
    return read_parquet_head(path, n_rows=sample_rows)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Validate the minimum columns expected by notebook 2."""

    required_columns = [
        "user",
        "card",
        "year",
        "month",
        "day",
        "time",
        "amount",
        "amount_abs",
        "amount_is_negative",
        "use_chip",
        "merchant_name",
        "merchant_city",
        "merchant_state",
        "zip",
        "mcc",
        "is_fraud",
        "merchant_state_was_missing",
        "zip_was_missing",
        "user_card_id",
        "hour",
        "datetime",
        "day_of_week",
        "is_weekend",
    ]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Columnas faltantes en la base limpia: {missing_columns}")


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Divide two aligned series while avoiding inf and NaN."""

    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def clean_numeric_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Replace non-finite values column by column to limit copies."""

    for column in columns:
        if column not in data.columns:
            continue

        series = data[column]
        changed = False

        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
            changed = True

        if pd.api.types.is_float_dtype(series):
            values = series.to_numpy(copy=False)
            inf_mask = np.isinf(values)
            if inf_mask.any():
                series = series.mask(inf_mask)
                changed = True

        if series.isna().any():
            series = series.fillna(fill_value)
            changed = True

        if changed:
            data[column] = series

    return data


def add_base_time_and_amount_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Create the notebook 2 base calendar and amount features."""

    prepared = df.copy()
    prepared["_row_order"] = np.arange(len(prepared), dtype="int64")
    prepared["datetime"] = pd.to_datetime(prepared["datetime"])
    prepared["year_month"] = prepared["datetime"].dt.to_period("M").dt.to_timestamp()
    prepared["year"] = prepared["datetime"].dt.year
    prepared["month"] = prepared["datetime"].dt.month
    prepared["quarter"] = prepared["datetime"].dt.quarter
    prepared["day_of_month"] = prepared["datetime"].dt.day
    prepared["hour"] = prepared["datetime"].dt.hour
    prepared["day_of_week"] = prepared["datetime"].dt.dayofweek
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype(int)
    evaluation_month = prepared["year_month"].max()

    clean_columns = {
        "merchant_name": "merchant_name_clean",
        "merchant_city": "merchant_city_clean",
        "merchant_state": "merchant_state_clean",
        "zip": "zip_clean",
        "mcc": "mcc_clean",
        "use_chip": "use_chip_clean",
    }
    for raw_col, clean_col in clean_columns.items():
        prepared[clean_col] = prepared[raw_col].fillna("UNKNOWN").astype(str)

    prepared["amount_log"] = np.log1p(prepared["amount_abs"])
    prepared["amount_is_zero_flag"] = (prepared["amount_abs"] == 0).astype(int)
    prepared["amount_below_1_flag"] = (prepared["amount_abs"] < 1).astype(int)
    prepared["amount_below_5_flag"] = (prepared["amount_abs"] < 5).astype(int)
    prepared["amount_is_round_10_flag"] = np.isclose(prepared["amount_abs"] % 10, 0, atol=1e-6).astype(int)
    prepared["amount_is_round_100_flag"] = np.isclose(prepared["amount_abs"] % 100, 0, atol=1e-6).astype(int)

    quantiles = prepared["amount_abs"].quantile([0.90, 0.95, 0.99])
    prepared["amount_above_global_p90_flag"] = (prepared["amount_abs"] > quantiles.loc[0.90]).astype(int)
    prepared["amount_above_global_p95_flag"] = (prepared["amount_abs"] > quantiles.loc[0.95]).astype(int)
    prepared["amount_above_global_p99_flag"] = (prepared["amount_abs"] > quantiles.loc[0.99]).astype(int)

    use_chip_lower = prepared["use_chip_clean"].str.lower()
    prepared["is_online_transaction"] = use_chip_lower.str.contains("online", na=False).astype(int)
    prepared["is_chip_transaction"] = use_chip_lower.str.contains("chip", na=False).astype(int)
    prepared["is_swipe_transaction"] = use_chip_lower.str.contains("swipe", na=False).astype(int)

    prepared["is_night"] = prepared["hour"].between(0, 5).astype(int)
    prepared["is_morning"] = prepared["hour"].between(6, 11).astype(int)
    prepared["is_afternoon"] = prepared["hour"].between(12, 17).astype(int)
    prepared["is_evening"] = prepared["hour"].between(18, 23).astype(int)
    prepared["is_business_hours"] = prepared["hour"].between(8, 18).astype(int)
    prepared["is_late_night"] = prepared["hour"].between(0, 3).astype(int)

    return prepared, evaluation_month


def add_user_card_history_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create cumulative user-card features."""

    prepared = df.sort_values(["user_card_id", "datetime", "_row_order"]).reset_index(drop=True).copy()
    prepared["uc_tx_count_hist"] = prepared.groupby("user_card_id").cumcount()
    prepared["uc_amount_sum_hist"] = prepared.groupby("user_card_id")["amount_abs"].cumsum() - prepared["amount_abs"]
    prepared["uc_amount_mean_hist"] = safe_divide(prepared["uc_amount_sum_hist"], prepared["uc_tx_count_hist"])
    prepared["uc_amount_max_hist"] = (
        prepared.groupby("user_card_id")["amount_abs"].cummax().groupby(prepared["user_card_id"]).shift(1).fillna(0)
    )
    prepared["uc_prev_datetime"] = prepared.groupby("user_card_id")["datetime"].shift(1)
    prepared["uc_days_since_prev_tx"] = (
        (prepared["datetime"] - prepared["uc_prev_datetime"]).dt.total_seconds().div(86_400).fillna(-1)
    )

    prepared["uc_no_history_flag"] = (prepared["uc_tx_count_hist"] == 0).astype(int)
    prepared["first_transaction_flag"] = (prepared["uc_days_since_prev_tx"] == -1).astype(int)
    prepared["long_inactivity_30d_flag"] = (prepared["uc_days_since_prev_tx"] > 30).astype(int)
    prepared["long_inactivity_90d_flag"] = (prepared["uc_days_since_prev_tx"] > 90).astype(int)
    prepared["amount_to_hist_mean_ratio"] = safe_divide(prepared["amount_abs"], prepared["uc_amount_mean_hist"])
    prepared["amount_above_hist_max_flag"] = (
        (prepared["uc_amount_max_hist"] > 0) & (prepared["amount_abs"] > prepared["uc_amount_max_hist"])
    ).astype(int)
    prepared["amount_gt_3x_hist_mean_flag"] = (
        (prepared["uc_amount_mean_hist"] > 0) & (prepared["amount_abs"] > 3 * prepared["uc_amount_mean_hist"])
    ).astype(int)

    history_columns = [
        "uc_tx_count_hist",
        "uc_amount_sum_hist",
        "uc_amount_mean_hist",
        "uc_amount_max_hist",
        "uc_days_since_prev_tx",
        "uc_no_history_flag",
        "first_transaction_flag",
        "long_inactivity_30d_flag",
        "long_inactivity_90d_flag",
        "amount_to_hist_mean_ratio",
        "amount_above_hist_max_flag",
        "amount_gt_3x_hist_mean_flag",
    ]
    return prepared, history_columns


def build_monthly_user_card_features(transactions: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    """Build user-card rolling monthly features without leakage."""

    monthly_source = transactions[
        [
            "user_card_id",
            "year_month",
            "amount_abs",
            "is_online_transaction",
            "amount_is_negative",
            "merchant_name_clean",
            "mcc_clean",
        ]
    ].copy()
    monthly_source["amount_abs_sq"] = monthly_source["amount_abs"] ** 2

    monthly = (
        monthly_source.groupby(["user_card_id", "year_month"], observed=True)
        .agg(
            uc_month_tx_count=("amount_abs", "size"),
            uc_month_amount_sum=("amount_abs", "sum"),
            uc_month_amount_sq_sum=("amount_abs_sq", "sum"),
            uc_month_amount_min=("amount_abs", "min"),
            uc_month_amount_max=("amount_abs", "max"),
            uc_month_online_tx_count=("is_online_transaction", "sum"),
            uc_month_negative_tx_count=("amount_is_negative", "sum"),
            uc_month_unique_merchants=("merchant_name_clean", "nunique"),
            uc_month_unique_mcc=("mcc_clean", "nunique"),
        )
        .reset_index()
    )

    user_cards = pd.Index(transactions["user_card_id"].drop_duplicates(), name="user_card_id")
    months = pd.date_range(transactions["year_month"].min(), transactions["year_month"].max(), freq="MS", name="year_month")
    full_index = pd.MultiIndex.from_product([user_cards, months])

    monthly = (
        monthly.set_index(["user_card_id", "year_month"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["user_card_id", "year_month"])
    )

    zero_fill_cols = [
        "uc_month_tx_count",
        "uc_month_amount_sum",
        "uc_month_amount_sq_sum",
        "uc_month_online_tx_count",
        "uc_month_negative_tx_count",
        "uc_month_unique_merchants",
        "uc_month_unique_mcc",
    ]
    monthly[zero_fill_cols] = monthly[zero_fill_cols].fillna(0)

    grouped = monthly.groupby("user_card_id", sort=False)
    for window in windows:
        tx_col = f"uc_tx_count_{window}m"
        sum_col = f"uc_amount_sum_{window}m"
        sq_sum_col = f"uc_amount_sq_sum_{window}m"
        mean_col = f"uc_amount_mean_{window}m"
        std_col = f"uc_amount_std_{window}m"

        monthly[tx_col] = grouped["uc_month_tx_count"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).sum()
        )
        monthly[sum_col] = grouped["uc_month_amount_sum"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).sum()
        )
        monthly[sq_sum_col] = grouped["uc_month_amount_sq_sum"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).sum()
        )
        monthly[f"uc_amount_min_{window}m"] = grouped["uc_month_amount_min"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).min()
        )
        monthly[f"uc_amount_max_{window}m"] = grouped["uc_month_amount_max"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).max()
        )
        monthly[f"uc_online_tx_count_{window}m"] = grouped["uc_month_online_tx_count"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).sum()
        )
        monthly[f"uc_negative_tx_count_{window}m"] = grouped["uc_month_negative_tx_count"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).sum()
        )
        monthly[f"uc_unique_merchants_{window}m"] = grouped["uc_month_unique_merchants"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).sum()
        )
        monthly[f"uc_unique_mcc_{window}m"] = grouped["uc_month_unique_mcc"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).sum()
        )

        monthly[mean_col] = safe_divide(monthly[sum_col], monthly[tx_col])
        variance = (
            monthly[sq_sum_col] - (monthly[sum_col] ** 2 / monthly[tx_col].replace(0, np.nan))
        ) / (monthly[tx_col] - 1).replace(0, np.nan)
        monthly[std_col] = np.sqrt(variance.clip(lower=0)).fillna(0)

        monthly[f"uc_avg_ticket_{window}m"] = safe_divide(monthly[sum_col], monthly[tx_col])
        monthly[f"uc_amount_cv_{window}m"] = safe_divide(monthly[std_col], monthly[mean_col])
        monthly[f"uc_amount_max_mean_ratio_{window}m"] = safe_divide(monthly[f"uc_amount_max_{window}m"], monthly[mean_col])
        monthly[f"uc_amount_min_mean_ratio_{window}m"] = safe_divide(monthly[f"uc_amount_min_{window}m"], monthly[mean_col])
        monthly[f"uc_online_tx_rate_{window}m"] = safe_divide(monthly[f"uc_online_tx_count_{window}m"], monthly[tx_col])
        monthly[f"uc_negative_tx_rate_{window}m"] = safe_divide(
            monthly[f"uc_negative_tx_count_{window}m"], monthly[tx_col]
        )

    return monthly.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_window_feature_column_lists(window_months: Sequence[int]) -> tuple[list[str], list[str]]:
    """Return the monthly-window feature lists used later in the pipeline."""

    extended_window_columns: list[str] = []
    for window in window_months:
        extended_window_columns.extend(
            [
                f"uc_tx_count_{window}m",
                f"uc_amount_sum_{window}m",
                f"uc_amount_mean_{window}m",
                f"uc_amount_min_{window}m",
                f"uc_amount_max_{window}m",
                f"uc_amount_std_{window}m",
                f"uc_avg_ticket_{window}m",
                f"uc_amount_cv_{window}m",
                f"uc_amount_max_mean_ratio_{window}m",
                f"uc_amount_min_mean_ratio_{window}m",
                f"uc_online_tx_count_{window}m",
                f"uc_online_tx_rate_{window}m",
                f"uc_negative_tx_count_{window}m",
                f"uc_negative_tx_rate_{window}m",
                f"uc_unique_merchants_{window}m",
                f"uc_unique_mcc_{window}m",
            ]
        )

    extended_change_columns = [
        "uc_tx_count_ratio_3m_6m",
        "uc_tx_count_ratio_3m_12m",
        "uc_tx_count_ratio_6m_12m",
        "uc_amount_sum_ratio_3m_6m",
        "uc_amount_sum_ratio_3m_12m",
        "uc_amount_sum_ratio_6m_12m",
        "uc_amount_mean_delta_3m_12m",
        "uc_amount_mean_ratio_3m_12m",
        "uc_no_tx_3m_flag",
        "uc_no_tx_6m_flag",
        "uc_activity_spike_3m_vs_12m_flag",
        "uc_amount_spike_3m_vs_12m_flag",
        "uc_mean_amount_increase_3m_vs_12m_flag",
    ]
    return extended_window_columns, extended_change_columns


def add_monthly_window_features(df: pd.DataFrame, window_months: Sequence[int]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge monthly rolling features back to the transaction frame."""

    monthly_user_card = build_monthly_user_card_features(df, window_months)
    extended_window_columns, extended_change_columns = build_window_feature_column_lists(window_months)

    monthly_user_card["uc_tx_count_ratio_3m_6m"] = safe_divide(monthly_user_card["uc_tx_count_3m"], monthly_user_card["uc_tx_count_6m"])
    monthly_user_card["uc_tx_count_ratio_3m_12m"] = safe_divide(monthly_user_card["uc_tx_count_3m"], monthly_user_card["uc_tx_count_12m"])
    monthly_user_card["uc_tx_count_ratio_6m_12m"] = safe_divide(monthly_user_card["uc_tx_count_6m"], monthly_user_card["uc_tx_count_12m"])
    monthly_user_card["uc_amount_sum_ratio_3m_6m"] = safe_divide(monthly_user_card["uc_amount_sum_3m"], monthly_user_card["uc_amount_sum_6m"])
    monthly_user_card["uc_amount_sum_ratio_3m_12m"] = safe_divide(
        monthly_user_card["uc_amount_sum_3m"], monthly_user_card["uc_amount_sum_12m"]
    )
    monthly_user_card["uc_amount_sum_ratio_6m_12m"] = safe_divide(
        monthly_user_card["uc_amount_sum_6m"], monthly_user_card["uc_amount_sum_12m"]
    )
    monthly_user_card["uc_amount_mean_delta_3m_12m"] = (
        monthly_user_card["uc_amount_mean_3m"] - monthly_user_card["uc_amount_mean_12m"]
    )
    monthly_user_card["uc_amount_mean_ratio_3m_12m"] = safe_divide(
        monthly_user_card["uc_amount_mean_3m"], monthly_user_card["uc_amount_mean_12m"]
    )
    monthly_user_card["uc_no_tx_3m_flag"] = (monthly_user_card["uc_tx_count_3m"] == 0).astype(int)
    monthly_user_card["uc_no_tx_6m_flag"] = (monthly_user_card["uc_tx_count_6m"] == 0).astype(int)
    monthly_user_card["uc_activity_spike_3m_vs_12m_flag"] = (monthly_user_card["uc_tx_count_ratio_3m_12m"] > 0.60).astype(int)
    monthly_user_card["uc_amount_spike_3m_vs_12m_flag"] = (monthly_user_card["uc_amount_sum_ratio_3m_12m"] > 0.60).astype(int)
    monthly_user_card["uc_mean_amount_increase_3m_vs_12m_flag"] = (
        monthly_user_card["uc_amount_mean_ratio_3m_12m"] > 1.50
    ).astype(int)

    monthly_features = monthly_user_card[["user_card_id", "year_month"] + extended_window_columns + extended_change_columns].copy()
    duplicate_feature_cols = [col for col in monthly_features.columns if col in df.columns and col not in {"user_card_id", "year_month"}]
    prepared = df.drop(columns=duplicate_feature_cols) if duplicate_feature_cols else df.copy()
    prepared = prepared.merge(monthly_features, on=["user_card_id", "year_month"], how="left", validate="many_to_one")
    prepared = clean_numeric_columns(prepared, extended_window_columns + extended_change_columns)
    return prepared, extended_window_columns, extended_change_columns


def add_amount_window_comparison_features(
    df: pd.DataFrame,
    window_months: Sequence[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Compare each transaction amount against its rolling history."""

    amount_feature_data: dict[str, pd.Series] = {}
    for window in window_months:
        mean_col = f"uc_amount_mean_{window}m"
        max_col = f"uc_amount_max_{window}m"
        std_col = f"uc_amount_std_{window}m"

        amount_feature_data[f"amount_to_{window}m_mean_ratio"] = safe_divide(df["amount_abs"], df[mean_col])
        amount_feature_data[f"amount_to_{window}m_max_ratio"] = safe_divide(df["amount_abs"], df[max_col])
        amount_feature_data[f"amount_minus_{window}m_mean"] = df["amount_abs"] - df[mean_col]
        amount_feature_data[f"amount_zscore_{window}m"] = safe_divide(df["amount_abs"] - df[mean_col], df[std_col])
        amount_feature_data[f"amount_above_{window}m_mean_flag"] = (
            (df[mean_col] > 0) & (df["amount_abs"] > df[mean_col])
        ).astype("int8")
        amount_feature_data[f"amount_above_2x_{window}m_mean_flag"] = (
            (df[mean_col] > 0) & (df["amount_abs"] > 2 * df[mean_col])
        ).astype("int8")
        amount_feature_data[f"amount_above_3x_{window}m_mean_flag"] = (
            (df[mean_col] > 0) & (df["amount_abs"] > 3 * df[mean_col])
        ).astype("int8")
        amount_feature_data[f"amount_above_{window}m_max_flag"] = (
            (df[max_col] > 0) & (df["amount_abs"] > df[max_col])
        ).astype("int8")

    existing_amount_cols = [column for column in amount_feature_data if column in df.columns]
    prepared = df.drop(columns=existing_amount_cols) if existing_amount_cols else df.copy()
    prepared = pd.concat([prepared, pd.DataFrame(amount_feature_data, index=prepared.index)], axis=1)
    prepared["amount_gt_3x_12m_mean_flag"] = prepared["amount_above_3x_12m_mean_flag"]

    amount_window_comparison_cols = [
        column
        for column in prepared.columns
        if column.startswith(("amount_to_", "amount_minus_", "amount_zscore_", "amount_above_"))
    ]
    prepared = clean_numeric_columns(prepared, amount_window_comparison_cols)
    return prepared, amount_window_comparison_cols


def add_channel_history_and_novelty_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Add historical channel and novelty features per user-card."""

    prepared = df.sort_values(["user_card_id", "datetime", "_row_order"]).reset_index(drop=True).copy()

    channel_columns = [
        ("is_online_transaction", "online"),
        ("is_chip_transaction", "chip"),
        ("is_swipe_transaction", "swipe"),
    ]
    channel_hist_cols: list[str] = []
    for source_col, channel_name in channel_columns:
        count_col = f"uc_{channel_name}_tx_count_hist"
        rate_col = f"uc_{channel_name}_tx_rate_hist"
        first_col = f"{channel_name}_first_time_for_card_flag"
        prepared[count_col] = prepared.groupby("user_card_id")[source_col].cumsum() - prepared[source_col]
        prepared[rate_col] = safe_divide(prepared[count_col], prepared["uc_tx_count_hist"])
        prepared[first_col] = ((prepared[source_col] == 1) & (prepared[count_col] == 0)).astype(int)
        channel_hist_cols.extend([count_col, rate_col, first_col])

    novelty_specs = [
        ("merchant_name_clean", "merchant"),
        ("mcc_clean", "mcc"),
        ("merchant_city_clean", "city"),
        ("merchant_state_clean", "state"),
        ("zip_clean", "zip"),
    ]
    novelty_cols: list[str] = []
    novelty_count_cols: list[str] = []
    unique_history_cols: list[str] = []
    for source_col, short_name in novelty_specs:
        new_col = f"new_{short_name}_for_card_flag"
        count_col = f"uc_{short_name}_tx_count_hist"
        unique_col = f"uc_unique_{short_name}s_hist"

        prepared[count_col] = prepared.groupby(["user_card_id", source_col]).cumcount()
        prepared[new_col] = (prepared[count_col] == 0).astype(int)
        prepared[unique_col] = prepared.groupby("user_card_id")[new_col].cumsum() - prepared[new_col]

        novelty_cols.append(new_col)
        novelty_count_cols.append(count_col)
        unique_history_cols.append(unique_col)

    return prepared, channel_hist_cols, novelty_cols, novelty_count_cols + unique_history_cols


def add_user_history_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add user-level cumulative history features."""

    cards_per_user = df.groupby("user")["card"].nunique().rename("n_cards_user")
    prepared = df.merge(cards_per_user, on="user", how="left", validate="many_to_one")
    prepared = prepared.sort_values(["user", "datetime", "_row_order"]).reset_index(drop=True)

    prepared["user_tx_count_hist"] = prepared.groupby("user").cumcount()
    prepared["user_amount_sum_hist"] = prepared.groupby("user")["amount_abs"].cumsum() - prepared["amount_abs"]
    prepared["user_amount_mean_hist"] = safe_divide(prepared["user_amount_sum_hist"], prepared["user_tx_count_hist"])
    prepared["user_amount_max_hist"] = (
        prepared.groupby("user")["amount_abs"].cummax().groupby(prepared["user"]).shift(1).fillna(0)
    )
    prepared["user_prev_datetime"] = prepared.groupby("user")["datetime"].shift(1)
    prepared["user_days_since_prev_tx"] = (
        (prepared["datetime"] - prepared["user_prev_datetime"]).dt.total_seconds().div(86_400).fillna(-1)
    )
    prepared["amount_to_user_hist_mean_ratio"] = safe_divide(prepared["amount_abs"], prepared["user_amount_mean_hist"])
    prepared["amount_to_user_hist_max_ratio"] = safe_divide(prepared["amount_abs"], prepared["user_amount_max_hist"])
    prepared["amount_above_user_hist_max_flag"] = (
        (prepared["user_amount_max_hist"] > 0) & (prepared["amount_abs"] > prepared["user_amount_max_hist"])
    ).astype(int)

    user_hist_cols = [
        "n_cards_user",
        "user_tx_count_hist",
        "user_amount_sum_hist",
        "user_amount_mean_hist",
        "user_amount_max_hist",
        "user_days_since_prev_tx",
        "amount_to_user_hist_mean_ratio",
        "amount_to_user_hist_max_ratio",
        "amount_above_user_hist_max_flag",
    ]
    return prepared, user_hist_cols


def add_same_day_velocity_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add same-day transaction velocity features."""

    prepared = df.sort_values(["user_card_id", "datetime", "_row_order"]).reset_index(drop=True).copy()
    prepared["transaction_date"] = prepared["datetime"].dt.date
    prepared["uc_same_day_tx_count_prev"] = prepared.groupby(["user_card_id", "transaction_date"]).cumcount()
    prepared["uc_same_day_amount_sum_prev"] = (
        prepared.groupby(["user_card_id", "transaction_date"])["amount_abs"].cumsum() - prepared["amount_abs"]
    )
    prepared["uc_same_day_amount_mean_prev"] = safe_divide(
        prepared["uc_same_day_amount_sum_prev"], prepared["uc_same_day_tx_count_prev"]
    )
    prepared["multiple_tx_same_day_flag"] = (prepared["uc_same_day_tx_count_prev"] >= 1).astype(int)
    prepared["high_velocity_same_day_flag"] = (prepared["uc_same_day_tx_count_prev"] >= 5).astype(int)
    prepared["very_high_velocity_same_day_flag"] = (prepared["uc_same_day_tx_count_prev"] >= 10).astype(int)

    velocity_cols = [
        "uc_same_day_tx_count_prev",
        "uc_same_day_amount_sum_prev",
        "uc_same_day_amount_mean_prev",
        "multiple_tx_same_day_flag",
        "high_velocity_same_day_flag",
        "very_high_velocity_same_day_flag",
    ]
    return prepared, velocity_cols


def add_global_frequency_features(df: pd.DataFrame, rare_frequency_threshold: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add global frequency features and rare-category flags."""

    prepared = df.copy()
    frequency_specs = [
        ("merchant_name_clean", "merchant_tx_count_global", "merchant_frequency_global"),
        ("mcc_clean", "mcc_tx_count_global", "mcc_frequency_global"),
        ("merchant_state_clean", "merchant_state_tx_count_global", "merchant_state_frequency_global"),
        ("merchant_city_clean", "merchant_city_tx_count_global", "merchant_city_frequency_global"),
        ("zip_clean", "zip_tx_count_global", "zip_frequency_global"),
        ("use_chip_clean", "use_chip_tx_count_global", "use_chip_frequency_global"),
    ]

    for source_col, count_col, frequency_col in frequency_specs:
        counts = prepared[source_col].value_counts(dropna=False)
        prepared[count_col] = prepared[source_col].map(counts).fillna(0).astype("int64")
        prepared[frequency_col] = prepared[count_col] / len(prepared)

    frequency_cols = [column for _, count_col, frequency_col in frequency_specs for column in (count_col, frequency_col)]

    rare_specs = [
        ("merchant_frequency_global", "rare_merchant_flag"),
        ("mcc_frequency_global", "rare_mcc_flag"),
        ("merchant_city_frequency_global", "rare_city_flag"),
        ("merchant_state_frequency_global", "rare_state_flag"),
        ("zip_frequency_global", "rare_zip_flag"),
    ]
    for frequency_col, flag_col in rare_specs:
        prepared[flag_col] = (prepared[frequency_col] < rare_frequency_threshold).astype(int)

    rare_cols = [flag_col for _, flag_col in rare_specs]
    return prepared, frequency_cols, rare_cols


def build_numeric_feature_catalog(
    df: pd.DataFrame,
    extended_window_columns: list[str],
    extended_change_columns: list[str],
    amount_window_comparison_cols: list[str],
    channel_hist_cols: list[str],
    novelty_cols: list[str],
    novelty_count_and_unique_cols: list[str],
    user_hist_cols: list[str],
    velocity_cols: list[str],
    frequency_cols: list[str],
    rare_cols: list[str],
) -> list[str]:
    """Return the ordered numeric feature list used for modeling."""

    base_numeric_features = [
        "amount_abs",
        "amount_log",
        "amount_is_negative",
        "merchant_state_was_missing",
        "zip_was_missing",
        "hour",
        "day_of_week",
        "is_weekend",
        "quarter",
        "day_of_month",
        "n_cards_user",
        "is_online_transaction",
        "is_chip_transaction",
        "is_swipe_transaction",
        "is_night",
        "is_morning",
        "is_afternoon",
        "is_evening",
        "is_business_hours",
        "is_late_night",
        "amount_above_global_p90_flag",
        "amount_above_global_p95_flag",
        "amount_above_global_p99_flag",
        "amount_is_zero_flag",
        "amount_below_1_flag",
        "amount_below_5_flag",
        "amount_is_round_10_flag",
        "amount_is_round_100_flag",
        "uc_tx_count_hist",
        "uc_amount_sum_hist",
        "uc_amount_mean_hist",
        "uc_amount_max_hist",
        "uc_days_since_prev_tx",
        "uc_no_history_flag",
        "first_transaction_flag",
        "long_inactivity_30d_flag",
        "long_inactivity_90d_flag",
        "amount_to_hist_mean_ratio",
        "amount_above_hist_max_flag",
        "amount_gt_3x_hist_mean_flag",
        "amount_gt_3x_12m_mean_flag",
    ]

    numeric_features = list(
        dict.fromkeys(
            base_numeric_features
            + extended_window_columns
            + extended_change_columns
            + amount_window_comparison_cols
            + channel_hist_cols
            + novelty_cols
            + novelty_count_and_unique_cols
            + user_hist_cols
            + velocity_cols
            + frequency_cols
            + rare_cols
        )
    )
    missing_features = [column for column in numeric_features if column not in df.columns]
    if missing_features:
        raise ValueError(f"Variables numericas no creadas: {missing_features}")
    return numeric_features


def build_modeling_exports(
    df: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the modeling dataset, the user-card snapshot and the feature catalog."""

    prepared = clean_numeric_columns(df, numeric_features)

    feature_nulls = pd.Series({column: int(prepared[column].isna().sum()) for column in numeric_features}).sort_values(
        ascending=False
    )
    non_finite_counts = pd.Series(
        {
            column: int(np.isinf(prepared[column].to_numpy(copy=False)).sum())
            if pd.api.types.is_float_dtype(prepared[column])
            else 0
            for column in numeric_features
        }
    ).sort_values(ascending=False)
    if feature_nulls.sum() != 0 or non_finite_counts.sum() != 0:
        raise ValueError("Hay variables numericas con nulos o infinitos despues de la limpieza final.")

    feature_catalog = pd.DataFrame(
        {
            "feature": numeric_features,
            "dtype": [str(prepared[column].dtype) for column in numeric_features],
            "missing_values": [prepared[column].isna().sum() for column in numeric_features],
        }
    )

    identifier_columns = [
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
    ]
    modeling_columns = list(dict.fromkeys(identifier_columns + ["is_fraud"] + numeric_features))
    df_modeling = prepared.loc[:, modeling_columns].copy()

    snapshot_columns = ["user", "card", "user_card_id", "datetime", "year_month"] + numeric_features
    user_card_snapshot = (
        prepared.groupby("user_card_id", sort=False)
        .tail(1)
        .loc[:, snapshot_columns]
        .rename(columns={"datetime": "snapshot_datetime", "year_month": "snapshot_month"})
        .reset_index(drop=True)
    )
    return df_modeling, user_card_snapshot, feature_catalog


def write_parquet_in_row_groups(
    data: pd.DataFrame,
    path: Path,
    batch_size: int = 250_000,
) -> None:
    """Write parquet in batches to avoid a giant in-memory Arrow table."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for start in range(0, len(data), batch_size):
            batch = data.iloc[start : start + batch_size]
            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def run_feature_engineering_pipeline(config: FeatureEngineeringConfig) -> dict[str, object]:
    """Run the complete notebook 2 feature-engineering flow."""

    df = read_transactions(
        config.input_path,
        use_sample=config.use_sample,
        row_groups=config.sample_row_groups,
        sample_rows=config.sample_rows,
    )
    validate_input_columns(df)

    df, evaluation_month = add_base_time_and_amount_features(df)
    df, _ = add_user_card_history_features(df)
    df, extended_window_columns, extended_change_columns = add_monthly_window_features(df, config.window_months)
    df, amount_window_comparison_cols = add_amount_window_comparison_features(df, config.window_months)
    df, channel_hist_cols, novelty_cols, novelty_count_and_unique_cols = add_channel_history_and_novelty_features(df)
    df, user_hist_cols = add_user_history_features(df)
    df, velocity_cols = add_same_day_velocity_features(df)
    df, frequency_cols, rare_cols = add_global_frequency_features(df, config.rare_frequency_threshold)

    numeric_features = build_numeric_feature_catalog(
        df=df,
        extended_window_columns=extended_window_columns,
        extended_change_columns=extended_change_columns,
        amount_window_comparison_cols=amount_window_comparison_cols,
        channel_hist_cols=channel_hist_cols,
        novelty_cols=novelty_cols,
        novelty_count_and_unique_cols=novelty_count_and_unique_cols,
        user_hist_cols=user_hist_cols,
        velocity_cols=velocity_cols,
        frequency_cols=frequency_cols,
        rare_cols=rare_cols,
    )

    df_modeling, user_card_snapshot, feature_catalog = build_modeling_exports(df, numeric_features)

    transaction_output_path = append_suffix(config.output_transaction_features_path, config.output_suffix)
    snapshot_output_path = append_suffix(config.output_user_card_snapshot_path, config.output_suffix)
    if config.save_outputs:
        write_parquet_in_row_groups(df_modeling, transaction_output_path)
        user_card_snapshot.to_parquet(snapshot_output_path, index=False)

    summary = pd.DataFrame(
        {
            "n_rows": [len(df)],
            "n_numeric_features": [len(numeric_features)],
            "fraud_rate": [df["is_fraud"].mean()],
            "evaluation_month": [evaluation_month],
        }
    )

    return {
        "data": df,
        "df_modeling": df_modeling,
        "user_card_snapshot": user_card_snapshot,
        "feature_catalog": feature_catalog,
        "numeric_features": numeric_features,
        "evaluation_month": evaluation_month,
        "summary": summary,
        "transaction_output_path": transaction_output_path,
        "snapshot_output_path": snapshot_output_path,
    }
