"""Reusable helpers for an interpretable rule-based fraud baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.train_models import (
    TemporalSplitConfig,
    get_modeling_dataset_columns,
    load_temporal_split_dataset,
)
DEFAULT_TEMPORAL_SPLIT = TemporalSplitConfig(
    train_start="1991-01",
    train_end="2017-12",
    validation_start="2018-01",
    validation_end="2018-12",
    test_start="2019-01",
    test_end="2019-10",
    excluded_periods=("2019-11", "2019-12", "2020-01", "2020-02"),
)

TARGET_COLUMN = "is_fraud"
MCC_COLUMN = "mcc"

# These variables are produced using full-dataset aggregates in notebook 2 and are
# intentionally excluded from the rule baseline to avoid temporal leakage.
GLOBAL_LEAKAGE_COLUMNS = {
    "amount_above_global_p90_flag",
    "amount_above_global_p95_flag",
    "amount_above_global_p99_flag",
    "merchant_tx_count_global",
    "merchant_frequency_global",
    "mcc_tx_count_global",
    "mcc_frequency_global",
    "merchant_state_tx_count_global",
    "merchant_state_frequency_global",
    "merchant_city_tx_count_global",
    "merchant_city_frequency_global",
    "zip_tx_count_global",
    "zip_frequency_global",
    "use_chip_tx_count_global",
    "use_chip_frequency_global",
    "rare_merchant_flag",
    "rare_mcc_flag",
    "rare_city_flag",
    "rare_state_flag",
    "rare_zip_flag",
}

IDENTIFIER_COLUMNS = [
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
]

RULE_COLUMN_CANDIDATES = {
    "high_amount_flags": [
        "amount_above_hist_max_flag",
        "amount_above_user_hist_max_flag",
        "amount_gt_3x_hist_mean_flag",
        "amount_gt_3x_12m_mean_flag",
    ],
    "high_amount_thresholds": [
        ("amount_zscore_12m", 3.0),
        ("amount_to_12m_mean_ratio", 3.0),
        ("amount_to_user_hist_mean_ratio", 3.0),
        ("amount_to_hist_mean_ratio", 3.0),
    ],
    "online_flag": ["is_online_transaction"],
    "online_novelty_flags": [
        "online_first_time_for_card_flag",
        "chip_first_time_for_card_flag",
        "swipe_first_time_for_card_flag",
    ],
    "new_merchant_flags": [
        "new_merchant_for_card_flag",
        "new_mcc_for_card_flag",
    ],
    "unusual_location_flags": [
        "new_city_for_card_flag",
        "new_state_for_card_flag",
        "new_zip_for_card_flag",
    ],
    "velocity_flags": [
        "high_velocity_same_day_flag",
        "very_high_velocity_same_day_flag",
        "uc_activity_spike_3m_vs_12m_flag",
        "uc_amount_spike_3m_vs_12m_flag",
    ],
    "velocity_thresholds": [
        ("uc_same_day_tx_count_prev", 5.0),
    ],
    "operational_anomaly_flags": [
        "amount_is_negative",
        "amount_is_zero_flag",
        "merchant_state_was_missing",
        "zip_was_missing",
    ],
    "inactivity_flags": [
        "uc_no_history_flag",
        "first_transaction_flag",
        "long_inactivity_90d_flag",
    ],
    "channel_shift_flags": [
        "online_first_time_for_card_flag",
        "chip_first_time_for_card_flag",
        "swipe_first_time_for_card_flag",
    ],
}


@dataclass(frozen=True)
class RuleArtifacts:
    """Artifacts required to score the rule baseline."""

    risky_mccs: tuple[str, ...]
    overall_train_fraud_rate: float
    rule_columns_used: dict[str, list[str]]
    mcc_risk_table: pd.DataFrame


def _to_month_period(value: str | pd.Period) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(value, freq="M")


def _period_start_timestamp(value: str | pd.Period) -> pd.Timestamp:
    return _to_month_period(value).to_timestamp(how="start")


def _split_name_to_filter(
    split_config: TemporalSplitConfig,
    split_name: str,
    year_month_col: str = "year_month",
):
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


def available_rule_columns(dataset_columns: Sequence[str]) -> list[str]:
    """Return the modeling columns required by the rule baseline."""

    requested = set(IDENTIFIER_COLUMNS + [TARGET_COLUMN])
    for candidates in RULE_COLUMN_CANDIDATES.values():
        if not candidates:
            continue
        if isinstance(candidates[0], tuple):
            requested.update(column for column, _ in candidates)  # type: ignore[misc]
        else:
            requested.update(candidates)  # type: ignore[arg-type]

    available = [column for column in requested if column in dataset_columns and column not in GLOBAL_LEAKAGE_COLUMNS]
    return list(dict.fromkeys(available))


def get_dataset_overview(path: str | Path) -> dict[str, Any]:
    """Inspect the parquet schema and summarize usable rule inputs."""

    columns = get_modeling_dataset_columns(path)
    safe_rule_columns = available_rule_columns(columns)
    excluded_global = sorted(set(columns).intersection(GLOBAL_LEAKAGE_COLUMNS))
    return {
        "n_columns": len(columns),
        "all_columns": columns,
        "safe_rule_columns": safe_rule_columns,
        "excluded_global_columns": excluded_global,
    }


def _series_false(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=frame.index)


def _any_positive_flag(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.Series, list[str]]:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return _series_false(frame), []

    flags = []
    for column in available:
        series = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        flags.append(series.gt(0))

    mask = flags[0].copy()
    for extra in flags[1:]:
        mask |= extra
    return mask, available


def _any_threshold(frame: pd.DataFrame, column_thresholds: Sequence[tuple[str, float]]) -> tuple[pd.Series, list[str]]:
    expressions: list[str] = []
    mask = _series_false(frame)
    for column, threshold in column_thresholds:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        mask |= series.ge(float(threshold))
        expressions.append(f"{column}>={threshold:.2f}")
    return mask, expressions


def _is_online(frame: pd.DataFrame) -> pd.Series:
    if "is_online_transaction" not in frame.columns:
        return _series_false(frame)
    return pd.to_numeric(frame["is_online_transaction"], errors="coerce").fillna(0).gt(0)


def _clean_mcc_value(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    return str(value)


def compute_train_mcc_risk_table(
    path: str | Path,
    split_config: TemporalSplitConfig = DEFAULT_TEMPORAL_SPLIT,
    mcc_col: str = MCC_COLUMN,
    target_col: str = TARGET_COLUMN,
    min_support: int = 1_000,
    top_n: int = 10,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Aggregate train fraud risk by MCC using batch scans to limit memory."""

    import pyarrow.dataset as ds

    dataset = ds.dataset(Path(path), format="parquet")
    scanner = dataset.scanner(
        columns=[mcc_col, target_col],
        filter=_split_name_to_filter(split_config, "train"),
        batch_size=batch_size,
        use_threads=True,
    )

    counts: dict[str, int] = {}
    fraud_counts: dict[str, int] = {}
    total_rows = 0
    total_frauds = 0

    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue

        frame[mcc_col] = frame[mcc_col].map(_clean_mcc_value)
        frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce").fillna(0).astype(int)

        grouped = frame.groupby(mcc_col, dropna=False)[target_col].agg(["size", "sum"]).reset_index()
        for row in grouped.itertuples(index=False):
            mcc_value = getattr(row, mcc_col)
            counts[mcc_value] = counts.get(mcc_value, 0) + int(row.size)
            fraud_counts[mcc_value] = fraud_counts.get(mcc_value, 0) + int(row.sum)

        total_rows += int(len(frame))
        total_frauds += int(frame[target_col].sum())

    overall_rate = float(total_frauds / total_rows) if total_rows else 0.0
    risk_frame = pd.DataFrame(
        {
            mcc_col: list(counts.keys()),
            "tx_count": [counts[key] for key in counts],
            "fraud_count": [fraud_counts.get(key, 0) for key in counts],
        }
    )
    if risk_frame.empty:
        raise ValueError("No fue posible construir la tabla de riesgo por MCC sobre train.")

    risk_frame["fraud_rate"] = risk_frame["fraud_count"] / risk_frame["tx_count"]
    risk_frame["lift_vs_train"] = np.where(overall_rate > 0, risk_frame["fraud_rate"] / overall_rate, np.nan)
    risk_frame["eligible_high_risk"] = (
        (risk_frame["tx_count"] >= int(min_support)) & (risk_frame["fraud_rate"] > overall_rate)
    ).astype(int)

    eligible = risk_frame[risk_frame["eligible_high_risk"] == 1].copy()
    if eligible.empty:
        eligible = risk_frame[risk_frame["tx_count"] >= int(min_support)].copy()
    if eligible.empty:
        eligible = risk_frame.copy()

    eligible = eligible.sort_values(["fraud_rate", "fraud_count", "tx_count"], ascending=[False, False, False]).head(top_n)
    high_risk_values = set(eligible[mcc_col].astype(str))

    risk_frame["selected_for_rule"] = risk_frame[mcc_col].astype(str).isin(high_risk_values).astype(int)
    risk_frame = risk_frame.sort_values(["selected_for_rule", "fraud_rate", "tx_count"], ascending=[False, False, False]).reset_index(drop=True)
    risk_frame.attrs["overall_train_fraud_rate"] = overall_rate
    return risk_frame


def fit_rule_artifacts(
    path: str | Path,
    split_config: TemporalSplitConfig = DEFAULT_TEMPORAL_SPLIT,
    min_mcc_support: int = 1_000,
    top_n_mcc: int = 10,
    batch_size: int = 250_000,
) -> RuleArtifacts:
    """Estimate train-only artifacts required by the rule baseline."""

    columns = get_modeling_dataset_columns(path)
    mcc_risk_table = compute_train_mcc_risk_table(
        path=path,
        split_config=split_config,
        min_support=min_mcc_support,
        top_n=top_n_mcc,
        batch_size=batch_size,
    )
    risky_mccs = tuple(sorted(mcc_risk_table.loc[mcc_risk_table["selected_for_rule"] == 1, MCC_COLUMN].astype(str).unique()))

    used_columns: dict[str, list[str]] = {}
    for rule_name, candidates in RULE_COLUMN_CANDIDATES.items():
        if candidates and isinstance(candidates[0], tuple):
            used_columns[rule_name] = [column for column, _ in candidates if column in columns and column not in GLOBAL_LEAKAGE_COLUMNS]  # type: ignore[misc]
        else:
            used_columns[rule_name] = [column for column in candidates if column in columns and column not in GLOBAL_LEAKAGE_COLUMNS]  # type: ignore[arg-type]

    return RuleArtifacts(
        risky_mccs=risky_mccs,
        overall_train_fraud_rate=float(mcc_risk_table.attrs.get("overall_train_fraud_rate", np.nan)),
        rule_columns_used=used_columns,
        mcc_risk_table=mcc_risk_table,
    )


def load_rule_split(
    path: str | Path,
    split_name: str,
    split_config: TemporalSplitConfig = DEFAULT_TEMPORAL_SPLIT,
    sort_by: Sequence[str] | str | None = "datetime",
) -> pd.DataFrame:
    """Load only the columns required by the rule baseline for a split."""

    columns = available_rule_columns(get_modeling_dataset_columns(path))
    return load_temporal_split_dataset(
        path=path,
        split_config=split_config,
        split_name=split_name,
        columns=columns,
        sort_by=sort_by,
    )


def build_rule_score_frame(
    frame: pd.DataFrame,
    artifacts: RuleArtifacts,
    mcc_col: str = MCC_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build individual rule triggers and a score summary frame."""

    if frame.empty:
        raise ValueError("The input frame is empty.")

    rule_masks: dict[str, pd.Series] = {}
    rule_logic: dict[str, list[str]] = {}

    high_amount_flag_mask, high_amount_flag_cols = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["high_amount_flags"])
    high_amount_threshold_mask, high_amount_threshold_exprs = _any_threshold(
        frame,
        RULE_COLUMN_CANDIDATES["high_amount_thresholds"],
    )
    rule_masks["rule_high_amount_deviation"] = high_amount_flag_mask | high_amount_threshold_mask
    rule_logic["rule_high_amount_deviation"] = high_amount_flag_cols + high_amount_threshold_exprs

    online_novelty_mask, online_novelty_cols = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["online_novelty_flags"])
    online_base = _is_online(frame)
    rule_masks["rule_online_channel_risk"] = online_base & online_novelty_mask
    rule_logic["rule_online_channel_risk"] = ["is_online_transaction"] + online_novelty_cols

    merchant_novelty_mask, merchant_novelty_cols = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["new_merchant_flags"])
    rule_masks["rule_new_merchant_or_category"] = merchant_novelty_mask
    rule_logic["rule_new_merchant_or_category"] = merchant_novelty_cols

    location_mask, location_cols = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["unusual_location_flags"])
    rule_masks["rule_unusual_location"] = location_mask
    rule_logic["rule_unusual_location"] = location_cols

    velocity_flag_mask, velocity_flag_cols = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["velocity_flags"])
    velocity_threshold_mask, velocity_threshold_exprs = _any_threshold(
        frame,
        RULE_COLUMN_CANDIDATES["velocity_thresholds"],
    )
    rule_masks["rule_velocity_spike"] = velocity_flag_mask | velocity_threshold_mask
    rule_logic["rule_velocity_spike"] = velocity_flag_cols + velocity_threshold_exprs

    operational_mask, operational_cols = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["operational_anomaly_flags"])
    rule_masks["rule_operational_anomaly"] = operational_mask
    rule_logic["rule_operational_anomaly"] = operational_cols

    mcc_values = frame[mcc_col].map(_clean_mcc_value) if mcc_col in frame.columns else pd.Series("UNKNOWN", index=frame.index)
    high_risk_mcc_mask = mcc_values.astype(str).isin(set(artifacts.risky_mccs))
    rule_masks["rule_high_risk_mcc"] = high_risk_mcc_mask
    rule_logic["rule_high_risk_mcc"] = [f"{mcc_col} in risky_mccs_train"]

    inactivity_mask, inactivity_cols = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["inactivity_flags"])
    channel_shift_mask, channel_shift_cols = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["channel_shift_flags"])
    rule_masks["rule_inactivity_plus_change"] = inactivity_mask & (
        merchant_novelty_mask | location_mask | online_base | channel_shift_mask
    )
    rule_logic["rule_inactivity_plus_change"] = inactivity_cols + merchant_novelty_cols + location_cols + ["is_online_transaction"] + channel_shift_cols

    rules_frame = pd.DataFrame({name: mask.astype(int) for name, mask in rule_masks.items()}, index=frame.index)
    if rules_frame.shape[1] == 0:
        raise ValueError("No fue posible construir ninguna regla con las columnas disponibles.")

    n_rules = int(rules_frame.shape[1])
    score_frame = pd.DataFrame(index=frame.index)
    score_frame["rules_triggered"] = rules_frame.sum(axis=1)
    score_frame["rule_score"] = score_frame["rules_triggered"] / float(n_rules)
    score_frame["rules_triggered_pct"] = score_frame["rule_score"]

    return rules_frame, score_frame


def score_rule_frame(
    frame: pd.DataFrame,
    artifacts: RuleArtifacts,
    mcc_col: str = MCC_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the rule trigger matrix and the final score vector."""

    if frame.empty:
        raise ValueError("The input frame is empty.")

    rule_masks: dict[str, pd.Series] = {}

    high_amount_flag_mask, _ = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["high_amount_flags"])
    high_amount_threshold_mask, _ = _any_threshold(frame, RULE_COLUMN_CANDIDATES["high_amount_thresholds"])
    rule_masks["rule_high_amount_deviation"] = high_amount_flag_mask | high_amount_threshold_mask

    online_novelty_mask, _ = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["online_novelty_flags"])
    online_base = _is_online(frame)
    rule_masks["rule_online_channel_risk"] = online_base & online_novelty_mask

    merchant_novelty_mask, _ = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["new_merchant_flags"])
    rule_masks["rule_new_merchant_or_category"] = merchant_novelty_mask

    location_mask, _ = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["unusual_location_flags"])
    rule_masks["rule_unusual_location"] = location_mask

    velocity_flag_mask, _ = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["velocity_flags"])
    velocity_threshold_mask, _ = _any_threshold(frame, RULE_COLUMN_CANDIDATES["velocity_thresholds"])
    rule_masks["rule_velocity_spike"] = velocity_flag_mask | velocity_threshold_mask

    operational_mask, _ = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["operational_anomaly_flags"])
    rule_masks["rule_operational_anomaly"] = operational_mask

    mcc_values = frame[mcc_col].map(_clean_mcc_value) if mcc_col in frame.columns else pd.Series("UNKNOWN", index=frame.index)
    rule_masks["rule_high_risk_mcc"] = mcc_values.astype(str).isin(set(artifacts.risky_mccs))

    inactivity_mask, _ = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["inactivity_flags"])
    channel_shift_mask, _ = _any_positive_flag(frame, RULE_COLUMN_CANDIDATES["channel_shift_flags"])
    rule_masks["rule_inactivity_plus_change"] = inactivity_mask & (
        merchant_novelty_mask | location_mask | online_base | channel_shift_mask
    )

    rules_frame = pd.DataFrame({name: mask.astype(int) for name, mask in rule_masks.items()}, index=frame.index)
    scores = rules_frame.sum(axis=1) / float(rules_frame.shape[1])
    return rules_frame, scores


def describe_rule_set(frame: pd.DataFrame, artifacts: RuleArtifacts) -> pd.DataFrame:
    """Summarize rule logic and trigger rates for a given frame."""

    rules_frame, scores = score_rule_frame(frame, artifacts)
    descriptions = {
        "rule_high_amount_deviation": "Monto alto respecto al comportamiento historico de la tarjeta o del usuario.",
        "rule_online_channel_risk": "Transaccion online con evidencia de cambio o novedad de canal.",
        "rule_new_merchant_or_category": "Comercio o MCC no observado previamente para la tarjeta.",
        "rule_unusual_location": "Ciudad, estado o codigo postal no observado previamente para la tarjeta.",
        "rule_velocity_spike": "Alta velocidad/frecuencia intradia o pico de actividad reciente.",
        "rule_operational_anomaly": "Anomalias operativas como montos negativos, ceros o ubicaciones faltantes.",
        "rule_high_risk_mcc": "MCC con elevada tasa historica de fraude en train.",
        "rule_inactivity_plus_change": "Reaparicion tras inactividad combinada con cambio de canal, comercio o ubicacion.",
    }
    rows = []
    for rule_name in rules_frame.columns:
        rows.append(
            {
                "rule_name": rule_name,
                "description": descriptions[rule_name],
                "trigger_rate": float(rules_frame[rule_name].mean()),
                "trigger_count": int(rules_frame[rule_name].sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("trigger_rate", ascending=False).reset_index(drop=True)
    summary.attrs["n_rules"] = int(rules_frame.shape[1])
    summary.attrs["score_mean"] = float(scores.mean())
    return summary


def build_rule_threshold_table(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    n_rules: int,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate a threshold grid using the same metrics family as notebook 4."""

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    y_true_array = np.asarray(y_true, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    n_obs = int(len(y_true_array))
    n_positive = int(y_true_array.sum())
    n_negative = int(n_obs - n_positive)
    positive_rate = float(y_true_array.mean()) if n_obs else 0.0
    pr_auc = float(average_precision_score(y_true_array, scores_array))
    roc_auc = float(roc_auc_score(y_true_array, scores_array))

    unique_scores, inverse = np.unique(scores_array, return_inverse=True)
    positive_counts = np.bincount(inverse, weights=y_true_array, minlength=len(unique_scores)).astype(int)
    total_counts = np.bincount(inverse, minlength=len(unique_scores)).astype(int)
    negative_counts = total_counts - positive_counts

    descending_order = np.argsort(unique_scores)[::-1]
    unique_scores_desc = unique_scores[descending_order]
    tp_cumsum = np.cumsum(positive_counts[descending_order])
    fp_cumsum = np.cumsum(negative_counts[descending_order])

    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        threshold = float(threshold)
        predicted_mask = unique_scores_desc >= threshold

        if predicted_mask.any():
            last_idx = int(np.flatnonzero(predicted_mask)[-1])
            tp = int(tp_cumsum[last_idx])
            fp = int(fp_cumsum[last_idx])
        else:
            tp = 0
            fp = 0

        fn = int(n_positive - tp)
        tn = int(n_negative - fp)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / n_positive) if n_positive > 0 else 0.0
        specificity = float(tn / n_negative) if n_negative > 0 else 0.0
        balanced_accuracy = float((recall + specificity) / 2)
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = float((tp + tn) / n_obs) if n_obs > 0 else 0.0

        rows.append(
            {
                "threshold": threshold,
                "n_obs": n_obs,
                "n_positive": n_positive,
                "positive_rate": positive_rate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pr_auc": pr_auc,
                "balanced_accuracy": balanced_accuracy,
                "roc_auc": roc_auc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "accuracy": accuracy,
                "n_predicted_positive": int(tp + fp),
                "min_rules_required": int(max(1, math.ceil(threshold * n_rules - 1e-12))),
            }
        )

    return pd.DataFrame(rows)


def select_best_threshold(
    threshold_table: pd.DataFrame,
    criterion: str = "f1",
) -> pd.Series:
    """Select the best validation threshold with deterministic tie-breaking."""

    if threshold_table.empty:
        raise ValueError("Threshold table is empty.")
    if criterion not in threshold_table.columns:
        raise ValueError(f"Criterion '{criterion}' not found in threshold table.")

    return (
        threshold_table.sort_values([criterion, "recall", "precision", "threshold"], ascending=[False, False, False, True])
        .iloc[0]
        .copy()
    )


def compare_with_ml_validation(
    model_results_path: str | Path,
    rule_validation_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Build a comparable validation table against existing ML experiments."""

    frame = pd.read_csv(model_results_path)
    validation = frame[frame["split"] == "validation"].copy()
    validation = validation.sort_values(["pr_auc", "recall", "f1", "precision"], ascending=[False, False, False, False])
    leaders = validation.groupby("model_name", as_index=False).head(1).copy()
    leaders["source"] = "ml_validation"
    leaders["comparison_scope"] = "validation"

    rule_row = {
        "model_name": "rule_based_binary",
        "feature_subset": "interpretable_rules",
        "balancing_strategy": "none",
        "pr_auc": rule_validation_metrics["pr_auc"],
        "roc_auc": rule_validation_metrics["roc_auc"],
        "recall": rule_validation_metrics["recall"],
        "precision": rule_validation_metrics["precision"],
        "f1": rule_validation_metrics["f1"],
        "threshold_used": rule_validation_metrics["threshold"],
        "source": "rule_validation",
        "comparison_scope": "validation",
        "accuracy": rule_validation_metrics.get("accuracy"),
        "tp": rule_validation_metrics["tp"],
        "fp": rule_validation_metrics["fp"],
        "tn": rule_validation_metrics["tn"],
        "fn": rule_validation_metrics["fn"],
    }

    leaders = leaders[
        [
            "model_name",
            "feature_subset",
            "balancing_strategy",
            "pr_auc",
            "roc_auc",
            "recall",
            "precision",
            "f1",
            "threshold_used",
            "tn",
            "fp",
            "fn",
            "tp",
            "source",
            "comparison_scope",
        ]
    ].copy()
    leaders["accuracy"] = (leaders["tp"] + leaders["tn"]) / (leaders["tp"] + leaders["tn"] + leaders["fp"] + leaders["fn"])

    comparison = pd.concat([leaders, pd.DataFrame([rule_row])], axis=0, ignore_index=True)
    return comparison.sort_values(["pr_auc", "recall", "f1"], ascending=[False, False, False]).reset_index(drop=True)


def compare_with_ml_test(
    final_comparison_path: str | Path,
    rule_test_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Build a test comparison table using notebook 4 outputs."""

    frame = pd.read_csv(final_comparison_path)
    frame["source"] = "ml_test"
    frame["comparison_scope"] = "test"
    rule_f1 = rule_test_metrics["f1"] if "f1" in rule_test_metrics else rule_test_metrics["f1_score"]
    rule_threshold = (
        rule_test_metrics["threshold"]
        if "threshold" in rule_test_metrics
        else rule_test_metrics["threshold_from_validation"]
    )
    rule_n_obs = rule_test_metrics["n_obs"] if "n_obs" in rule_test_metrics else rule_test_metrics["test_rows_used"]

    rule_row = {
        "rank": np.nan,
        "model_name": "rule_based_binary",
        "feature_subset": "interpretable_rules",
        "balancing_strategy": "none",
        "n_features": np.nan,
        "pr_auc_test": rule_test_metrics["pr_auc"],
        "roc_auc_test": rule_test_metrics["roc_auc"],
        "recall_test": rule_test_metrics["recall"],
        "precision_test": rule_test_metrics["precision"],
        "f1_test": rule_f1,
        "threshold_from_validation": rule_threshold,
        "train_rows_used": np.nan,
        "test_rows_used": rule_n_obs,
        "model_key": "rule_based_binary",
        "source": "rule_test",
        "comparison_scope": "test",
    }

    comparison = pd.concat([frame, pd.DataFrame([rule_row])], axis=0, ignore_index=True)
    return comparison.sort_values(["pr_auc_test", "recall_test", "f1_test"], ascending=[False, False, False]).reset_index(drop=True)
