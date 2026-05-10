"""Generate the compact v2 notebooks."""

from __future__ import annotations

from textwrap import dedent
from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook_1() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Análisis Exploratorio de Datos v2

            Este notebook conserva el flujo del original, pero concentra la lógica repetitiva en `src/features/transaction_eda.py`.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent
            elif not (PROJECT_ROOT / "src").exists():
                for parent in PROJECT_ROOT.parents:
                    if (parent / "src").exists():
                        PROJECT_ROOT = parent
                        break
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.utils.notebook_common import ensure_project_root_on_path, env_flag, env_int, env_int_list, append_suffix

            PROJECT_ROOT = ensure_project_root_on_path(Path.cwd())

            from src.features.transaction_eda import (
                build_core_eda_outputs,
                build_errors_risk_summary,
                build_numeric_correlation_sample,
                fraud_rate_by_group,
                get_transaction_eda_paths,
                load_raw_transactions,
                prepare_transaction_eda_frame,
                top_volume_and_risk_by_group,
            )

            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", 100)
            pd.set_option("display.float_format", "{:,.4f}".format)

            USE_SAMPLE = env_flag("FFDM_SAMPLE_RUN", False)
            SAMPLE_ROWS = env_int("FFDM_SAMPLE_ROWS", 50000)
            SAMPLE_ROW_GROUPS = env_int_list("FFDM_SAMPLE_ROW_GROUPS", [0])
            SAVE_OUTPUTS = env_flag("FFDM_SAVE_OUTPUTS", True)
            OUTPUT_SUFFIX = "_sample_v2" if USE_SAMPLE else ""

            PATHS = get_transaction_eda_paths(PROJECT_ROOT)
            """
        ),
        code(
            """
            raw_df = load_raw_transactions(
                PATHS,
                use_sample=USE_SAMPLE,
                sample_row_groups=SAMPLE_ROW_GROUPS,
                sample_rows=SAMPLE_ROWS,
            )

            errors_risk_summary = build_errors_risk_summary(raw_df)
            df_cards = prepare_transaction_eda_frame(raw_df)
            core_outputs = build_core_eda_outputs(df_cards)

            if SAVE_OUTPUTS:
                clean_output = append_suffix(PATHS.clean_parquet, OUTPUT_SUFFIX)
                df_cards.to_parquet(clean_output, index=False)
                print(f"Base limpia guardada en: {clean_output}")

            print(f"USE_SAMPLE={USE_SAMPLE} | filas={len(df_cards):,} | columnas={df_cards.shape[1]}")
            """
        ),
        code(
            """
            display(df_cards.head(3))
            display(core_outputs["unique_null_summary"].head(20))
            display(core_outputs["null_summary"].head(20))
            display(core_outputs["duplicate_summary"])
            display(errors_risk_summary.head(20))
            display(core_outputs["target_distribution"])
            """
        ),
        code(
            """
            display(core_outputs["amount_by_target"])
            display(core_outputs["amount_quantiles_by_target"])

            amount_boxplot_sample = df_cards.loc[df_cards["amount"] <= df_cards["amount"].quantile(0.99), ["is_fraud", "amount"]]
            if len(amount_boxplot_sample) > 200_000:
                amount_boxplot_sample = amount_boxplot_sample.sample(n=200_000, random_state=42)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(core_outputs["target_distribution"]["is_fraud"].astype(str), core_outputs["target_distribution"]["count"])
            axes[0].set_title("Distribución de la variable objetivo")
            axes[0].set_xlabel("Fraude")
            axes[0].set_ylabel("Número de transacciones")

            amount_boxplot_sample.boxplot(column="amount", by="is_fraud", grid=False, ax=axes[1])
            axes[1].set_title("Distribución de Amount por clase")
            axes[1].set_xlabel("Is Fraud")
            axes[1].set_ylabel("Amount")
            fig.suptitle("")
            fig.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            fraud_by_weekend = core_outputs["fraud_by_weekend"].copy()
            fraud_by_weekend["period_type"] = fraud_by_weekend["is_weekend"].map({0: "Día hábil", 1: "Fin de semana"})

            merchant_state_top, merchant_state_risk = top_volume_and_risk_by_group(df_cards, "merchant_state", min_count_for_risk=1000)
            merchant_top, _ = top_volume_and_risk_by_group(df_cards, "merchant_name", min_count_for_risk=1000)
            mcc_top, mcc_risk = top_volume_and_risk_by_group(df_cards, "mcc", min_count_for_risk=1000)

            display(core_outputs["fraud_by_year"])
            display(core_outputs["fraud_by_month"])
            display(core_outputs["fraud_by_hour"])
            display(fraud_by_weekend)
            display(merchant_state_top.head(20))
            display(merchant_state_risk.head(20))
            display(mcc_top.head(20))
            display(mcc_risk.head(20))
            display(merchant_top.head(20))
            """
        ),
        code(
            """
            numeric_columns = ["year", "month", "day", "hour", "amount", "mcc", "is_fraud"]
            correlation_matrix = build_numeric_correlation_sample(df_cards, numeric_columns=numeric_columns, max_rows=500000)
            display(correlation_matrix)

            plt.figure(figsize=(8, 6))
            plt.imshow(correlation_matrix)
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
            plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
            plt.colorbar()
            plt.title("Matriz de correlación - muestra")
            plt.tight_layout()
            plt.show()
            """
        ),
    ]
    return nb


def build_notebook_2() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Ingeniería de Características v2

            La lógica completa del notebook original se movió a `src/features/transaction_features.py`. Aquí solo se parametriza y se inspeccionan los artefactos.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path

            import pandas as pd

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent
            elif not (PROJECT_ROOT / "src").exists():
                for parent in PROJECT_ROOT.parents:
                    if (parent / "src").exists():
                        PROJECT_ROOT = parent
                        break
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.utils.notebook_common import ensure_project_root_on_path, env_flag, env_int, env_int_list

            PROJECT_ROOT = ensure_project_root_on_path(Path.cwd())

            from src.features.transaction_features import FeatureEngineeringConfig, run_feature_engineering_pipeline

            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", 100)
            pd.set_option("display.float_format", "{:,.4f}".format)

            USE_SAMPLE = env_flag("FFDM_SAMPLE_RUN", False)
            SAMPLE_ROWS = env_int("FFDM_SAMPLE_ROWS", 50000)
            SAMPLE_ROW_GROUPS = tuple(env_int_list("FFDM_SAMPLE_ROW_GROUPS", [0]))
            SAVE_OUTPUTS = env_flag("FFDM_SAVE_OUTPUTS", True)
            OUTPUT_SUFFIX = "_sample_v2" if USE_SAMPLE else ""

            DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
            config = FeatureEngineeringConfig(
                input_path=DATA_PROCESSED_DIR / "card_transactions_clean.parquet",
                output_transaction_features_path=DATA_PROCESSED_DIR / "transactions_modeling.parquet",
                output_user_card_snapshot_path=DATA_PROCESSED_DIR / "user_card_snapshot_features.parquet",
                use_sample=USE_SAMPLE,
                sample_row_groups=SAMPLE_ROW_GROUPS,
                sample_rows=SAMPLE_ROWS,
                save_outputs=SAVE_OUTPUTS,
                output_suffix=OUTPUT_SUFFIX,
            )
            """
        ),
        code(
            """
            artifacts = run_feature_engineering_pipeline(config)
            print(artifacts["summary"].to_string(index=False))
            print(f"Dataset transaccional: {artifacts['transaction_output_path']}")
            print(f"Snapshot por tarjeta: {artifacts['snapshot_output_path']}")
            """
        ),
        code(
            """
            display(artifacts["summary"])
            display(artifacts["feature_catalog"].head(25))
            display(artifacts["df_modeling"].head(3))
            display(artifacts["user_card_snapshot"].head(3))
            """
        ),
    ]
    return nb


def build_notebook_3() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Modeling Experiments v2

            El notebook mantiene el flujo del original, pero la ejecución vive en `src/models/experiment_runner.py`.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path

            import pandas as pd

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent
            elif not (PROJECT_ROOT / "src").exists():
                for parent in PROJECT_ROOT.parents:
                    if (parent / "src").exists():
                        PROJECT_ROOT = parent
                        break
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.utils.notebook_common import ensure_project_root_on_path, env_flag

            PROJECT_ROOT = ensure_project_root_on_path(Path.cwd())

            from src.models.experiment_runner import make_default_experiment_config, run_modeling_experiments

            pd.set_option("display.max_columns", 200)
            pd.set_option("display.width", 160)

            USE_SAMPLE = env_flag("FFDM_SAMPLE_RUN", False)
            SAVE_OUTPUTS = env_flag("FFDM_SAVE_OUTPUTS", True)
            OUTPUT_SUFFIX = "_sample_v2" if USE_SAMPLE else ""

            config = make_default_experiment_config(
                PROJECT_ROOT,
                sample_run=USE_SAMPLE,
                save_outputs=SAVE_OUTPUTS,
                output_suffix=OUTPUT_SUFFIX,
            )
            """
        ),
        code(
            """
            artifacts = run_modeling_experiments(config)
            print("Workflow completado.")
            """
        ),
        code(
            """
            display(artifacts["validation_summary_frame"])
            display(artifacts["split_summary"])
            display(artifacts["monthly_summary"].head(20))
            display(artifacts["ranking_frame"].head(20))
            """
        ),
        code(
            """
            display(
                artifacts["results_frame"][
                    [
                        "model_name",
                        "feature_subset",
                        "balancing_strategy",
                        "pr_auc",
                        "recall",
                        "f1",
                        "roc_auc",
                        "precision",
                        "train_time_seconds",
                        "predict_time_seconds",
                    ]
                ].head(20)
            )

            if not artifacts["omitted_models_frame"].empty:
                display(artifacts["omitted_models_frame"])
            """
        ),
    ]
    return nb


def build_notebook_4() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Selección Final del Modelo v2

            El flujo se compactó en `src/models/final_model_selection.py`, manteniendo la misma idea del notebook original: seleccionar candidatos en validación, reentrenar, fijar threshold y evaluar en test.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path

            import pandas as pd

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent
            elif not (PROJECT_ROOT / "src").exists():
                for parent in PROJECT_ROOT.parents:
                    if (parent / "src").exists():
                        PROJECT_ROOT = parent
                        break
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.utils.notebook_common import ensure_project_root_on_path, env_flag

            PROJECT_ROOT = ensure_project_root_on_path(Path.cwd())

            from src.models.final_model_selection import make_default_final_selection_config, run_final_model_selection

            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 180)

            USE_SAMPLE = env_flag("FFDM_SAMPLE_RUN", False)
            SAVE_OUTPUTS = env_flag("FFDM_SAVE_OUTPUTS", True)
            OUTPUT_SUFFIX = "_sample_v2" if USE_SAMPLE else ""

            config = make_default_final_selection_config(
                PROJECT_ROOT,
                sample_run=USE_SAMPLE,
                save_outputs=SAVE_OUTPUTS,
                output_suffix=OUTPUT_SUFFIX,
            )
            """
        ),
        code(
            """
            artifacts = run_final_model_selection(config)
            print("Selección final completada.")
            """
        ),
        code(
            """
            display(artifacts["top_3"])
            if not artifacts["unsupported_high_rank"].empty:
                display(artifacts["unsupported_high_rank"])
            display(artifacts["df_retrained_results"])
            """
        ),
        code(
            """
            display(artifacts["df_thresholds"].head(20))
            display(artifacts["df_final_features"].head(20))
            display(artifacts["final_summary"])
            print(artifacts["summary_text"])
            """
        ),
    ]
    return nb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebooks_dir = project_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "1_analisis_exploratorio_v2.ipynb": build_notebook_1(),
        "2_ingenieria_caracteristicas_v2.ipynb": build_notebook_2(),
        "3_modeling_experiments_cards_v2.ipynb": build_notebook_3(),
        "4_model_selection_cards_v2.ipynb": build_notebook_4(),
    }

    for name, notebook in targets.items():
        nbf.write(notebook, notebooks_dir / name)
        print(f"Generado: {notebooks_dir / name}")


if __name__ == "__main__":
    main()
