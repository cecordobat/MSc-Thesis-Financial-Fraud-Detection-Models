"""Generate the rule-based baseline notebook for transactional fraud detection."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return new_code_cell(dedent(text).strip() + "\n")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        md(
            """
            # Modelo Basado en Reglas para Deteccion de Fraude Transaccional

            Este notebook construye una linea base interpretable basada en reglas para compararla con los modelos de Machine Learning ya implementados en el proyecto. El flujo reutiliza las variables generadas por `notebooks/2_ingenieria_caracteristicas.ipynb`, respeta exactamente las particiones temporales usadas en los notebooks de modelamiento y fija el umbral exclusivamente con el conjunto de validacion.

            **Objetivos del notebook**

            - Cargar el dataset final de modelamiento derivado de la ingenieria de caracteristicas.
            - Inspeccionar las variables disponibles antes de definir reglas.
            - Construir un puntaje interpretable a partir de reglas binarias transaccionales.
            - Convertir ese puntaje en un clasificador binario usando un threshold optimizado solo en validacion.
            - Evaluar el resultado final en test con las mismas metricas reportadas en `notebooks/4_model_selection_cards.ipynb`.
            - Comparar el baseline basado en reglas frente a los modelos de Machine Learning ya evaluados en el proyecto.
            """
        ),
        md(
            """
            # Introduccion

            El enfoque basado en reglas busca representar patrones de fraude que resultan faciles de interpretar para analistas y comites de riesgo. A diferencia de los modelos supervisados complejos, este baseline no aprende fronteras no lineales ni combinaciones de alta dimensionalidad; en su lugar, agrega evidencia de riesgo mediante un conjunto reducido de reglas comprensibles.

            Metodologicamente, el notebook sigue dos restricciones estrictas:

            1. Las reglas se construyen exclusivamente a partir de variables ya disponibles en el dataset final de modelamiento.
            2. El conjunto `test` se utiliza una unica vez para la evaluacion final. La seleccion del threshold se realiza unicamente en `validation`.

            Adicionalmente, se excluyen del baseline las variables de frecuencia global calculadas sobre todo el dataset, dado que ese tipo de agregaciones puede introducir leakage temporal cuando se usan sin reestimacion por split.
            """
        ),
        md("# Carga de librerias"),
        code(
            """
            from pathlib import Path
            import json
            import math
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import Markdown, display
            from sklearn.metrics import (
                auc,
                classification_report,
                confusion_matrix,
                precision_recall_curve,
                roc_curve,
            )

            NOTEBOOK_DIR = Path.cwd().resolve()
            PROJECT_ROOT = next((path for path in [NOTEBOOK_DIR, *NOTEBOOK_DIR.parents] if (path / "src").exists()), NOTEBOOK_DIR)
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.models.rule_based_model import (
                DEFAULT_TEMPORAL_SPLIT,
                GLOBAL_LEAKAGE_COLUMNS,
                TARGET_COLUMN,
                build_rule_threshold_table,
                compare_with_ml_test,
                compare_with_ml_validation,
                describe_rule_set,
                fit_rule_artifacts,
                get_dataset_overview,
                load_rule_split,
                score_rule_frame,
                select_best_threshold,
            )

            pd.set_option("display.max_columns", 200)
            pd.set_option("display.width", 180)
            sns.set_theme(style="whitegrid")
            """
        ),
        md("# Carga de datos procesados"),
        code(
            """
            DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
            OUTPUTS_DIR = PROJECT_ROOT / "outputs"
            TABLES_DIR = OUTPUTS_DIR / "tables"
            FIGURES_DIR = OUTPUTS_DIR / "figures"

            MODELING_DATASET_PATH = DATA_PROCESSED_DIR / "transactions_modeling.parquet"
            SNAPSHOT_DATASET_PATH = DATA_PROCESSED_DIR / "user_card_snapshot_features.parquet"
            SPLIT_SUMMARY_PATH = TABLES_DIR / "split_summary.csv"
            MODEL_RESULTS_PATH = TABLES_DIR / "model_experiment_results.csv"
            FINAL_COMPARISON_PATH = TABLES_DIR / "final_model_comparison.csv"

            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)

            required_paths = {
                "Dataset de modelamiento": MODELING_DATASET_PATH,
                "Snapshot usuario-tarjeta": SNAPSHOT_DATASET_PATH,
                "Resumen de splits": SPLIT_SUMMARY_PATH,
                "Resultados de validacion ML": MODEL_RESULTS_PATH,
                "Comparacion final ML en test": FINAL_COMPARISON_PATH,
            }

            missing_paths = [name for name, path in required_paths.items() if not path.exists()]
            if missing_paths:
                raise FileNotFoundError(f"Faltan archivos requeridos para este notebook: {missing_paths}")

            generated_feature_files = pd.DataFrame(
                {
                    "artifact_name": [
                        "transactions_modeling.parquet",
                        "user_card_snapshot_features.parquet",
                    ],
                    "path": [
                        str(MODELING_DATASET_PATH),
                        str(SNAPSHOT_DATASET_PATH),
                    ],
                    "exists": [
                        MODELING_DATASET_PATH.exists(),
                        SNAPSHOT_DATASET_PATH.exists(),
                    ],
                    "size_mb": [
                        round(MODELING_DATASET_PATH.stat().st_size / 1024**2, 2),
                        round(SNAPSHOT_DATASET_PATH.stat().st_size / 1024**2, 2),
                    ],
                }
            )
            display(generated_feature_files)

            split_summary = pd.read_csv(SPLIT_SUMMARY_PATH)
            display(split_summary)

            display(
                Markdown(
                    f'''
                    **Particiones temporales reutilizadas**

                    - Train: `{DEFAULT_TEMPORAL_SPLIT.train_start}` a `{DEFAULT_TEMPORAL_SPLIT.train_end}`
                    - Validation: `{DEFAULT_TEMPORAL_SPLIT.validation_start}` a `{DEFAULT_TEMPORAL_SPLIT.validation_end}`
                    - Test: `{DEFAULT_TEMPORAL_SPLIT.test_start}` a `{DEFAULT_TEMPORAL_SPLIT.test_end}`
                    - Excluded: `{", ".join(DEFAULT_TEMPORAL_SPLIT.excluded_periods)}`
                    '''
                )
            )
            """
        ),
        md("# Revision de variables disponibles"),
        code(
            """
            dataset_overview = get_dataset_overview(MODELING_DATASET_PATH)

            available_rule_columns_df = pd.DataFrame({"safe_rule_columns": dataset_overview["safe_rule_columns"]})
            excluded_global_columns_df = pd.DataFrame({"excluded_global_columns": dataset_overview["excluded_global_columns"]})

            print(f"Numero total de columnas en transactions_modeling.parquet: {dataset_overview['n_columns']}")
            print(f"Numero de columnas seguras consideradas por el baseline: {len(dataset_overview['safe_rule_columns'])}")
            print(f"Numero de columnas excluidas por posible leakage temporal: {len(dataset_overview['excluded_global_columns'])}")

            display(available_rule_columns_df)
            display(excluded_global_columns_df)
            """
        ),
        md("# Definicion de reglas"),
        code(
            """
            artifacts = fit_rule_artifacts(MODELING_DATASET_PATH)
            validation_frame = load_rule_split(MODELING_DATASET_PATH, "validation")
            test_frame = load_rule_split(MODELING_DATASET_PATH, "test")

            rule_summary_validation = describe_rule_set(validation_frame, artifacts)
            risky_mccs_train = artifacts.mcc_risk_table.copy()

            risky_mccs_train.to_csv(TABLES_DIR / "rule_based_risky_mccs_train.csv", index=False)
            rule_summary_validation.to_csv(TABLES_DIR / "rule_based_rule_summary_validation.csv", index=False)

            display(
                Markdown(
                    f'''
                    **Criterio general de las reglas**

                    El baseline activa evidencia de fraude cuando la transaccion presenta desalineamientos frente al historial de la tarjeta o del usuario. En esta implementacion se consolidaron `{rule_summary_validation.attrs['n_rules']}` reglas.

                    **Variables globales excluidas**

                    Se excluyeron `{len(dataset_overview['excluded_global_columns'])}` variables de frecuencia global para evitar usar agregaciones calculadas con informacion futura respecto a `train`.
                    '''
                )
            )

            display(rule_summary_validation)
            display(risky_mccs_train[risky_mccs_train["selected_for_rule"] == 1].head(15))
            """
        ),
        md(
            """
            # Modelo basado en puntaje

            El puntaje del baseline se define como la proporcion de reglas activadas en cada transaccion:

            \\[
            \\text{rule\\_score}_i = \\frac{\\sum_{r=1}^{R} \\mathbb{1}(\\text{regla } r \\text{ activa en } i)}{R}
            \\]

            Este puntaje no reemplaza a la decision binaria final, pero resume de forma interpretable cuanta evidencia de riesgo acumula una transaccion. Posteriormente, ese puntaje se transforma en una prediccion binaria mediante un threshold optimizado en validacion.
            """
        ),
        code(
            """
            validation_rules, validation_scores = score_rule_frame(validation_frame, artifacts)
            test_rules, test_scores = score_rule_frame(test_frame, artifacts)

            validation_score_summary = (
                pd.DataFrame(
                    {
                        "rule_score": validation_scores,
                        "rules_triggered": validation_rules.sum(axis=1),
                        TARGET_COLUMN: validation_frame[TARGET_COLUMN].to_numpy(),
                    }
                )
                .groupby(TARGET_COLUMN)
                .agg(
                    n_obs=("rule_score", "size"),
                    mean_score=("rule_score", "mean"),
                    median_score=("rule_score", "median"),
                    mean_rules_triggered=("rules_triggered", "mean"),
                    p90_score=("rule_score", lambda s: float(s.quantile(0.90))),
                    max_score=("rule_score", "max"),
                )
                .reset_index()
            )

            display(validation_score_summary)
            """
        ),
        md(
            """
            # Modelo basado en reglas binarias

            La prediccion binaria se obtiene comparando el puntaje del baseline con un threshold fijo. Como el score corresponde a la proporcion de reglas activadas, thresholds mayores exigen mas evidencia simultanea antes de marcar una transaccion como fraude.
            """
        ),
        md("# Optimizacion de umbral en validacion"),
        code(
            """
            threshold_table_validation = build_rule_threshold_table(
                validation_frame[TARGET_COLUMN],
                validation_scores,
                n_rules=validation_rules.shape[1],
            )
            best_threshold_row = select_best_threshold(threshold_table_validation, criterion="f1")
            best_threshold = float(best_threshold_row["threshold"])
            min_rules_required = int(best_threshold_row["min_rules_required"])

            threshold_table_validation.to_csv(TABLES_DIR / "rule_based_threshold_optimization_validation.csv", index=False)
            threshold_table_validation.to_excel(TABLES_DIR / "rule_based_threshold_optimization_validation.xlsx", index=False)

            threshold_columns_order = [
                "threshold",
                "min_rules_required",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
                "tp",
                "fp",
                "tn",
                "fn",
                "n_predicted_positive",
            ]

            display(
                Markdown(
                    f'''
                    **Threshold seleccionado en validacion**

                    - Threshold: `{best_threshold:.2f}`
                    - Reglas minimas activadas: `{min_rules_required}`
                    - Precision: `{best_threshold_row['precision']:.6f}`
                    - Recall: `{best_threshold_row['recall']:.6f}`
                    - F1-score: `{best_threshold_row['f1']:.6f}`
                    - PR-AUC: `{best_threshold_row['pr_auc']:.6f}`
                    - ROC-AUC: `{best_threshold_row['roc_auc']:.6f}`
                    '''
                )
            )
            display(threshold_table_validation[threshold_columns_order].head(20))
            display(threshold_table_validation.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).head(10))
            """
        ),
        md("# Evaluacion final en test"),
        code(
            """
            final_metrics_test = build_rule_threshold_table(
                test_frame[TARGET_COLUMN],
                test_scores,
                n_rules=test_rules.shape[1],
                thresholds=np.array([best_threshold]),
            ).iloc[0]

            y_test = test_frame[TARGET_COLUMN].to_numpy(dtype=int)
            y_test_pred = (test_scores >= best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
            classification_report_test = pd.DataFrame(
                classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
            ).T

            final_metrics_test_row = {
                "model_name": "rule_based_binary",
                "feature_subset": "interpretable_rules",
                "balancing_strategy": "none",
                "n_rules": int(test_rules.shape[1]),
                "threshold_from_validation": float(best_threshold),
                "min_rules_required": int(final_metrics_test["min_rules_required"]),
                "accuracy": float(final_metrics_test["accuracy"]),
                "precision": float(final_metrics_test["precision"]),
                "recall": float(final_metrics_test["recall"]),
                "f1_score": float(final_metrics_test["f1"]),
                "roc_auc": float(final_metrics_test["roc_auc"]),
                "pr_auc": float(final_metrics_test["pr_auc"]),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "test_rows_used": int(final_metrics_test["n_obs"]),
                "test_positive_rows": int(final_metrics_test["n_positive"]),
            }

            final_metrics_test_df = pd.DataFrame([final_metrics_test_row])
            confusion_matrix_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual_0", "Actual_1"],
                columns=["Pred_0", "Pred_1"],
            )

            final_metrics_test_df.to_csv(TABLES_DIR / "rule_based_metrics_test.csv", index=False)
            final_metrics_test_df.to_excel(TABLES_DIR / "rule_based_metrics_test.xlsx", index=False)
            classification_report_test.to_csv(TABLES_DIR / "rule_based_classification_report_test.csv")
            confusion_matrix_df.to_csv(TABLES_DIR / "rule_based_confusion_matrix_test.csv")

            display(final_metrics_test_df)
            display(confusion_matrix_df)
            display(classification_report_test)
            """
        ),
        code(
            """
            # Matriz de confusion
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"Matriz de Confusion - Rule Based Model\\n(Threshold={best_threshold:.2f}, min reglas={min_rules_required})")
            ax.set_xlabel("Prediccion")
            ax.set_ylabel("Valor real")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "confusion_matrix_rule_based_model.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)

            # Curva Precision-Recall
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_scores)
            pr_curve_area = auc(recall_curve, precision_curve)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(recall_curve, precision_curve, color="darkorange", lw=2.5, label=f"PR curve (area={pr_curve_area:.4f})")
            ax.axhline(y=float(y_test.mean()), color="red", linestyle="--", lw=1.5, label=f"Baseline fraud rate={y_test.mean():.4f}")
            ax.set_title("Curva Precision-Recall - Rule Based Model")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(loc="best")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "precision_recall_curve_rule_based_model.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)

            # Curva ROC
            fpr, tpr, _ = roc_curve(y_test, test_scores)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC curve (AUC={final_metrics_test['roc_auc']:.4f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5)
            ax.set_title("Curva ROC - Rule Based Model")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "roc_curve_rule_based_model.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            """
        ),
        md("# Comparacion contra modelos de Machine Learning"),
        code(
            """
            validation_comparison = compare_with_ml_validation(MODEL_RESULTS_PATH, best_threshold_row.to_dict())
            test_comparison = compare_with_ml_test(FINAL_COMPARISON_PATH, final_metrics_test_row)

            validation_comparison.to_csv(TABLES_DIR / "rule_based_vs_ml_validation.csv", index=False)
            test_comparison.to_csv(TABLES_DIR / "rule_based_vs_ml_test.csv", index=False)

            rule_validation_rank = int(validation_comparison.index[validation_comparison["model_name"] == "rule_based_binary"][0]) + 1
            rule_test_rank = int(test_comparison.index[test_comparison["model_name"] == "rule_based_binary"][0]) + 1

            best_validation_ml = validation_comparison[validation_comparison["model_name"] != "rule_based_binary"].iloc[0]
            best_test_ml = test_comparison[test_comparison["model_name"] != "rule_based_binary"].iloc[0]

            display(
                Markdown(
                    f'''
                    **Alcance de la comparacion**

                    - `validation`: comparacion amplia contra las mejores corridas por familia reportadas en `model_experiment_results.csv`, incluyendo modelos como SVC, Random Forest, MLP, XGBoost y otros.
                    - `test`: comparacion final contra los modelos reentrenados y evaluados en `notebooks/4_model_selection_cards.ipynb`.
                    '''
                )
            )

            display(validation_comparison.head(15))
            display(test_comparison)
            """
        ),
        md("# Conclusiones del modelo basado en reglas"),
        code(
            """
            validation_pr_auc_gap = float(best_validation_ml["pr_auc"] - best_threshold_row["pr_auc"])
            test_pr_auc_gap = float(best_test_ml["pr_auc_test"] - final_metrics_test_row["pr_auc"])

            conclusion_markdown = f'''
            **Sintesis de resultados**

            El modelo basado en reglas produjo un **threshold optimo de {best_threshold:.2f} en validacion**, equivalente a exigir al menos **{min_rules_required} reglas activadas de {int(test_rules.shape[1])}**. Bajo esa configuracion, el baseline obtuvo en validacion **precision = {best_threshold_row["precision"]:.4f}**, **recall = {best_threshold_row["recall"]:.4f}**, **F1 = {best_threshold_row["f1"]:.4f}**, **PR-AUC = {best_threshold_row["pr_auc"]:.4f}** y **ROC-AUC = {best_threshold_row["roc_auc"]:.4f}**.

            En **test**, usando exactamente el threshold fijado en validacion y sin recalibracion adicional, el baseline alcanzo **accuracy = {final_metrics_test_row["accuracy"]:.4f}**, **precision = {final_metrics_test_row["precision"]:.4f}**, **recall = {final_metrics_test_row["recall"]:.4f}**, **F1 = {final_metrics_test_row["f1_score"]:.4f}**, **PR-AUC = {final_metrics_test_row["pr_auc"]:.4f}** y **ROC-AUC = {final_metrics_test_row["roc_auc"]:.4f}**. Esto implica que el sistema identifica aproximadamente **{final_metrics_test_row["recall"]*100:.2f}%** de los fraudes de test, pero con una precision de solo **{final_metrics_test_row["precision"]*100:.2f}%**, lo que evidencia una carga relevante de falsos positivos.

            **Interpretacion comparativa**

            En la comparacion amplia de **validacion**, el baseline basado en reglas ocupo la posicion **{rule_validation_rank} de {len(validation_comparison)}** al ordenar por PR-AUC. El mejor modelo de Machine Learning en ese corte fue **{best_validation_ml["model_name"]}** con **PR-AUC = {best_validation_ml["pr_auc"]:.4f}**, muy por encima del baseline (**brecha = {validation_pr_auc_gap:.4f}**). En la comparacion final de **test**, el baseline ocupo la posicion **{rule_test_rank} de {len(test_comparison)}**; el mejor modelo final fue **{best_test_ml["model_name"]}** con **PR-AUC = {best_test_ml["pr_auc_test"]:.4f}**, nuevamente superior al baseline por una brecha de **{test_pr_auc_gap:.4f}**.

            **Valor metodologico del baseline**

            A pesar de su menor desempeno relativo, el modelo basado en reglas conserva utilidad academica y operativa como linea base interpretable. Primero, demuestra que una parte de la senal de fraude puede capturarse con reglas transparentes apoyadas en monto, novedad de comercio, ubicacion, velocidad transaccional y riesgo por MCC. Segundo, ofrece un punto de referencia sencillo para justificar cuantitativamente el valor incremental de los modelos de Machine Learning.

            **Limitaciones**

            Los resultados observados muestran que el baseline no es competitivo frente a los mejores modelos supervisados del proyecto. Su **PR-AUC bajo**, su **recall moderado** y su **precision reducida** indican que las reglas explicitas no logran representar adecuadamente la complejidad de los patrones fraudulentos. Este comportamiento respalda la necesidad de emplear modelos de Machine Learning capaces de capturar interacciones no lineales, cambios sutiles en el comportamiento historico y combinaciones de variables que exceden la capacidad de un sistema de reglas fijo.
            '''

            display(Markdown(conclusion_markdown))

            generated_outputs = pd.DataFrame(
                {
                    "artifact": [
                        "outputs/tables/rule_based_threshold_optimization_validation.csv",
                        "outputs/tables/rule_based_metrics_test.csv",
                        "outputs/tables/rule_based_classification_report_test.csv",
                        "outputs/tables/rule_based_vs_ml_validation.csv",
                        "outputs/tables/rule_based_vs_ml_test.csv",
                        "outputs/figures/confusion_matrix_rule_based_model.png",
                        "outputs/figures/roc_curve_rule_based_model.png",
                        "outputs/figures/precision_recall_curve_rule_based_model.png",
                    ]
                }
            )
            display(generated_outputs)
            """
        ),
    ]

    notebook = new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    }
    return notebook


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "notebooks" / "3_1_rule_based_model_cards.ipynb"
    notebook = build_notebook()
    nbformat.write(notebook, output_path)
    print(f"Notebook generado en: {output_path}")


if __name__ == "__main__":
    main()
