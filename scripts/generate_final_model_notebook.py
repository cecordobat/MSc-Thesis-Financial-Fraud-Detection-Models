"""Generate notebook 5 for the final XGBoost thesis model."""

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
            # Modelo Final XGBoost para Tarjetas

            ## Título y objetivo

            Este notebook reconstruye y evalúa el **modelo final XGBoost** seleccionado previamente en `notebooks/4_model_selection_cards.ipynb`. Su propósito es documentar paso a paso la configuración final del modelo que se utilizará como referencia en la tesis, sin abrir una nueva fase de exploración ni comparación de algoritmos.

            **Alcance metodológico**

            - No se comparan nuevos modelos.
            - No se recalcula la selección de variables.
            - No se reajustan hiperparámetros con `test`.
            - No se optimiza nuevamente el threshold usando `test`.
            - Se utiliza exclusivamente la configuración ganadora ya fijada en notebook 4.
            """
        ),
        md(
            """
            El modelo final corresponde a una tarea de clasificación binaria a nivel transaccional, donde:

            - una fila representa una transacción
            - la variable objetivo es `is_fraud`
            - la métrica principal es **PR-AUC / average precision**

            El enfoque de este notebook es deliberadamente más estrecho que el de los notebooks 3 y 4: aquí solo se reconstruye el **XGBoost final seleccionado** para dejar trazabilidad empírica, outputs exportables y un texto académico reutilizable en el capítulo de resultados.
            """
        ),
        md("## Imports"),
        code(
            """
            from pathlib import Path
            import json
            import sys

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import xgboost as xgb
            from IPython.display import Markdown, display
            from sklearn.metrics import classification_report, confusion_matrix

            NOTEBOOK_DIR = Path.cwd().resolve()
            PROJECT_ROOT = next((path for path in [NOTEBOOK_DIR, *NOTEBOOK_DIR.parents] if (path / "src").exists()), NOTEBOOK_DIR)
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            pd.set_option("display.max_columns", 200)
            pd.set_option("display.width", 200)
            sns.set_theme(style="whitegrid")
            """
        ),
        md("## Conexión con src"),
        code(
            """
            from src.models.final_xgboost_model import (
                build_configuration_table,
                build_feature_importance_frame,
                build_final_metrics_table,
                build_final_model_artifact,
                build_split_summary,
                compute_train_validation_caps,
                get_dataset_schema,
                load_final_split,
                load_final_xgboost_inputs,
                load_train_validation_frame,
                make_default_final_xgboost_config,
                predict_final_model,
                train_final_xgboost_model,
                validate_required_files,
                validate_selected_features,
            )
            from src.utils.metrics import get_predicted_labels
            from src.utils.notebook_common import ensure_project_root_on_path, save_table_with_optional_excel
            from src.utils.plotting import (
                plot_confusion_matrix,
                plot_feature_importance,
                plot_precision_recall_curve,
                plot_probability_distribution,
                plot_roc_curve,
            )

            PROJECT_ROOT = ensure_project_root_on_path(PROJECT_ROOT)
            print(f"PROJECT_ROOT = {PROJECT_ROOT}")
            """
        ),
        md("## Definición de rutas"),
        code(
            """
            config = make_default_final_xgboost_config(PROJECT_ROOT)

            MODELING_DATASET_PATH = config.modeling_data_path
            SELECTED_FEATURES_PATH = config.selected_features_path
            FINAL_MODEL_COMPARISON_PATH = config.final_model_comparison_path
            THRESHOLD_OPTIMIZATION_PATH = config.threshold_optimization_path
            BEST_MODEL_METADATA_PATH = config.best_model_metadata_path
            BEST_MODEL_FINAL_ARTIFACT_PATH = config.best_model_final_artifact_path
            SOURCE_MODEL_ARTIFACT_PATH = config.source_model_artifact_path
            FINAL_METRICS_SUMMARY_PATH = config.final_metrics_summary_path
            TABLES_DIR = config.tables_dir
            FIGURES_DIR = config.figures_dir
            MODELS_DIR = config.models_dir

            required_files = validate_required_files(config)
            display(required_files)
            """
        ),
        md("## Carga de datos"),
        code(
            """
            dataset_columns = get_dataset_schema(MODELING_DATASET_PATH)
            split_summary = build_split_summary(config)

            dataset_overview = pd.DataFrame(
                [
                    {
                        "dataset_path": str(MODELING_DATASET_PATH),
                        "n_rows_total": int(split_summary["n_rows"].sum()),
                        "n_columns_total": len(dataset_columns),
                        "target_column_present": config.target_col in dataset_columns,
                        "year_month_present": "year_month" in dataset_columns,
                        "datetime_present": "datetime" in dataset_columns,
                    }
                ]
            )

            target_distribution = pd.DataFrame(
                [
                    {
                        "scope": "full_dataset_including_excluded_periods",
                        "n_rows": int(split_summary["n_rows"].sum()),
                        "n_frauds": int(split_summary["n_positive"].sum()),
                        "fraud_rate": float(split_summary["n_positive"].sum() / split_summary["n_rows"].sum()),
                    },
                    {
                        "scope": "train_validation_test_only",
                        "n_rows": int(split_summary.loc[split_summary["split"].isin(["train", "validation", "test"]), "n_rows"].sum()),
                        "n_frauds": int(split_summary.loc[split_summary["split"].isin(["train", "validation", "test"]), "n_positive"].sum()),
                        "fraud_rate": float(
                            split_summary.loc[split_summary["split"].isin(["train", "validation", "test"]), "n_positive"].sum()
                            / split_summary.loc[split_summary["split"].isin(["train", "validation", "test"]), "n_rows"].sum()
                        ),
                    },
                ]
            )

            display(dataset_overview)
            display(target_distribution)
            display(pd.DataFrame({"dataset_columns_preview": dataset_columns[:50]}))
            """
        ),
        md("## Carga de configuración ganadora"),
        code(
            """
            inputs = load_final_xgboost_inputs(config)
            metadata = inputs["metadata"]
            hyperparameters = inputs["hyperparameters"]

            winner_summary = pd.DataFrame(
                [
                    {
                        "model_name": metadata["model_name"],
                        "feature_subset": metadata["feature_subset"],
                        "balancing_strategy": metadata["balancing_strategy"],
                        "threshold_final": inputs["threshold_final"],
                        "threshold_from_source_artifact": inputs["threshold_from_source_artifact"],
                        "random_state": inputs["random_state"],
                        "train_validation_row_cap": inputs["train_validation_row_cap"],
                        "training_data_rule": metadata["training_data"],
                        "xgboost_version_current_env": xgb.__version__,
                    }
                ]
            )

            hyperparameters_df = pd.DataFrame(
                [{"hyperparameter": key, "value": value} for key, value in hyperparameters.items()]
            )

            display(
                Markdown(
                    f'''
                    **Fuente de verdad metodológica**

                    - Artefacto final exacto de notebook 4: `{BEST_MODEL_FINAL_ARTIFACT_PATH.name}`
                    - Metadata final: `{BEST_MODEL_METADATA_PATH.name}`
                    - Threshold final de validación: `{inputs["threshold_final"]:.2f}`
                    - Threshold heredado del artefacto base del experimento: `{inputs["threshold_from_source_artifact"]:.2f}`

                    En este notebook se usa como referencia oficial el artefacto final ya materializado en notebook 4, porque es el objeto exacto con el que se reportaron las métricas finales del modelo seleccionado.
                    '''
                )
            )

            display(winner_summary)
            display(hyperparameters_df)
            """
        ),
        md("## Validación de variables seleccionadas"),
        code(
            """
            feature_validation = validate_selected_features(
                dataset_columns=dataset_columns,
                selected_features=inputs["selected_features"],
                target_col=config.target_col,
            )

            selected_features_final = inputs["selected_features_table"].copy()
            selected_features_final = selected_features_final[selected_features_final["selected_top_100"] == True].copy()
            selected_features_final["feature_order_final"] = selected_features_final["feature"].map(
                {feature: idx + 1 for idx, feature in enumerate(inputs["selected_features"])}
            )
            selected_features_final = selected_features_final.sort_values("feature_order_final").reset_index(drop=True)

            feature_validation_df = pd.DataFrame(
                [
                    {"validation_check": "n_selected_features", "value": feature_validation["n_selected_features"]},
                    {"validation_check": "missing_features", "value": feature_validation["missing_features"]},
                    {"validation_check": "duplicated_features", "value": feature_validation["duplicated_features"]},
                    {"validation_check": "identifier_hits", "value": feature_validation["identifier_hits"]},
                    {"validation_check": "direct_target_hits", "value": feature_validation["direct_target_hits"]},
                    {"validation_check": "is_valid", "value": feature_validation["is_valid"]},
                ]
            )

            if not feature_validation["is_valid"]:
                raise ValueError(f"Las variables seleccionadas no son válidas: {feature_validation}")

            display(feature_validation_df)
            display(selected_features_final.head(20))
            """
        ),
        md("## Reproducción del split temporal"),
        code(
            """
            caps = compute_train_validation_caps(split_summary, max_rows=inputs["train_validation_row_cap"])

            effective_fit_summary = pd.DataFrame(
                [
                    {
                        "split_for_fit": "train_effective",
                        "n_rows_full_split": caps["train_rows_full"],
                        "rows_used_in_final_fit": caps["train_cap"],
                    },
                    {
                        "split_for_fit": "validation_effective",
                        "n_rows_full_split": caps["validation_rows_full"],
                        "rows_used_in_final_fit": caps["validation_cap"],
                    },
                    {
                        "split_for_fit": "train_plus_validation_effective",
                        "n_rows_full_split": caps["train_validation_rows_full"],
                        "rows_used_in_final_fit": inputs["train_validation_row_cap"],
                    },
                    {
                        "split_for_fit": "test_full",
                        "n_rows_full_split": int(split_summary.loc[split_summary["split"] == "test", "n_rows"].iloc[0]),
                        "rows_used_in_final_fit": int(split_summary.loc[split_summary["split"] == "test", "n_rows"].iloc[0]),
                    },
                ]
            )

            display(split_summary)
            display(effective_fit_summary)
            """
        ),
        md("## Preparación de X e y"),
        code(
            """
            train_frame = load_final_split(
                config,
                "train",
                inputs["selected_features"],
                max_rows=caps["train_cap"],
                sort_by=None,
            )
            validation_frame = load_final_split(
                config,
                "validation",
                inputs["selected_features"],
                max_rows=caps["validation_cap"],
                sort_by=None,
            )
            test_frame = load_final_split(
                config,
                "test",
                inputs["selected_features"],
                max_rows=None,
                sort_by=None,
            )

            X_train = train_frame.loc[:, inputs["selected_features"]].copy()
            y_train = train_frame.loc[:, config.target_col].copy()
            X_val = validation_frame.loc[:, inputs["selected_features"]].copy()
            y_val = validation_frame.loc[:, config.target_col].copy()
            X_test = test_frame.loc[:, inputs["selected_features"]].copy()
            y_test = test_frame.loc[:, config.target_col].copy()

            frame_shapes = pd.DataFrame(
                [
                    {"split": "train_effective", "n_rows": len(X_train), "n_features": X_train.shape[1], "n_frauds": int(y_train.sum()), "fraud_rate": float(y_train.mean())},
                    {"split": "validation_effective", "n_rows": len(X_val), "n_features": X_val.shape[1], "n_frauds": int(y_val.sum()), "fraud_rate": float(y_val.mean())},
                    {"split": "test_full", "n_rows": len(X_test), "n_features": X_test.shape[1], "n_frauds": int(y_test.sum()), "fraud_rate": float(y_test.mean())},
                ]
            )

            display(frame_shapes)
            """
        ),
        md("## Construcción del modelo final XGBoost"),
        code(
            """
            source_model_artifact = inputs["source_artifact"]
            final_model_artifact = joblib.load(BEST_MODEL_FINAL_ARTIFACT_PATH)
            final_model = final_model_artifact["model"]
            threshold_final = float(final_model_artifact["threshold_final"])

            configuration_reference_df = pd.DataFrame(
                [
                    {
                        "official_final_artifact": BEST_MODEL_FINAL_ARTIFACT_PATH.name,
                        "source_experiment_artifact": SOURCE_MODEL_ARTIFACT_PATH.name,
                        "model_name": metadata["model_name"],
                        "feature_subset": metadata["feature_subset"],
                        "balancing_strategy": metadata["balancing_strategy"],
                        "random_state": inputs["random_state"],
                        "threshold_final": threshold_final,
                        "number_of_features_used": len(inputs["selected_features"]),
                    }
                ]
            )

            display(
                Markdown(
                    '''
                    **Decisión de reconstrucción**

                    Notebook 4 ya reentrenó el modelo final en `train + validation` y guardó el artefacto resultante en `best_model_final.joblib`. Para preservar exactamente las métricas finales ya reportadas en la tesis, este notebook usa ese artefacto final como objeto oficial de evaluación.

                    Adicionalmente, en la siguiente sección se ejecuta un reentrenamiento de referencia con la misma configuración únicamente para medir tiempo de entrenamiento y documentar el `scale_pos_weight` efectivo aplicado al conjunto final de ajuste. Ese rerun no reemplaza el artefacto final oficial.
                    '''
                )
            )

            display(configuration_reference_df)
            """
        ),
        md("## Entrenamiento final"),
        code(
            """
            train_validation_fit_frame = pd.concat([train_frame, validation_frame], axis=0, ignore_index=True)

            timing_rerun = train_final_xgboost_model(
                source_artifact=source_model_artifact,
                train_frame=train_validation_fit_frame,
                feature_columns=inputs["selected_features"],
                target_col=config.target_col,
            )

            timing_rerun_df = pd.DataFrame(
                [
                    {
                        "train_rows_used": timing_rerun["train_rows_used"],
                        "train_positive_rows": timing_rerun["train_positive_rows"],
                        "effective_scale_pos_weight": timing_rerun["effective_scale_pos_weight"],
                        "train_time_seconds_reference_rerun": timing_rerun["train_time_seconds"],
                    }
                ]
            )

            display(timing_rerun_df)
            """
        ),
        md("## Predicción sobre test"),
        code(
            """
            prediction_bundle = predict_final_model(
                model=final_model,
                frame=test_frame,
                feature_columns=inputs["selected_features"],
            )

            y_test_scores = prediction_bundle["scores"]
            predict_time_seconds = prediction_bundle["predict_time_seconds"]
            y_test_pred = get_predicted_labels(y_test_scores, threshold=threshold_final)

            prediction_summary = pd.DataFrame(
                [
                    {
                        "threshold_used": threshold_final,
                        "predict_time_seconds": predict_time_seconds,
                        "n_test_rows": len(y_test),
                        "n_predicted_positive": int(y_test_pred.sum()),
                    }
                ]
            )

            display(prediction_summary)
            """
        ),
        md("## Evaluación final"),
        code(
            """
            final_metrics_df = build_final_metrics_table(
                y_true=y_test,
                scores=y_test_scores,
                threshold=threshold_final,
                model_name=metadata["model_name"],
                feature_subset=metadata["feature_subset"],
                balancing_strategy=metadata["balancing_strategy"],
                number_of_features_used=len(inputs["selected_features"]),
                train_time_seconds=timing_rerun["train_time_seconds"],
                predict_time_seconds=predict_time_seconds,
            )

            confusion_values = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
            confusion_matrix_df = pd.DataFrame(
                [
                    {"actual_label": "No Fraud", "pred_no_fraud": int(confusion_values[0, 0]), "pred_fraud": int(confusion_values[0, 1])},
                    {"actual_label": "Fraud", "pred_no_fraud": int(confusion_values[1, 0]), "pred_fraud": int(confusion_values[1, 1])},
                ]
            )

            classification_report_df = (
                pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True, zero_division=0))
                .T.reset_index().rename(columns={"index": "class_or_average"})
            )

            final_metrics_row = final_metrics_df.iloc[0].to_dict()

            summary_reference = inputs["final_metrics_summary"].copy()
            metric_col = summary_reference.columns[0]
            value_col = summary_reference.columns[1]
            summary_reference[value_col] = summary_reference[value_col].astype(float)
            metric_mapping = {
                "PR-AUC": final_metrics_row["pr_auc"],
                "ROC-AUC": final_metrics_row["roc_auc"],
                "Precision": final_metrics_row["precision"],
                "Recall": final_metrics_row["recall"],
                "F1-Score": final_metrics_row["f1_score"],
                "Accuracy": final_metrics_row["accuracy"],
            }
            consistency_check = summary_reference[[metric_col, value_col]].rename(columns={metric_col: "metric_name", value_col: "notebook4_value"})
            consistency_check["notebook5_value"] = consistency_check["metric_name"].map(metric_mapping)
            consistency_check["absolute_difference"] = (consistency_check["notebook4_value"] - consistency_check["notebook5_value"]).abs()

            if not (consistency_check["absolute_difference"] < 1e-6).all():
                raise AssertionError("Las métricas de notebook 5 no reproducen exactamente el artefacto final de notebook 4.")

            display(final_metrics_df)
            display(confusion_matrix_df)
            display(classification_report_df)
            display(consistency_check)
            """
        ),
        md("## Figuras finales"),
        code(
            """
            plot_confusion_matrix(
                confusion_values=confusion_values,
                labels=("No Fraud", "Fraud"),
                title=f"Matriz de Confusión - {metadata['model_name']}_{metadata['feature_subset']}\\n(Threshold={threshold_final:.2f})",
                output_path=FIGURES_DIR / "confusion_matrix_best_model.png",
            )
            plot_roc_curve(
                y_true=y_test,
                scores=y_test_scores,
                model_name=f"{metadata['model_name']}_{metadata['feature_subset']}",
                output_path=FIGURES_DIR / "roc_curve_best_model.png",
            )
            plot_precision_recall_curve(
                y_true=y_test,
                scores=y_test_scores,
                model_name=f"{metadata['model_name']}_{metadata['feature_subset']}",
                output_path=FIGURES_DIR / "precision_recall_curve_best_model.png",
            )
            plot_probability_distribution(
                scores=y_test_scores,
                y_true=y_test,
                threshold=threshold_final,
                title="Distribución de probabilidades - Modelo final XGBoost",
                output_path=FIGURES_DIR / "final_model_probability_distribution.png",
            )

            feature_importance_df = build_feature_importance_frame(final_model, inputs["selected_features"])
            plot_feature_importance(
                importance_frame=feature_importance_df,
                top_n=30,
                title="Importancia de variables - Top 30 del modelo final XGBoost",
                output_path=FIGURES_DIR / "final_model_feature_importance_top_30.png",
            )

            display(feature_importance_df.head(30))
            """
        ),
        md("## Tablas finales"),
        code(
            """
            configuration_df = build_configuration_table(
                config=config,
                inputs=inputs,
                effective_scale_pos_weight=timing_rerun["effective_scale_pos_weight"],
            )

            save_table_with_optional_excel(final_metrics_df, TABLES_DIR / "final_model_test_metrics.csv", TABLES_DIR / "final_model_test_metrics.xlsx")
            save_table_with_optional_excel(confusion_matrix_df, TABLES_DIR / "final_model_confusion_matrix.csv", TABLES_DIR / "final_model_confusion_matrix.xlsx")
            save_table_with_optional_excel(classification_report_df, TABLES_DIR / "final_model_classification_report.csv", TABLES_DIR / "final_model_classification_report.xlsx")
            save_table_with_optional_excel(selected_features_final, TABLES_DIR / "final_model_selected_features.csv", TABLES_DIR / "final_model_selected_features.xlsx")
            save_table_with_optional_excel(configuration_df, TABLES_DIR / "final_model_configuration.csv", TABLES_DIR / "final_model_configuration.xlsx")

            generated_tables = pd.DataFrame(
                {
                    "table_path": [
                        "outputs/tables/final_model_test_metrics.csv",
                        "outputs/tables/final_model_confusion_matrix.csv",
                        "outputs/tables/final_model_classification_report.csv",
                        "outputs/tables/final_model_selected_features.csv",
                        "outputs/tables/final_model_configuration.csv",
                    ]
                }
            )
            display(generated_tables)
            """
        ),
        md("## Guardado del modelo final"),
        code(
            """
            final_model_output_path = MODELS_DIR / "xgboost_top_100_scale_pos_weight_manual_final.joblib"

            final_artifact_payload = build_final_model_artifact(
                model=final_model,
                feature_columns=inputs["selected_features"],
                threshold_final=threshold_final,
                hyperparameters=hyperparameters,
                random_state=inputs["random_state"],
                final_metrics_row=final_metrics_row,
                configuration_row=configuration_df.iloc[0].to_dict(),
            )
            final_artifact_payload["data_paths_used"] = {
                "modeling_dataset_path": str(MODELING_DATASET_PATH),
                "selected_features_path": str(SELECTED_FEATURES_PATH),
                "source_model_artifact_path": str(SOURCE_MODEL_ARTIFACT_PATH),
                "best_model_final_artifact_path": str(BEST_MODEL_FINAL_ARTIFACT_PATH),
            }
            final_artifact_payload["timing_reference_rerun"] = timing_rerun_df.iloc[0].to_dict()

            joblib.dump(final_artifact_payload, final_model_output_path)

            display(pd.DataFrame([{"saved_model_path": str(final_model_output_path), "exists": final_model_output_path.exists()}]))
            """
        ),
        md("## Resumen para tesis"),
        code(
            """
            thesis_summary_markdown = f'''
            **Resumen académico del modelo final**

            El modelo final seleccionado para la detección de fraude transaccional con tarjetas corresponde a un **XGBoost** entrenado sobre el subconjunto de variables **top_100**, con estrategia de balanceo **scale_pos_weight_manual** y umbral final de decisión **{threshold_final:.2f}**, definido exclusivamente con el conjunto de **validation** en notebook 4.

            En la evaluación final sobre **test**, el modelo obtuvo las siguientes métricas: **accuracy = {final_metrics_row["accuracy"]:.4f}**, **precision = {final_metrics_row["precision"]:.4f}**, **recall = {final_metrics_row["recall"]:.4f}**, **F1-score = {final_metrics_row["f1_score"]:.4f}**, **PR-AUC = {final_metrics_row["pr_auc"]:.4f}** y **ROC-AUC = {final_metrics_row["roc_auc"]:.4f}**.

            Desde una perspectiva interpretativa, la **precision** indica la proporción de alertas de fraude que efectivamente corresponden a fraude real, mientras que el **recall** muestra la capacidad del modelo para recuperar fraudes verdaderos. El **F1-score** resume el equilibrio entre precision y recall, la **PR-AUC** es la métrica principal por tratarse de un problema altamente desbalanceado, y la **ROC-AUC** describe la capacidad global de discriminación del modelo.

            Metodológicamente, es fundamental subrayar que el conjunto **test** se utilizó exclusivamente para la **evaluación final** del modelo ya seleccionado; no participó en selección de variables, ajuste de hiperparámetros ni optimización del threshold.
            '''

            display(Markdown(thesis_summary_markdown))
            """
        ),
        md("## Conclusiones del notebook"),
        code(
            """
            conclusions_markdown = '''
            **Conclusiones**

            Este notebook cierra la evaluación empírica del modelo final de la tesis. Las decisiones de modelo, subconjunto de variables, estrategia de balanceo, reentrenamiento metodológico y threshold final provienen del notebook 4 y se respetan aquí sin introducir una nueva etapa de selección.

            El objetivo principal de este notebook es dejar una reconstrucción clara del modelo final XGBoost, junto con sus métricas, tablas, figuras y artefactos exportables para el capítulo 5 de la tesis. Los archivos generados en `outputs/tables`, `outputs/figures` y `outputs/models` deben utilizarse como insumo directo para la redacción de resultados y conclusiones.
            '''

            generated_outputs = pd.DataFrame(
                {
                    "artifact": [
                        "notebooks/5_model_final_cards.ipynb",
                        "outputs/tables/final_model_test_metrics.csv",
                        "outputs/tables/final_model_confusion_matrix.csv",
                        "outputs/tables/final_model_classification_report.csv",
                        "outputs/tables/final_model_selected_features.csv",
                        "outputs/tables/final_model_configuration.csv",
                        "outputs/figures/confusion_matrix_best_model.png",
                        "outputs/figures/roc_curve_best_model.png",
                        "outputs/figures/precision_recall_curve_best_model.png",
                        "outputs/figures/final_model_probability_distribution.png",
                        "outputs/figures/final_model_feature_importance_top_30.png",
                        "outputs/models/xgboost_top_100_scale_pos_weight_manual_final.joblib",
                    ]
                }
            )

            display(Markdown(conclusions_markdown))
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
    output_path = project_root / "notebooks" / "5_model_final_cards.ipynb"
    nbformat.write(build_notebook(), output_path)
    print(f"Notebook generado en: {output_path}")


if __name__ == "__main__":
    main()
