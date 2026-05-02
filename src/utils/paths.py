from pathlib import Path


def get_project_root() -> Path:
    """
    Retorna la raíz del proyecto.
    Asume que este archivo vive en src/utils/.
    """
    return Path(__file__).resolve().parents[2]


def get_data_original_dir() -> Path:
    return get_project_root() / "data" / "original"


def get_data_processed_dir() -> Path:
    return get_project_root() / "data" / "processed"


def get_artifacts_dir() -> Path:
    return get_project_root() / "artifacts"


def get_figures_dir() -> Path:
    return get_artifacts_dir() / "figures"


def get_metrics_dir() -> Path:
    return get_artifacts_dir() / "metrics"


def get_models_dir() -> Path:
    return get_artifacts_dir() / "models"


def get_transactions_parquet_path() -> Path:
    return get_data_processed_dir() / "transactions_clean.parquet"
