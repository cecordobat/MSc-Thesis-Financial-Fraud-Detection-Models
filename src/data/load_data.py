from pathlib import Path

import pandas as pd


def read_transactions_parquet(path: Path) -> pd.DataFrame:
    """
    Carga el archivo Parquet de transacciones usando pandas.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    return pd.read_parquet(path)


def read_sample_from_parquet(path: Path, n_rows: int = 10) -> pd.DataFrame:
    """
    Lee una muestra pequeña del archivo Parquet.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    return pd.read_parquet(path).head(n_rows)
