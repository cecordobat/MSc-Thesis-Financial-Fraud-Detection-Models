"""Common helpers used by the notebook v2 pipelines."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a notebook or script location."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("No se pudo ubicar la raiz del proyecto.")


def ensure_project_root_on_path(start: Path | None = None) -> Path:
    """Resolve the project root and add it to ``sys.path`` if required."""

    project_root = find_project_root(start=start)
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return project_root


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int | None = None) -> int | None:
    """Read an integer from the environment."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_int_list(name: str, default: Sequence[int] | None = None) -> list[int]:
    """Read a comma-separated list of integers from the environment."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return list(default or [])
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def append_suffix(path: str | Path, suffix: str = "") -> Path:
    """Append a suffix before the file extension."""

    target = Path(path)
    if not suffix:
        return target
    return target.with_name(f"{target.stem}{suffix}{target.suffix}")


def save_table_with_optional_excel(
    frame: pd.DataFrame,
    csv_path: str | Path,
    excel_path: str | Path | None = None,
) -> None:
    """Save a dataframe to CSV and, when possible, to Excel."""

    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_target, index=False)

    if excel_path is None:
        return

    excel_target = Path(excel_path)
    excel_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(excel_target, index=False)


def read_parquet_row_groups(
    path: str | Path,
    row_groups: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read one or more parquet row groups into pandas."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = Path(path)
    parquet_file = pq.ParquetFile(parquet_path)
    selected_groups = list(row_groups or [0])
    tables = [parquet_file.read_row_group(group, columns=list(columns) if columns else None) for group in selected_groups]
    if not tables:
        return pd.DataFrame(columns=list(columns or []))
    return pa.concat_tables(tables).to_pandas()


def read_parquet_head(
    path: str | Path,
    n_rows: int = 10,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read only the first ``n_rows`` rows from a parquet file."""

    import pyarrow.parquet as pq

    parquet_path = Path(path)
    parquet_file = pq.ParquetFile(parquet_path)
    batch_iter = parquet_file.iter_batches(
        batch_size=n_rows,
        columns=list(columns) if columns else None,
    )
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        return pd.DataFrame(columns=list(columns or parquet_file.schema_arrow.names))
    return first_batch.to_pandas()
