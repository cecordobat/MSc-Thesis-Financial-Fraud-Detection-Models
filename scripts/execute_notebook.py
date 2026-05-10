"""Execute a notebook in-place with nbclient."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python scripts/execute_notebook.py <notebook_path>")

    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    notebook_path = Path(sys.argv[1]).resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"No existe el notebook: {notebook_path}")

    repo_root = notebook_path.parents[1]
    os.environ.setdefault("JUPYTER_CONFIG_DIR", str(repo_root / ".jupyter_config"))
    os.environ.setdefault("JUPYTER_DATA_DIR", str(repo_root / ".jupyter_data"))
    os.environ.setdefault("IPYTHONDIR", str(repo_root / ".ipython"))

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Ejecutado y guardado: {notebook_path}")


if __name__ == "__main__":
    main()
