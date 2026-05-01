"""
Funciones para la parte 1
"""

import pandas as pd
import os
from typing import Optional


def convertir_parquet(ruta_csv: str, ruta_parquet: str, chunksize: Optional[int] = None) -> None:
    """
    Convierte un archivo CSV a formato Parquet para optimizar el procesamiento de grandes datasets.

    Args:
        ruta_csv (str): Ruta completa al archivo CSV de entrada
        ruta_parquet (str): Ruta completa donde se guardará el archivo Parquet
        chunksize (Optional[int]): Tamaño de chunks para procesamiento en memoria limitada.
                                  Si None, carga todo el archivo de una vez.

    Raises:
        FileNotFoundError: Si el archivo CSV no existe
        Exception: Para otros errores durante la conversión

    Example:
        convertir_parquet('data/original/transacciones.csv', 'data/processed/transacciones.parquet')
    """
    try:
        # Verificar que el archivo CSV existe
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"El archivo CSV no existe: {ruta_csv}")

        print(f"Iniciando conversión de {ruta_csv} a {ruta_parquet}")

        if chunksize:
            # Procesamiento por chunks para archivos muy grandes
            print(f"Procesando en chunks de {chunksize} filas")

            # Leer el archivo en chunks y concatenar
            chunks = []
            for chunk in pd.read_csv(ruta_csv, chunksize=chunksize, low_memory=False):
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)
        else:
            # Carga completa del archivo
            print("Cargando archivo completo...")
            df = pd.read_csv(ruta_csv, low_memory=False)

        # Crear directorio de salida si no existe
        os.makedirs(os.path.dirname(ruta_parquet), exist_ok=True)

        # Convertir a Parquet
        df.to_parquet(ruta_parquet, index=False, engine='pyarrow')

        print(f"Conversión exitosa: {len(df):,} filas guardadas en {ruta_parquet}")
        print(f"Tamaño original: {os.path.getsize(ruta_csv) / (1024**3):.2f} GB")
        print(f"Tamaño Parquet: {os.path.getsize(ruta_parquet) / (1024**3):.2f} GB")

    except Exception as e:
        print(f"❌ Error durante la conversión: {str(e)}")
        raise


def leer_parquet_si_existe(ruta_parquet: str) -> Optional[pd.DataFrame]:
    """
    Lee un archivo Parquet si existe, de lo contrario retorna None y muestra un mensaje.

    Args:
        ruta_parquet (str): Ruta completa al archivo Parquet

    Returns:
        pd.DataFrame: DataFrame si el archivo existe, None en caso contrario

    Example:
        df = leer_parquet_si_existe('data/processed/transacciones.parquet')
        if df is not None:
            print(f"Archivo cargado: {df.shape[0]} filas")
    """
    if os.path.exists(ruta_parquet):
        try:
            df = pd.read_parquet(ruta_parquet)
            print(f"Archivo cargado: {ruta_parquet}")
            print(f" Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            return df
        except Exception as e:
            print(f"Error al leer el archivo: {str(e)}")
            return None
    else:
        print(f"El archivo no existe: {ruta_parquet}")
        return None


if __name__ == "__main__":
    # Ejemplo de uso para el archivo de transacciones
    ruta_csv = "data/original/HI-Large_Trans.csv"
    ruta_parquet = "data/processed/HI-Large_Trans.parquet"

    # Para archivos muy grandes, usar chunksize (ej: 1 millón de filas por chunk)
    convertir_parquet(ruta_csv, ruta_parquet, chunksize=1_000_000)