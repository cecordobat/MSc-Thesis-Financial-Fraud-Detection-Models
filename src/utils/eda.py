import pandas as pd


def get_dataset_shape(df: pd.DataFrame) -> dict:
    """
    Retorna número de filas y columnas de un DataFrame.
    """
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
    }


def get_null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna resumen de valores nulos por columna.
    """
    summary = df.isnull().sum().reset_index()
    summary.columns = ["column", "null_count"]
    summary["null_percentage"] = (summary["null_count"] / len(df) * 100).round(2)
    summary = summary.sort_values("null_count", ascending=False).reset_index(drop=True)
    return summary


def get_target_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Retorna conteo y proporción de la variable objetivo.
    """
    distribution = df[target_col].value_counts(dropna=False).rename_axis(target_col).reset_index(name="count")
    distribution["proportion"] = (distribution["count"] / distribution["count"].sum()).round(4)
    return distribution.sort_values(target_col).reset_index(drop=True)


def get_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """
    Retorna resumen estadístico para columnas numéricas.
    """
    summary = {}
    for col in numeric_columns:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_median"] = df[col].median()
        summary[f"{col}_std"] = df[col].std()
        summary[f"{col}_min"] = df[col].min()
        summary[f"{col}_max"] = df[col].max()
    return pd.DataFrame([summary])


def get_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    target_col: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Retorna distribución de una variable categórica.

    Si se pasa target_col, también calcula tasa de fraude por categoría.
    """
    if target_col is None:
        distribution = df[column].value_counts(dropna=False).reset_index()
        distribution.columns = [column, "count"]
        return distribution.head(top_n).reset_index(drop=True)

    distribution = (
        df.groupby(column)
        .agg(
            count=(target_col, "size"),
            fraud_rate=(target_col, "mean"),
            fraud_count=(target_col, "sum"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return distribution






def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los nombres de columnas.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("?", "", regex=False)
    )

    return df




def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte los tipos de datos del dataset IBM Credit Card Transactions.

    Reglas aplicadas:
    - user y card: se convierten primero a entero para eliminar posibles ".0"
      y luego a string, porque son identificadores.
    - year, month, day: enteros.
    - time: string.
    - amount: float64, eliminando "$" y ",".
    - use_chip: string.
    - merchant_name: string, porque es un identificador anonimizado del comercio.
    - merchant_city, merchant_state: string.
    - zip: string, eliminando posibles ".0".
    - mcc: string, porque es un código categórico.
    - errors: string, conservando nulos.
    - is_fraud: Int8, convirtiendo "No" a 0 y "Yes" a 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original con nombres de columnas ya limpiados.

    Returns
    -------
    pd.DataFrame
        DataFrame con tipos de datos convertidos.
    """

    df = df.copy()

    # Identificadores: primero numérico entero para quitar ".0", luego string
    for col in ["user", "card"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .astype("Int64")
                .astype("string")
            )

    # Variables temporales enteras
    integer_cols = ["year", "month", "day"]

    for col in integer_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

    # Tiempo
    if "time" in df.columns:
        df["time"] = df["time"].astype("string").str.strip()

    # Monto
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .astype("string")
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype("float64")

    # Categóricas textuales
    categorical_cols = [
        "use_chip",
        "merchant_name",
        "merchant_city",
        "merchant_state",
        "errors",
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # ZIP como categórica geográfica
    if "zip" in df.columns:
        df["zip"] = (
            df["zip"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    # MCC como código categórico
    if "mcc" in df.columns:
        df["mcc"] = (
            pd.to_numeric(df["mcc"], errors="coerce")
            .astype("Int64")
            .astype("string")
        )

    # Variable objetivo binaria
    if "is_fraud" in df.columns:
        fraud_clean = (
            df["is_fraud"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        unexpected_values = sorted(
            set(fraud_clean.dropna().unique()) - {"yes", "no"}
        )

        if unexpected_values:
            raise ValueError(
                f"Valores inesperados en is_fraud: {unexpected_values}"
            )

        df["is_fraud"] = (
            fraud_clean
            .map({"no": 0, "yes": 1})
            .astype("Int8")
        )

    return df

def create_mode_mapping(
    df: pd.DataFrame,
    group_cols: str | list[str],
    target_col: str
) -> dict:
    """
    Crea un diccionario de imputación usando la moda del target_col
    para cada grupo definido en group_cols.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.
    group_cols : str or list[str]
        Columna o columnas usadas para agrupar.
    target_col : str
        Columna que se quiere imputar.

    Returns
    -------
    dict
        Diccionario con la moda de target_col por grupo.
    """

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    mapping = (
        df
        .dropna(subset=group_cols + [target_col])
        .groupby(group_cols)[target_col]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    return mapping

def impute_merchant_state_by_city(
    df: pd.DataFrame,
    city_col: str = "merchant_city",
    state_col: str = "merchant_state",
    output_col: str = "merchant_state_imputed"
) -> pd.DataFrame:
    """
    Imputa merchant_state usando la moda de merchant_state por merchant_city.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.
    city_col : str
        Columna con la ciudad del comercio.
    state_col : str
        Columna con el estado del comercio.
    output_col : str
        Nombre de la nueva columna imputada.

    Returns
    -------
    pd.DataFrame
        DataFrame con la columna output_col añadida.
    """

    df = df.copy()

    city_to_state = create_mode_mapping(
        df=df,
        group_cols=city_col,
        target_col=state_col
    )

    df[output_col] = df[state_col]

    mask_missing = df[output_col].isna()

    df.loc[mask_missing, output_col] = (
        df.loc[mask_missing, city_col].map(city_to_state)
    )

    return df

def impute_zip_by_city_state(
    df: pd.DataFrame,
    city_col: str = "merchant_city",
    state_col: str = "merchant_state_imputed",
    zip_col: str = "zip",
    output_col: str = "zip_imputed"
) -> pd.DataFrame:
    """
    Imputa zip usando la moda de zip por combinación de ciudad y estado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.
    city_col : str
        Columna con la ciudad del comercio.
    state_col : str
        Columna con el estado del comercio, idealmente ya imputado.
    zip_col : str
        Columna original del zip.
    output_col : str
        Nombre de la nueva columna imputada.

    Returns
    -------
    pd.DataFrame
        DataFrame con la columna output_col añadida.
    """

    df = df.copy()

    city_state_to_zip = create_mode_mapping(
        df=df,
        group_cols=[city_col, state_col],
        target_col=zip_col
    )

    df[output_col] = df[zip_col].astype(object)

    mask_missing = df[output_col].isna()

    keys = list(
        zip(
            df.loc[mask_missing, city_col],
            df.loc[mask_missing, state_col]
        )
    )

    df.loc[mask_missing, output_col] = [
        city_state_to_zip.get(key) for key in keys
    ]

    return df

def impute_location_fields(
    df: pd.DataFrame,
    city_col: str = "merchant_city",
    state_col: str = "merchant_state",
    zip_col: str = "zip",
    state_output_col: str = "merchant_state_imputed",
    zip_output_col: str = "zip_imputed",
    add_missing_indicators: bool = True
) -> pd.DataFrame:
    """
    Aplica la imputación de campos geográficos:
    1. Imputa merchant_state usando la moda por merchant_city.
    2. Imputa zip usando la moda por merchant_city + merchant_state imputado.
    3. Opcionalmente crea variables indicadoras de valores faltantes originales.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.
    city_col : str
        Columna de ciudad.
    state_col : str
        Columna de estado.
    zip_col : str
        Columna de zip.
    state_output_col : str
        Nombre de la columna imputada para estado.
    zip_output_col : str
        Nombre de la columna imputada para zip.
    add_missing_indicators : bool
        Si True, crea columnas binarias indicando si el valor original era nulo.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas imputadas añadidas.
    """

    df = df.copy()

    # Normalizar espacios en columnas de texto para mejorar el mapeo
    for col in [city_col, state_col, zip_col]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()

    if add_missing_indicators:
        df[f"{state_col}_was_missing"] = df[state_col].isna().astype(int)
        df[f"{zip_col}_was_missing"] = df[zip_col].isna().astype(int)

    df = impute_merchant_state_by_city(
        df=df,
        city_col=city_col,
        state_col=state_col,
        output_col=state_output_col
    )

    df = impute_zip_by_city_state(
        df=df,
        city_col=city_col,
        state_col=state_output_col,
        zip_col=zip_col,
        output_col=zip_output_col
    )

    # Cuando la imputación no tiene datos, marcar como UNKNOWN en lugar de dejar nulos
    if df[state_output_col].dtype == object:
        df[state_output_col] = df[state_output_col].fillna('UNKNOWN')
    else:
        df[state_output_col] = df[state_output_col].astype(object).fillna('UNKNOWN')

    df[zip_output_col] = df[zip_output_col].astype(object).fillna('UNKNOWN')

    return df



def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los nombres de columnas:
    - Quita espacios al inicio/final.
    - Convierte a minúsculas.
    - Reemplaza espacios por guiones bajos.
    - Elimina signos de interrogación.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("?", "", regex=False)
    )

    return df


def get_unique_and_null_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un resumen de valores únicos y nulos por columna.
    
    Incluye:
    - Nombre de columna
    - Tipo de dato
    - Número de valores únicos (incluyendo NaN)
    - Número de valores faltantes
    - Porcentaje de valores faltantes (redondeado a 4 decimales)
    
    Ordenado por número de valores únicos descendente.
    """
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "unique_values": df.nunique(dropna=False).values,
        "missing_values": df.isna().sum().values,
        "missing_percentage": (df.isna().mean() * 100).round(4).values
    })
    
    summary = summary.sort_values("unique_values", ascending=False).reset_index(drop=True)
    
    return summary


def finalize_transaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas redundantes y ordena las columnas según el orden original del dataset.

    Se eliminan:
    - merchant_state
    - zip
    - errors / errors?

    Si existen columnas imputadas, se renombran para usar los nombres originales:
    - merchant_state_imputed -> merchant_state
    - zip_imputed -> zip
    """
    df = df.copy()

    columns_to_drop = [
        col for col in ["merchant_state", "zip", "errors", "errors?"]
        if col in df.columns
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    if "merchant_state_imputed" in df.columns:
        df = df.rename(columns={"merchant_state_imputed": "merchant_state"})
    if "zip_imputed" in df.columns:
        df = df.rename(columns={"zip_imputed": "zip"})

    desired_order = [
        "user",
        "card",
        "year",
        "month",
        "day",
        "time",
        "amount",
        "use_chip",
        "merchant_name",
        "merchant_city",
        "merchant_state",
        "zip",
        "mcc",
        "is_fraud",
        "is_fraud?",
    ]

    ordered_columns = [col for col in desired_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in ordered_columns]

    return df[ordered_columns + remaining_columns]