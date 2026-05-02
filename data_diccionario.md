# Diccionario de datos - Credit Card Transactions Dataset

Este documento describe las variables principales del conjunto de datos de transacciones con tarjetas de crédito usado en el proyecto de detección de fraude.

El dataset corresponde al conjunto sintético de transacciones de tarjetas publicado por IBM en el repositorio TabFormer. De acuerdo con la documentación oficial del repositorio, el dataset se encuentra en `data/credit_card` y contiene aproximadamente 24 millones de registros transaccionales. El paper asociado describe el uso de este conjunto sintético para tareas de detección de fraude y generación de datos sintéticos.  

Fuentes principales:

- IBM TabFormer GitHub Repository: https://github.com/IBM/TabFormer
- Paper: Tabular Transformers for Modeling Multivariate Time Series: https://arxiv.org/abs/2011.01843

---

## 1. Descripción general

El conjunto de datos contiene transacciones realizadas con tarjetas de crédito. Cada fila representa una transacción individual e incluye información temporal, monto, canal de uso de la tarjeta, datos del comercio, posibles errores transaccionales y una etiqueta binaria que indica si la transacción fue fraudulenta.

El objetivo principal del proyecto es utilizar estas variables para construir modelos de detección de fraude transaccional.

---

## 2. Convención de nombres

Las columnas originales del dataset se limpian usando las siguientes reglas:

- Convertir nombres a minúsculas.
- Eliminar espacios al inicio y al final.
- Reemplazar espacios internos por guiones bajos.
- Eliminar signos de interrogación.

Ejemplo:

| Nombre original | Nombre limpio |
|---|---|
| `User` | `user` |
| `Use Chip` | `use_chip` |
| `Merchant Name` | `merchant_name` |
| `Errors?` | `errors` |
| `Is Fraud?` | `is_fraud` |

---

## 3. Diccionario de variables

| Columna original | Columna limpia | Descripción | Tipo conceptual | Tipo recomendado en pandas | Tratamiento recomendado |
|---|---|---|---|---|---|
| `User` | `user` | Identificador anónimo del usuario o tarjetahabiente. | Identificador / categórica de alta cardinalidad | `string` | No usar como variable numérica. Puede usarse para agregaciones por usuario o análisis secuencial. |
| `Card` | `card` | Identificador de la tarjeta asociada al usuario. | Identificador / categórica | `string` | No usar como variable numérica. Puede usarse para agregaciones por tarjeta. |
| `Year` | `year` | Año en que ocurrió la transacción. | Temporal discreta | `Int32` | Combinar con `month`, `day` y `time` para crear una variable temporal completa. |
| `Month` | `month` | Mes en que ocurrió la transacción. | Temporal cíclica | `Int32` | Usar para crear timestamp. También puede codificarse de forma cíclica. |
| `Day` | `day` | Día del mes en que ocurrió la transacción. | Temporal discreta | `Int32` | Usar para crear timestamp y extraer variables temporales adicionales. |
| `Time` | `time` | Hora y minuto de la transacción en formato `HH:MM`. | Temporal | `string` | Combinar con fecha para crear timestamp. También permite extraer hora, minuto o franja horaria. |
| `Amount` | `amount` | Monto monetario de la transacción. | Numérica continua | `float64` | Eliminar símbolo `$`, convertir a número y considerar transformaciones como `log1p`. |
| `Use Chip` | `use_chip` | Canal o modo de uso de la tarjeta. Por ejemplo, transacción con chip, banda magnética o en línea. | Categórica nominal | `string` | Tratar como categoría nominal. Puede usarse one-hot encoding, frequency encoding o encoding categórico. |
| `Merchant Name` | `merchant_name` | Identificador o nombre anonimizado del comercio. | Categórica de alta cardinalidad | `string` | No usar como numérica. Puede usarse frequency encoding, target encoding o agregaciones por comercio. |
| `Merchant City` | `merchant_city` | Ciudad donde está ubicado el comercio. | Categórica nominal | `string` | Usar como variable categórica y como apoyo para imputar `merchant_state` y `zip`. |
| `Merchant State` | `merchant_state` | Estado donde está ubicado el comercio. Puede contener valores nulos. | Categórica nominal | `string` | Imputar nulos usando la moda por `merchant_city`. Crear indicador de valor faltante. |
| `Zip` | `zip` | Código postal del comercio. Puede contener valores nulos. | Categórica nominal | `string` | No tratar como número continuo. Limpiar terminación `.0`, preservar como texto e imputar usando `merchant_city` + `merchant_state`. |
| `MCC` | `mcc` | Merchant Category Code. Código que representa la categoría del comercio. | Categórica nominal codificada | `string` o `Int32` | No interpretar como magnitud numérica. Usar como categoría. |
| `Errors?` | `errors` | Error registrado durante la transacción, si existe. Los nulos suelen indicar ausencia de error reportado. | Categórica nominal | `string` | Reemplazar nulos por `none` o crear indicador de valor faltante. |
| `Is Fraud?` | `is_fraud` | Variable objetivo que indica si la transacción fue fraudulenta. | Binaria / target | `Int32` | Convertir `No` a `0` y `Yes` a `1`. Usar como variable objetivo. |

---

## 4. Clasificación conceptual de las variables

### 4.1 Identificadores

Estas variables identifican entidades dentro del dataset, pero no representan magnitudes numéricas:

- `user`
- `card`
- `merchant_name`

Aunque algunas de estas columnas pueden verse como números, deben tratarse como identificadores o variables categóricas de alta cardinalidad.

---

### 4.2 Variables temporales

Estas variables describen el momento de la transacción:

- `year`
- `month`
- `day`
- `time`

Se recomienda combinarlas para crear una variable `timestamp`.

Ejemplo de variables derivadas posibles:

- `transaction_hour`
- `transaction_minute`
- `transaction_dayofweek`
- `transaction_weekend`
- `transaction_month`
- `transaction_period_of_day`

---

### 4.3 Variable numérica continua

La principal variable numérica continua es:

- `amount`

Esta variable representa el monto monetario de la transacción. Puede requerir transformaciones adicionales debido a asimetría o valores extremos.

---

### 4.4 Variables categóricas del comercio

Estas variables describen el comercio donde ocurrió la transacción:

- `merchant_name`
- `merchant_city`
- `merchant_state`
- `zip`
- `mcc`

Notas importantes:

- `merchant_name` puede tener alta cardinalidad.
- `zip` debe tratarse como texto, no como variable numérica.
- `mcc` es un código de categoría comercial, no una variable continua.
- `merchant_state` y `zip` pueden contener valores faltantes.

---

### 4.5 Variables transaccionales categóricas

Estas variables describen características específicas de la operación:

- `use_chip`
- `errors`

`use_chip` indica el canal o modo de uso de la tarjeta.  
`errors` registra errores asociados a la transacción, cuando existen.

---

### 4.6 Variable objetivo

La variable objetivo del problema de clasificación es:

- `is_fraud`

Valores esperados:

| Valor original | Valor transformado |
|---|---:|
| `No` | `0` |
| `Yes` | `1` |

---

## 5. Reglas de preprocesamiento recomendadas

### 5.1 Limpieza de nombres de columnas

Se recomienda transformar los nombres de columnas antes de cualquier procesamiento:

```python
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("?", "", regex=False)
)
