# Diccionario de datos - Credit Card Transactions Dataset

Este documento describe las variables del conjunto de datos de transacciones con tarjetas de crédito utilizado en el proyecto de detección de fraude.

Cada fila del dataset representa una transacción individual. Las columnas contienen información del usuario, tarjeta, fecha y hora de la transacción, monto, canal de uso de la tarjeta, comercio, ubicación del comercio, posibles errores transaccionales y la etiqueta objetivo de fraude.

---

## 1. Diccionario de variables

| Columna | Descripción | Tipo conceptual |
|---|---|---|
| `User` | Identificador anónimo del usuario o tarjetahabiente que realiza la transacción. | Identificador |
| `Card` | Identificador de la tarjeta asociada al usuario. Representa la tarjeta usada por el usuario en la transacción. | Identificador |
| `Year` | Año en que ocurrió la transacción. | Temporal |
| `Month` | Mes en que ocurrió la transacción. | Temporal |
| `Day` | Día del mes en que ocurrió la transacción. | Temporal |
| `Time` | Hora y minuto en que ocurrió la transacción, en formato `HH:MM`. | Temporal |
| `Amount` | Monto monetario de la transacción. | Numérica continua |
| `Use Chip` | Método o canal de uso de la tarjeta durante la transacción, por ejemplo transacción con chip, banda magnética o en línea. | Categórica nominal |
| `Merchant Name` | Identificador anonimizado del comercio donde ocurrió la transacción. | Identificador / categórica |
| `Merchant City` | Ciudad donde está ubicado el comercio. | Categórica geográfica |
| `Merchant State` | Estado, región o país asociado a la ubicación del comercio. | Categórica geográfica |
| `Zip` | Código postal asociado a la ubicación del comercio. | Categórica geográfica |
| `MCC` | Merchant Category Code. Código que representa la categoría comercial del establecimiento. | Categórica codificada |
| `Errors?` | Error registrado durante la transacción, si existe. | Categórica nominal |
| `Is Fraud?` | Indica si la transacción fue fraudulenta. | Variable objetivo binaria |
