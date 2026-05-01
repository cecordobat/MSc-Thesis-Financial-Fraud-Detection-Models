# Estructura del Repositorio - MSc Thesis Financial Fraud Detection Models

```
MSc-Thesis-Financial-Fraud-Detection-Models/
│
├── README.md                          # Documentación principal del proyecto
├── LICENSE                            # Licencia MIT
├── requirements.txt                   # Dependencias de Python
├── .env                              # Variables de entorno (API keys, etc.)
├── .git/                             # Control de versiones Git
├── .gitignore                        # Archivos ignorados por Git
│
├── data/                             # Carpeta de datos
│   ├── original/                     # Datos crudos originales
│   │   ├── HI-Large_accounts.csv     # Dataset completo de cuentas
│   │   ├── HI-Large_Patterns.txt     # Patrones completos
│   │   ├── HI-Large_Trans.csv        # Transacciones completas
│   │   ├── muestra_accounts.csv      # Muestra de cuentas
│   │   ├── muestra_patterns.csv      # Muestra de patrones
│   │   └── muestra_tranx.csv         # Muestra de transacciones
│   │
│   └── processed/                    # Datos procesados y listos para modelos
│       └── HI-Large_accounts_limpio.parquet  # Cuentas procesadas
│
├── notebooks/                        # Análisis exploratorio y experimentos
│   ├── 1_EDA_users.ipynb            # Análisis exploratorio de usuarios/cuentas
│   ├── 2_EDA_patterns.ipynb         # Análisis exploratorio de patrones
│   ├── 3_EDA_tranx.ipynb            # Análisis exploratorio de transacciones
│   ├── 14_03_notebook_class_1.ipynb # Análisis de clases/desbalance
│   ├── Ejemplo_Esquema_Entrega_4_mod1.ipynb  # Ejemplo de esquema de entrega
│   ├── muestra_accounts.csv         # Datos auxiliares en notebooks
│   └── Universidad_Nacional_de_Colombia_Coat_of_Arms_Redesign_Green_(2016).svg
│
├── src/                              # Código fuente modular
│   ├── features/                     # Ingeniería de características
│   │   └── .gitkeep                 # (vacío - pendiente de implementación)
│   │
│   ├── models/                       # Implementación de modelos ML/DL
│   │   └── .gitkeep                 # (vacío - pendiente de implementación)
│   │
│   └── utils/                        # Funciones utilitarias
│       └── .gitkeep                 # (vacío - pendiente de implementación)
│
└── docs/                             # Documentación de la tesis
    └── .gitkeep                      # (vacío - pendiente de documentación)
```

## Descripción por Carpeta

### 📁 `/data/`
- **original/**: Datos crudos sin procesar
  - Versiones completas (HI-Large) para análisis
  - Muestras (muestra) para desarrollo y pruebas rápidas
- **processed/**: Datos limpios y transformados listos para entrenar modelos

### 📁 `/notebooks/`
Jupyter Notebooks para:
- Exploración de datos (EDA)
- Análisis de desbalance de clases
- Experimentos iniciales
- Documentación del análisis

### 📁 `/src/`
Código fuente modular organizado en 3 módulos:
- **features/**: Funciones para limpieza, transformación y generación de características
- **models/**: Clases y funciones para entrenar/evaluar modelos (RF, XGBoost, NN)
- **utils/**: Funciones auxiliares (visualizaciones, métricas, etc.)

### 📁 `/docs/`
Documentación de la tesis, artículos, referencias

## Archivos Principales
- `README.md` - Descripción del proyecto y cómo usar
- `requirements.txt` - Librerías necesarias (pandas, scikit-learn, tensorflow, etc.)
- `.env` - Variables de entorno (OpenAI API key, etc.)
- `LICENSE` - Licencia MIT

## Próximos Pasos para Completar
- [ ] Implementar módulos en `/src/features/`
- [ ] Implementar módulos en `/src/models/`
- [ ] Implementar utilidades en `/src/utils/`
- [ ] Ejecutar y documentar resultados de notebooks
- [ ] Añadir documentación en `/docs/`
