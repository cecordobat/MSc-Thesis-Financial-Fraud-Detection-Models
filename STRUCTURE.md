# Estructura del Repositorio - MSc Thesis Financial Fraud Detection Models

```
MSc-Thesis-Financial-Fraud-Detection-Models/
│
├── configs/                          # Configuraciones y variables de entorno
│   ├── .gitkeep                      # Mantener carpeta en git
│   └── paths.example.env             # Ejemplo de configuración de rutas
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
│   ├── interim/                      # Datos intermedios (procesamiento temporal)
│   │   └── .gitkeep                  # Mantener carpeta en git
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
│   ├── data/                         # Scripts para carga y procesamiento de datos
│   │   └── .gitkeep                  # (vacío - pendiente de implementación)
│   │
│   ├── features/                     # Ingeniería de características
│   │   └── .gitkeep                  # (vacío - pendiente de implementación)
│   │
│   ├── models/                       # Implementación de modelos ML/DL
│   │   └── .gitkeep                  # (vacío - pendiente de implementación)
│   │
│   └── utils/                        # Funciones utilitarias
│       └── .gitkeep                  # (vacío - pendiente de implementación)
│
├── artifacts/                        # Resultados y artefactos del proyecto
│   ├── models/                       # Modelos entrenados guardados
│   │   └── .gitkeep                  # Mantener carpeta en git
│   │
│   ├── metrics/                      # Métricas de evaluación de modelos
│   │   └── .gitkeep                  # Mantener carpeta en git
│   │
│   └── figures/                      # Gráficas y visualizaciones
│       └── .gitkeep                  # Mantener carpeta en git
│
├── README.md                         # Documentación principal del proyecto
├── requirements.txt                  # Dependencias de Python
├── .gitignore                        # Archivos ignorados por Git
├── .env.example                      # Ejemplo de variables de entorno
├── LICENSE                           # Licencia MIT
├── STRUCTURE.md                      # Este archivo de estructura
└── .git/                             # Control de versiones Git
```
