Proyecto Fase 2
==============================

Estructurado y versionado para fase 2

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

### Pruebas Automatizadas (Unitarias e Integración)
Las pruebas se encuentran en `tests/test_pipeline_mibici.py` y cubren preprocesamiento (`FeatureEngineer`), inferencia, cálculo de métricas y el flujo completo de extremo a extremo.

Para ejecutarlas desde la raíz del proyecto:
```bash
pytest -q

### Serving del Modelo con FastAPI
El servicio expone la inferencia mediante FastAPI en `src/api/main.py`.

* **Artefacto local:** `models/pipeline_optimizado_mibici.pkl`
* **Registro en MLflow Model Registry:** `models:/MiBici_IsolationForest_Model/1`
* **Endpoints principales:**
  * `GET /health`: verifica el estado del servicio y carga del modelo.
  * `POST /predict`: recibe `{"records": [{...}, ...]}` (validado con Pydantic) y retorna `modelo_version`, `total_registros`, `anomalias_detectadas` y la lista de `predicciones` (`1` normal, `-1` anomalía).

Para iniciar el servidor localmente:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

### Reproducibilidad en Entorno Limpio (Contenedor)
Las dependencias fijas se encuentran en `requirements.txt` y las semillas aleatorias(`random_state=42`) en `config.yaml`.

* **Modelo registrado (MLflow Model Registry):** `models:/MiBici_IsolationForest_Model/1`
* **Artefacto binario empaquetado:** `models/pipeline_optimizado_mibici.pkl`
* **Entorno de validación:** Contenedor limpio construido con `Dockerfile` y ejecutado mediante **Podman** en WSL (Ubuntu).

Para construir y ejecutar el servicio en un contenedor limpio:

```bash
podman build -t mibici-api:v1 .
podman run -d --name mibici-repro -p 8001:8000 mibici-api:v1

### Contenedorización y Registro de Imágenes
El servicio de inferencia (FastAPI + modelo `models/pipeline_optimizado_mibici.pkl`) está empaquetado mediante `Dockerfile` y optimizado con `.dockerignore`.

Construcción y ejecución local:
```bash
podman build -t ml-service:latest .
podman run -p 8000:8000 ml-service:latest

### Simulación y Monitoreo de Data Drift
El script `src/monitoring/evaluate_drift.py` inspecciona las variables esperadas por `models/pipeline_optimizado_mibici.pkl` (`Año_de_nacimiento`), simula escenarios de distribución alterada y compara su comportamiento contra la línea base.

* **Escenarios evaluados:**
  * `referencia`: Tasa de anomalías del `8.63%` (`delta: 0.00%`), sin alertas (`NO_ACTION`).
  * `drift_estacional_media`: Tasa de anomalías del `8.17%` (`delta: 0.46%`), drift detectado en `Año_de_nacimiento` (`alerta: True`).
  * `drift_faltantes`: Tasa de anomalías del `7.40%` (`delta: 1.23%`), drift detectado en `Año_de_nacimiento` (`alerta: True` — `CHECK_FEATURE_PIPELINE`).
* **Criterios de alerta y acciones:** `PSI >= 0.20`, prueba KS (`p-value < 0.05`) o variación en tasa de anomalías `>= 10%`; se proponen acciones de revisión del *feature pipeline* o *retrain* del modelo según severidad.
* **Artefactos y tracking:** Reporte JSON y gráficas guardados en `reports/drift/` y registrados en MLflow bajo el experimento `MiBici_Data_Drift_Monitoring`.