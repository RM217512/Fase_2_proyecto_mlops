import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

# Asegurar que la raíz del proyecto esté en sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.build_features import FeatureEngineer
from src.models.train_model import MiBiciPipeline


@pytest.fixture
def config_path() -> str:
    return str(ROOT_DIR / "config.yaml")


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Carga un muestra real de data/raw si existe; si no, genera datos sintéticos."""
    raw_dir = ROOT_DIR / "data" / "raw"
    csv_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if csv_files:
        return pd.read_csv(csv_files[0]).head(200)

    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id_usuario": np.arange(1, 201),
        "duracion_seg": rng.normal(900, 200, 200).clip(60, 5000),
        "edad": rng.integers(18, 65, 200),
        "lat_origen": rng.normal(20.67, 0.02, 200),
        "lon_origen": rng.normal(-103.35, 0.02, 200),
        "lat_destino": rng.normal(20.68, 0.02, 200),
        "lon_destino": rng.normal(-103.34, 0.02, 200),
    })


def test_unit_preprocesamiento_feature_engineer(config_path: str, sample_raw_df: pd.DataFrame):
    """Prueba unitaria: verifica que FeatureEngineer limpie y devuelva DataFrame válido."""
    ingeniero = FeatureEngineer(config_path=config_path)
    df_limpio = ingeniero.limpiar_columnas_inutiles(sample_raw_df)

    assert isinstance(df_limpio, pd.DataFrame)
    assert not df_limpio.empty
    assert len(df_limpio) == len(sample_raw_df)


def test_unit_inferencia_y_metricas(config_path: str, sample_raw_df: pd.DataFrame):
    """Prueba unitaria: valida creación del pipeline, predicción (-1 / 1) y métrica."""
    X = sample_raw_df.select_dtypes(include="number").fillna(0)
    assert X.shape[1] >= 2

    ml_pipe = MiBiciPipeline(config_path=config_path)
    pipe = ml_pipe.crear_pipeline(n_estimators=50, contamination=0.05)
    pipe.fit(X)
    preds = pipe.predict(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({-1, 1})

    score = silhouette_score(X, preds, sample_size=min(100, len(X)), random_state=42)
    assert -1.0 <= score <= 1.0


def test_integracion_flujo_extremo_a_extremo(config_path: str, sample_raw_df: pd.DataFrame):
    """Prueba de integración: carga -> preprocesamiento -> predicción -> métricas."""
    # 1. Datos cargados (fixture) y preprocesamiento
    ingeniero = FeatureEngineer(config_path=config_path)
    df_limpio = ingeniero.limpiar_columnas_inutiles(sample_raw_df)
    X = df_limpio.drop(columns=getattr(ingeniero, "col_id", []), errors="ignore").select_dtypes(include="number")
    X = X.fillna(X.median(numeric_only=True))

    # 2. Entrenamiento y predicción
    ml_pipe = MiBiciPipeline(config_path=config_path)
    pipe = ml_pipe.crear_pipeline(n_estimators=50, contamination=0.05)
    pipe.fit(X)
    predicciones = pipe.predict(X)

    # 3. Cálculo de métricas de evaluación
    score = silhouette_score(X, predicciones, sample_size=min(150, len(X)), random_state=42)
    tasa_anomalias = float((predicciones == -1).mean())

    assert predicciones.shape == (len(X),)
    assert -1.0 <= score <= 1.0
    assert 0.0 <= tasa_anomalias <= 1.0