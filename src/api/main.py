from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "pipeline_optimizado_mibici.pkl"

app = FastAPI(
    title="MiBici Anomaly Detection API",
    description="Servicio de inferencia para detección de anomalías en viajes de MiBici usando Pipeline (StandardScaler + PCA + Isolation Forest).",
    version="1.0.0",
)

modelo_pipeline = None


class PredictRequest(BaseModel):
    """Schema de entrada: lista de registros (filas) con variables numéricas del viaje."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "records": [
                        {
                            "duracion_seg": 900.0,
                            "edad": 28,
                            "lat_origen": 20.6736,
                            "lon_origen": -103.344,
                            "lat_destino": 20.6812,
                            "lon_destino": -103.351,
                        }
                    ]
                }
            ]
        }
    )
    records: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Lista de diccionarios representando cada viaje/registro a evaluar.",
    )


class PrediccionItem(BaseModel):
    indice: int
    prediccion: int = Field(..., description="-1 = anomalía, 1 = normal")
    etiqueta: str = Field(..., description="'anomalia' o 'normal'")


class PredictResponse(BaseModel):
    """Schema de salida de la inferencia."""
    modelo_version: str
    total_registros: int
    anomalias_detectadas: int
    predicciones: List[PrediccionItem]


@app.on_event("startup")
def cargar_modelo() -> None:
    global modelo_pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No se encontró el artefacto del modelo en: {MODEL_PATH}")
    modelo_pipeline = joblib.load(MODEL_PATH)


@app.get("/health", tags=["Salud"])
def health_check() -> Dict[str, str]:
    return {"status": "ok", "model_loaded": str(modelo_pipeline is not None)}


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
def predict(payload: PredictRequest) -> PredictResponse:
    if modelo_pipeline is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado en memoria.")

    try:
        df_input = pd.DataFrame(payload.records)
        X_num = df_input.select_dtypes(include="number")
        if X_num.empty:
            raise ValueError("No se encontraron columnas numéricas válidas en los registros enviados.")

        # Alinear columnas con las usadas en el entrenamiento si el modelo las expone
        expected_cols = getattr(modelo_pipeline, "feature_names_in_", None)
        if expected_cols is None and hasattr(modelo_pipeline, "named_steps"):
            for step in modelo_pipeline.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    expected_cols = step.feature_names_in_
                    break

        if expected_cols is not None:
            X_num = X_num.reindex(columns=expected_cols)

        X_clean = X_num.fillna(X_num.median(numeric_only=True)).fillna(0.0)
        preds = modelo_pipeline.predict(X_clean)

        items = [
            PrediccionItem(
                indice=i,
                prediccion=int(p),
                etiqueta="anomalia" if int(p) == -1 else "normal",
            )
            for i, p in enumerate(preds)
        ]
        anomalias = sum(1 for item in items if item.prediccion == -1)

        return PredictResponse(
            modelo_version="models:/MiBici_IsolationForest_Model/1",
            total_registros=len(items),
            anomalias_detectadas=anomalias,
            predicciones=items,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {exc}") from exc