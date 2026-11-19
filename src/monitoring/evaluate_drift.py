from pathlib import Path
import json
import joblib
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score

PSI_ALERT_THRESHOLD = 0.20
KS_PVALUE_THRESHOLD = 0.05
ANOMALY_RATE_DELTA_THRESHOLD = 0.10


def get_model_features(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    raise ValueError("No se pudieron detectar feature_names_in_ en el modelo.")


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    quantiles = np.linspace(0, 100, buckets + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] -= 1e-6
    bins[-1] += 1e-6

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = np.clip(exp_counts / max(len(expected), 1), 1e-4, None)
    act_pct = np.clip(act_counts / max(len(actual), 1), 1e-4, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def build_reference_for_features(features: list[str], seed: int = 42) -> pd.DataFrame:
    # Intentar cargar CSV si ya tiene las columnas exactas del modelo
    for csv_path in Path("data").rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path, nrows=5000)
            if all(c in df.columns for c in features):
                clean = df[features].dropna()
                if len(clean) >= 100:
                    return clean.head(3000)
        except Exception:
            continue

    rng = np.random.default_rng(seed)
    data = {}
    for col in features:
        low = col.lower()
        if "año" in low or "nacimiento" in low or "birth" in low:
            data[col] = rng.integers(1965, 2003, size=2000).astype(float)
        elif "edad" in low or "age" in low:
            data[col] = rng.normal(32, 9, size=2000).clip(18, 75)
        elif "duracion" in low or "duration" in low or "seg" in low or "min" in low:
            data[col] = rng.normal(780, 220, size=2000).clip(60, 3600)
        elif "lat" in low:
            data[col] = rng.normal(20.675, 0.015, size=2000)
        elif "lon" in low:
            data[col] = rng.normal(-103.348, 0.015, size=2000)
        elif "hora" in low or "hour" in low:
            data[col] = rng.integers(0, 24, size=2000).astype(float)
        elif "dia" in low or "day" in low or "weekday" in low:
            data[col] = rng.integers(0, 7, size=2000).astype(float)
        else:
            data[col] = rng.normal(0.0, 1.0, size=2000)
    return pd.DataFrame(data)[features]


def simulate_drifted_data(df_ref: pd.DataFrame, features: list[str], seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # 1. Cambio estacional / desplazamiento de medias
    df_seasonal = df_ref.copy()
    for col in features:
        low = col.lower()
        s = df_seasonal[col].to_numpy(dtype=float)
        if "año" in low or "nacimiento" in low:
            df_seasonal[col] = np.clip(s + rng.normal(-6.0, 3.0, size=len(s)), 1940, 2008)
        elif "edad" in low:
            df_seasonal[col] = np.clip(s + rng.normal(6.5, 3.0, size=len(s)), 16, 90)
        elif "duracion" in low or "duration" in low or "seg" in low or "min" in low:
            df_seasonal[col] = np.clip(s * 1.45 + rng.normal(180, 90, size=len(s)), 30, None)
        elif "lat" in low or "lon" in low:
            df_seasonal[col] = s + rng.normal(0.015, 0.008, size=len(s))
        else:
            std = float(np.std(s)) if float(np.std(s)) > 1e-6 else 1.0
            df_seasonal[col] = s + rng.normal(0.8 * std, 0.3 * std, size=len(s))

    # 2. Valores faltantes en variables de entrada
    df_missing = df_ref.copy()
    for col in features[: min(2, len(features))]:
        mask_na = rng.random(size=len(df_ref)) < 0.15
        df_missing.loc[mask_na, col] = np.nan

    return {
        "referencia": df_ref.copy(),
        "drift_estacional_media": df_seasonal,
        "drift_faltantes": df_missing,
    }


def evaluate_set(model, features: list[str], df_ref: pd.DataFrame, df_curr: pd.DataFrame, base_stats: dict | None = None):
    curr_clean = df_curr[features].copy()
    for col in features:
        if curr_clean[col].isna().any():
            fill_val = float(df_ref[col].median())
            curr_clean[col] = curr_clean[col].fillna(fill_val)

    preds = model.predict(curr_clean)
    anomaly_rate = float(np.mean(preds == -1))

    n_sample = min(len(curr_clean), 2000)
    idx = np.linspace(0, len(curr_clean) - 1, n_sample, dtype=int)
    sil = float(silhouette_score(curr_clean.iloc[idx], preds[idx])) if len(np.unique(preds[idx])) > 1 else 0.0

    feature_metrics = {}
    drifted_features = []
    for col in features:
        ref_vals = df_ref[col].to_numpy(dtype=float)
        cur_vals = curr_clean[col].to_numpy(dtype=float)
        psi = compute_psi(ref_vals, cur_vals)
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
        mean_shift = float(np.mean(cur_vals) - np.mean(ref_vals))

        is_drifted = bool(psi >= PSI_ALERT_THRESHOLD or ks_pval < KS_PVALUE_THRESHOLD)
        if is_drifted:
            drifted_features.append(col)

        feature_metrics[col] = {
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 6),
            "mean_ref": round(float(np.mean(ref_vals)), 4),
            "mean_curr": round(float(np.mean(cur_vals)), 4),
            "mean_shift": round(mean_shift, 4),
            "alert_drift": is_drifted,
        }

    base_anom = base_stats["anomaly_rate"] if base_stats else anomaly_rate
    anom_delta = abs(anomaly_rate - base_anom)
    alert_triggered = bool(len(drifted_features) > 0 or anom_delta >= ANOMALY_RATE_DELTA_THRESHOLD)

    if anom_delta >= ANOMALY_RATE_DELTA_THRESHOLD or max(m["psi"] for m in feature_metrics.values()) >= 0.35:
        recommended_action = "RETRAIN_MODEL: Reentrenar Isolation Forest con datos recientes y recalibrar umbrales."
    elif alert_triggered:
        recommended_action = "CHECK_FEATURE_PIPELINE: Auditar transformaciones, valores faltantes y estacionalidad en features."
    else:
        recommended_action = "NO_ACTION: Distribución estable dentro de umbrales."

    return {
        "total_records": int(len(curr_clean)),
        "missing_ratio_raw": round(float(df_curr[features].isna().mean().mean()), 4),
        "anomaly_rate": round(anomaly_rate, 4),
        "anomaly_rate_delta_vs_base": round(float(anom_delta), 4),
        "silhouette_proxy": round(sil, 4),
        "drifted_features": drifted_features,
        "alert_triggered": alert_triggered,
        "recommended_action": recommended_action,
        "features": feature_metrics,
    }


def save_plots(scenarios_eval: dict, features: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(scenarios_eval.keys())
    rates = [scenarios_eval[k]["anomaly_rate"] * 100 for k in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, rates)
    ax.set_ylabel("Tasa de anomalías (%)")
    ax.set_title("Impacto de Data Drift en la Tasa de Anomalías (MiBici)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    anom_plot = out_dir / "drift_anomaly_rate.png"
    plt.savefig(anom_plot, dpi=150)
    plt.close(fig)

    psi_vals = [scenarios_eval["drift_estacional_media"]["features"][f]["psi"] for f in features]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(features, psi_vals)
    ax.axvline(PSI_ALERT_THRESHOLD, linestyle="--", label=f"Umbral PSI ({PSI_ALERT_THRESHOLD})")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("PSI por Variable — Escenario Estacional / Desplazamiento")
    ax.legend()
    plt.tight_layout()
    psi_plot = out_dir / "drift_psi_features.png"
    plt.savefig(psi_plot, dpi=150)
    plt.close(fig)

    return [anom_plot, psi_plot]


def main():
    model_path = Path("models/pipeline_optimizado_mibici.pkl")
    model = joblib.load(model_path)
    features = get_model_features(model)

    df_ref = build_reference_for_features(features, seed=42)
    scenarios_df = simulate_drifted_data(df_ref, features, seed=42)

    base_eval = evaluate_set(model, features, df_ref, scenarios_df["referencia"])
    scenarios_eval = {"referencia": base_eval}

    for name, df_sc in scenarios_df.items():
        if name == "referencia":
            continue
        scenarios_eval[name] = evaluate_set(model, features, df_ref, df_sc, base_stats=base_eval)

    reports_dir = Path("reports/drift")
    plots = save_plots(scenarios_eval, features, reports_dir)

    summary_path = reports_dir / "drift_report.json"
    summary_path.write_text(json.dumps(scenarios_eval, indent=2, ensure_ascii=False), encoding="utf-8")

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("MiBici_Data_Drift_Monitoring")
    with mlflow.start_run(run_name="drift_simulation_v1"):
        for name, res in scenarios_eval.items():
            mlflow.log_metric(f"{name}_anomaly_rate", res["anomaly_rate"])
            mlflow.log_metric(f"{name}_anomaly_delta", res["anomaly_rate_delta_vs_base"])
            mlflow.log_metric(f"{name}_alert_triggered", int(res["alert_triggered"]))
        mlflow.log_artifact(str(summary_path))
        for p in plots:
            mlflow.log_artifact(str(p))

    for name, res in scenarios_eval.items():
        print(f"=== Escenario: {name} ===")
        print(f"  Tasa anomalías : {res['anomaly_rate']*100:.2f}% (delta vs base: {res['anomaly_rate_delta_vs_base']*100:.2f}%)")
        print(f"  Features drift : {res['drifted_features']}")
        print(f"  Alerta activa  : {res['alert_triggered']}")
        print(f"  Acción         : {res['recommended_action']}")


if __name__ == "__main__":
    main()