import yaml
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import itertools

class MiBiciPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.random_state = self.config['model']['random_state']
        self.pca_comp = self.config['model']['pca_components']
        self.mejor_pipeline = None
        
    def crear_pipeline(self, n_estimators=100, contamination='auto'):
        """Crea una instancia del pipeline con hiperparámetros específicos."""
        return Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.pca_comp, random_state=self.random_state)),
            ('anomaly_detector', IsolationForest(
                n_estimators=n_estimators, 
                contamination=contamination, 
                random_state=self.random_state))
        ])

    def optimizar_hiperparametros(self, X):
        """Ajuste de hiperparámetros iterando sobre la malla de config.yaml."""
        print("Iniciando ajuste de hiperparámetros...")
        n_est_list = self.config['tuning']['n_estimators_list']
        cont_list = self.config['tuning']['contamination_list']
        
        mejor_score = -1
        mejores_params = {}
        
        # Iterar sobre todas las combinaciones posibles
        for n_est, cont in itertools.product(n_est_list, cont_list):
            pipeline_temp = self.crear_pipeline(n_estimators=n_est, contamination=cont)
            pipeline_temp.fit(X)
            predicciones = pipeline_temp.predict(X)
            
            # Evaluar usando una muestra para evitar saturar la memoria RAM
            score = silhouette_score(X, predicciones, sample_size=5000, random_state=self.random_state)
            print(f"Probando n_estimators={n_est}, contamination={cont} -> Silhouette Score: {score:.4f}")
            
            if score > mejor_score:
                mejor_score = score
                mejores_params = {'n_estimators': n_est, 'contamination': cont}
                self.mejor_pipeline = pipeline_temp
                
        print(f"\nMejores hiperparámetros encontrados: {mejores_params} con Score: {mejor_score:.4f}")
        return self.mejor_pipeline

    def evaluar_modelo(self, X):
        """Evalúa el modelo optimizado calculando métricas de rendimiento específicas."""
        if not self.mejor_pipeline:
            raise ValueError("Primero debes ejecutar optimizar_hiperparametros().")
            
        predicciones = self.mejor_pipeline.predict(X)
        score_final = silhouette_score(X, predicciones, sample_size=10000, random_state=self.random_state)
        
        anomalias = (predicciones == -1).sum()
        normales = (predicciones == 1).sum()
        
        print("\n--- REPORTE DE EVALUACIÓN ---")
        print(f"Silhouette Score (Rendimiento de separación): {score_final:.4f}")
        print(f"Viajes normales detectados: {normales}")
        print(f"Anomalías detectadas: {anomalias}")
        print(f"Porcentaje de anomalías: {(anomalias / len(predicciones)) * 100:.2f}%")
        
        return predicciones

    def guardar_modelo(self, ruta_guardado="models/pipeline_optimizado_mibici.pkl"):
        """Exporta el pipeline optimizado."""
        joblib.dump(self.mejor_pipeline, ruta_guardado)
        print(f"\nPipeline optimizado guardado en: {ruta_guardado}")