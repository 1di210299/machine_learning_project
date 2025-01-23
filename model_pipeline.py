import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from scipy import stats, signal
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

warnings.filterwarnings('ignore', category=UserWarning)

class ModelPipeline:
    def __init__(self):
        self.models = {'elf': {}, 'mag': {}}
        self.scalers = {'elf': {}, 'mag': {}}
        self.feature_extractors = {'elf': {}, 'mag': {}}
        self.optimal_thresholds = {'elf': {}, 'mag': {}}
        self.feature_names = None
        
    def extract_sequence_features(self, data):
        """Extrae características de la secuencia temporal con características adicionales"""
        data = pd.Series(data).interpolate(method='linear', limit_direction='both').values
        
        if np.all(data == data[0]):
            data = data + np.random.normal(0, 1e-6, len(data))
        
        fft_vals = np.abs(np.fft.fft(data))
        
        features = {
            'max_value': np.max(data),
            'min_value': np.min(data),
            'peak_to_peak': np.max(data) - np.min(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'mean_crossing_rate': np.mean(np.abs(np.diff(np.sign(data - np.mean(data)))) / 2),
            'gradient_max': np.max(np.gradient(data)),
            'gradient_min': np.min(np.gradient(data)),
            'valley_value': np.min(np.abs(data)),
            'peak_to_valley_ratio': np.max(np.abs(data)) / (np.min(np.abs(data)) + 1e-6),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'energy_ratio': np.sum(data**2) / len(data),
            'trend': np.polyfit(np.arange(len(data)), data, 1)[0],
            'entropy': stats.entropy(fft_vals + 1e-10),
            'peak_frequency': np.argmax(fft_vals),
            'spectral_centroid': np.sum(fft_vals * np.arange(len(fft_vals))) / (np.sum(fft_vals) + 1e-10),
            'spectral_spread': np.sqrt(np.sum(fft_vals * (np.arange(len(fft_vals)) - np.sum(fft_vals * np.arange(len(fft_vals))) / (np.sum(fft_vals) + 1e-10))**2) / (np.sum(fft_vals) + 1e-10))
        }
        return features
        
    def segment_features(self, data, segment_size=None):
        """Extrae características por segmentos con métricas adicionales"""
        data = pd.Series(data).interpolate(method='linear', limit_direction='both').values
        
        if segment_size is None:
            segment_size = max(20, len(data) // 10)  # Ajuste dinámico
        
        segments = np.array_split(data, len(data)//segment_size)
        features = {}
        
        for i, segment in enumerate(segments):
            fft_segment = np.abs(np.fft.fft(segment))
            features.update({
                f'segment_{i}_mean': np.mean(segment),
                f'segment_{i}_std': np.std(segment),
                f'segment_{i}_max': np.max(segment),
                f'segment_{i}_min': np.min(segment),
                f'segment_{i}_peak_to_peak': np.max(segment) - np.min(segment),
                f'segment_{i}_rms': np.sqrt(np.mean(np.square(segment))),
                f'segment_{i}_entropy': stats.entropy(fft_segment + 1e-10),
                f'segment_{i}_trend': np.polyfit(np.arange(len(segment)), segment, 1)[0]
            })
        return features

    def preprocess_data(self, df):
        """Preprocesa los datos extrayendo características y maneja valores NaN"""
        feature_list = []
        
        # Limpiar NaN en el DataFrame completo
        df_cleaned = df.ffill().bfill()
        
        for _, row in df_cleaned.iterrows():
            data = row[[f'Data_{i}' for i in range(320)]].values
            
            features = {}
            try:
                features.update(self.extract_sequence_features(data))
                features.update(self.segment_features(data))
            except Exception as e:
                print(f"Error procesando fila: {e}")
                # Valores por defecto en caso de error
                features = {name: 0.0 for name in self.feature_names} if self.feature_names else {}
            
            feature_list.append(features)
        
        # Crear DataFrame y manejar valores NaN
        features_df = pd.DataFrame(feature_list)
        
        # Guardar nombres de características si no existen
        if self.feature_names is None:
            self.feature_names = features_df.columns.tolist()
        
        # Manejar valores NaN
        features_df = features_df.fillna(features_df.median())
        if features_df.isna().any().any():
            features_df = features_df.fillna(0)
        
        return features_df

    def balance_dataset(self, X, y, method='advanced', sensor_type=None):
        if isinstance(X, np.ndarray) and np.isnan(X).any():
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
        
        if method == 'advanced':
            try:
                if sensor_type == 'mag':
                    # Para MAG, primero sobremuestreo y luego submuestreo
                    smote = SMOTE(sampling_strategy=0.6, random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    # Calcular el número correcto de muestras para submuestreo
                    n_samples = len(y)
                    cluster = ClusterCentroids(sampling_strategy={0: int(n_samples * 0.7)}, random_state=42)
                    X_resampled, y_resampled = cluster.fit_resample(X_resampled, y_resampled)
                else:
                    # Para ELF mantener la lógica original
                    smote = SMOTE(sampling_strategy=0.7, random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                
                return X_resampled, y_resampled
            except Exception as e:
                print(f"Error en el balanceo avanzado: {e}")
                return X, y
        return X, y

    def find_optimal_threshold(self, y_true, y_prob, sensor_type=None):
        thresholds = np.linspace(0.1, 0.6, 100)  # Rango ajustado
        best_score = -1
        best_threshold = 0.5  # Valor por defecto
        
        try:
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                recall = recall_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                
                # Diferentes pesos según el tipo de sensor
                if sensor_type == 'mag':
                    combined_score = (0.2 * precision + 0.8 * recall)
                else:
                    combined_score = (0.4 * precision + 0.6 * recall)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_threshold = threshold
            
            return best_threshold
        except Exception as e:
            print(f"Error en find_optimal_threshold: {e}")
            return 0.5  # Valor por defecto en caso de error

    def analyze_feature_importance(self, model, X, feature_names):
        """Analiza la importancia de las características usando SHAP y XGBoost"""
        # Para modelos calibrados, necesitamos acceder al estimador base
        if isinstance(model, CalibratedClassifierCV):
            base_model = model.estimator  # Cambiado de estimator_ a estimator
        else:
            base_model = model

        # XGBoost feature importance
        importance_dict = dict(zip(feature_names, base_model.feature_importances_))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        # SHAP values
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X)
            
            plt.figure(figsize=(15, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig('shap_importance.png')
            plt.close()
        except Exception as e:
            print(f"Error en análisis SHAP: {e}")
        
        return sorted_importance

    def train_model(self, df, sensor_type, size):
        print(f"\n=== ENTRENANDO {sensor_type.upper()} (tamaño {size}) ===")
        print(f"Dimensiones DataFrame: {df.shape}")

        # Hacer una copia del DataFrame y mezclar aleatoriamente
        df_shuffled = df.copy()
        df_shuffled = df_shuffled.sample(frac=1, random_state=42).reset_index(drop=True)

        # Preparar datos
        print("\nPreprocesando datos...")
        X = self.preprocess_data(df_shuffled)
        y = df_shuffled['Label']

        print(f"Características extraídas: {X.shape[1]}")

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        print("\nDistribución de clases:")
        print("Train:", pd.Series(y_train).value_counts(normalize=True))
        print("Test:", pd.Series(y_test).value_counts(normalize=True))

        # Escalado
        scaler = QuantileTransformer(output_distribution='normal')
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[sensor_type][size] = scaler

        # Balanceo
        print("\nBalanceando dataset...")
        X_balanced, y_balanced = self.balance_dataset(X_train_scaled, y_train, method='advanced', sensor_type=sensor_type)

        # Entrenamiento con parámetros optimizados
        print("\nEntrenando modelo...")
        base_model = XGBClassifier(
            max_depth=5 if sensor_type == 'mag' else 3,
            learning_rate=0.02 if sensor_type == 'mag' else 0.05,
            n_estimators=600 if sensor_type == 'mag' else 300,
            min_child_weight=4 if sensor_type == 'mag' else 3,
            subsample=0.85 if sensor_type == 'mag' else 0.8,
            colsample_bytree=0.85 if sensor_type == 'mag' else 0.8,
            scale_pos_weight=10 if sensor_type == 'mag' else 1.5,
            alpha=0.2 if sensor_type == 'mag' else 0.1,
            lambda_=0.6 if sensor_type == 'mag' else 0.5,
            random_state=42,
            tree_method='hist'
        )
        base_model.fit(X_balanced, y_balanced)

        # Validación cruzada específica para MAG
        # Validación cruzada específica para MAG
        if sensor_type == 'mag':
            n_splits = 5
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = []
            
            # Asegurar que trabajamos con arrays numpy
            X_balanced_array = X_balanced if isinstance(X_balanced, np.ndarray) else X_balanced.to_numpy()
            y_balanced_array = y_balanced if isinstance(y_balanced, np.ndarray) else y_balanced.to_numpy()
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_balanced_array, y_balanced_array)):
                X_fold_train = X_balanced_array[train_idx]
                X_fold_val = X_balanced_array[val_idx]
                y_fold_train = y_balanced_array[train_idx]
                y_fold_val = y_balanced_array[val_idx]
                
                base_model.fit(X_fold_train, y_fold_train)
                y_fold_pred = base_model.predict_proba(X_fold_val)[:, 1]
                cv_scores.append(recall_score(y_fold_val, y_fold_pred >= 0.5))
            
            print(f"\nValidación cruzada MAG - Recall promedio: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

        # Análisis de características importantes para MAG
        if sensor_type == 'mag':
            feature_importance = self.analyze_feature_importance(base_model, X_test_scaled, X.columns)
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
            
            print("\nTop 10 características más importantes para MAG:")
            for feature, importance in top_features.items():
                print(f"{feature}: {importance:.4f}")

        # Análisis de características con el modelo base
        print("\nAnalizando importancia de características...")
        feature_importance = self.analyze_feature_importance(base_model, X_test_scaled, X.columns)
        
        # Calibración del modelo
        print("\nCalibrando modelo...")
        calibrated_model = CalibratedClassifierCV(
            base_model, cv=5, method='isotonic'
        )
        calibrated_model.fit(X_balanced, y_balanced)

        # Evaluación
        print("\nEvaluando modelo...")
        y_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        
        # Análisis de características
        print("\nAnalizando importancia de características...")
        feature_importance = self.analyze_feature_importance(calibrated_model, X_test_scaled, X.columns)
        
        # Encontrar umbral óptimo
        threshold = self.find_optimal_threshold(y_test, y_prob, sensor_type)
        self.optimal_thresholds[sensor_type][size] = threshold
        print(f"\nUmbral óptimo: {threshold:.3f}")

        # Predicciones finales
        y_pred = (y_prob >= threshold).astype(int)

        # Métricas detalladas
        print("\n=== MÉTRICAS DE EVALUACIÓN ===")
        print(classification_report(y_test, y_pred))

        # Matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("\nMatriz de confusión:")
        print(f"Verdaderos Negativos: {tn}")
        print(f"Falsos Positivos: {fp}")
        print(f"Falsos Negativos: {fn}")
        print(f"Verdaderos Positivos: {tp}")

        # Métricas adicionales
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"\nPrecision-Recall AUC: {pr_auc:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        
        # Distribución de probabilidades
        print("\nDistribución de probabilidades predichas:")
        print(f"Media: {np.mean(y_prob):.3f}")
        print(f"Mediana: {np.median(y_prob):.3f}")
        print(f"Desv. Est.: {np.std(y_prob):.3f}")

        # Análisis de falsos negativos
        false_negatives = X_test_scaled[(y_test == 1) & (y_pred == 0)]
        if len(false_negatives) > 0:
            print("\nAnalizando falsos negativos...")
            base_explainer = shap.TreeExplainer(base_model)
            shap_values = base_explainer.shap_values(false_negatives)
            plt.figure(figsize=(15, 8))
            shap.summary_plot(shap_values, false_negatives, feature_names=X.columns, show=False)
            plt.title("SHAP Values for False Negatives")
            plt.tight_layout()
            plt.savefig('false_negatives_shap.png')
            plt.close()

        # Métricas detalladas
        print("\n=== MÉTRICAS DE EVALUACIÓN ===")
        print(f"Recall Clase 1: {recall_score(y_test, y_pred):.3f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

        # Guardar modelo
        self.models[sensor_type][size] = calibrated_model

    def save_models(self, base_dir='model'):
        print("\n=== GUARDANDO MODELOS Y CONFIGURACIÓN ===")
        for sensor_type in ['elf', 'mag']:
            for size, model in self.models[sensor_type].items():
                if model is None:
                    continue
                
                folder = os.path.join(base_dir, sensor_type.upper(), str(size))
                os.makedirs(folder, exist_ok=True)

                model_path = os.path.join(folder, f"model_{sensor_type}_{size}.pkl")
                scaler_path = os.path.join(folder, f"scaler_{sensor_type}_{size}.pkl")
                config_path = os.path.join(folder, f"config_{sensor_type}_{size}.pkl")

                joblib.dump(model, model_path)
                joblib.dump(self.scalers[sensor_type][size], scaler_path)
                
                config = {
                    'optimal_threshold': self.optimal_thresholds[sensor_type][size],
                    'feature_names': self.feature_names
                }
                joblib.dump(config, config_path)

                print(f"[OK] {sensor_type.upper()} tamaño {size}:")
                print(f"   Modelo     : {model_path}")
                print(f"   Escalador  : {scaler_path}")
                print(f"   Config     : {config_path}")

    def load_models(self, base_dir='model'):
        print("\nCargando modelos...")
        for sensor_type in ['elf', 'mag']:
            for size in [160, 320, 480]:
                folder = os.path.join(base_dir, sensor_type.upper(), str(size))
                if not os.path.exists(folder):
                    continue

                model_path = os.path.join(folder, f"model_{sensor_type}_{size}.pkl")
                scaler_path = os.path.join(folder, f"scaler_{sensor_type}_{size}.pkl")
                config_path = os.path.join(folder, f"config_{sensor_type}_{size}.pkl")

                if all(os.path.exists(p) for p in [model_path, scaler_path, config_path]):
                    self.models[sensor_type][size] = joblib.load(model_path)
                    self.scalers[sensor_type][size] = joblib.load(scaler_path)
                    config = joblib.load(config_path)
                    self.optimal_thresholds[sensor_type][size] = config['optimal_threshold']
                    self.feature_names = config.get('feature_names')
                    print(f"Cargado {sensor_type.upper()} tamaño {size}")

def main():
    mp = ModelPipeline()
    
    print("Cargando datos...")
    data_dir = 'data'
    
    for sensor_type in ['ELF', 'MAG']:
        for size in [320]:
            csv_path = os.path.join(data_dir, sensor_type, str(size), 
                                  f'passages_export_{sensor_type}.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Shuffle controlado
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                mp.train_model(df, sensor_type.lower(), size)
    
    # Guardar modelos
    mp.save_models()

if __name__ == "__main__":
    main()