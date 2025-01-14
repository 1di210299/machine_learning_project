import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats, signal

warnings.filterwarnings('ignore', category=UserWarning)

class ModelPipeline:
    def __init__(self):
        self.models = {'elf': {}, 'mag': {}}
        self.scalers = {'elf': {}, 'mag': {}}
        self.feature_extractors = {'elf': {}, 'mag': {}}
        self.optimal_thresholds = {'elf': {}, 'mag': {}}
        
    def extract_sequence_features(self, data):
        """Extrae características de la secuencia temporal"""
        # Asegurar que no hay NaN
        data = pd.Series(data).interpolate(method='linear', limit_direction='both').values
        
        return {
            'max_value': np.max(data),
            'min_value': np.min(data),
            'peak_to_peak': np.max(data) - np.min(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'zero_crossings': len(np.where(np.diff(np.signbit(data)))[0]),
            'gradient_max': np.max(np.gradient(data)),
            'gradient_min': np.min(np.gradient(data)),
            'peak_value': np.max(np.abs(data)),
        'valley_value': np.min(np.abs(data)),
        'peak_to_valley_ratio': np.max(np.abs(data)) / (np.min(np.abs(data)) + 1e-6),
        'energy_ratio': np.sum(data**2) / len(data)
        }
        

    def segment_features(self, data, segment_size=40):
        """Extrae características por segmentos"""
        # Asegurar que no hay NaN
        data = pd.Series(data).interpolate(method='linear', limit_direction='both').values
        
        segments = np.array_split(data, len(data)//segment_size)
        features = {}
        for i, segment in enumerate(segments):
            features.update({
                f'segment_{i}_mean': np.mean(segment),
                f'segment_{i}_std': np.std(segment),
                f'segment_{i}_max': np.max(segment),
                f'segment_{i}_min': np.min(segment),
                f'segment_{i}_peak_to_peak': np.max(segment) - np.min(segment)
            })
        return features

    def preprocess_data(self, df):
        """Preprocesa los datos extrayendo características y maneja valores NaN"""
        feature_list = []
        
        # Primero, limpiar NaN en el DataFrame completo
        df_cleaned = df.fillna(method='ffill').fillna(method='bfill')
        
        for _, row in df_cleaned.iterrows():
            data = row[[f'Data_{i}' for i in range(320)]].values
            
            features = {}
            try:
                features.update(self.extract_sequence_features(data))
                features.update(self.segment_features(data))
            except Exception as e:
                print(f"Error procesando fila: {e}")
                # Valores por defecto en caso de error
                features = {
                    'max_value': 0.0,
                    'min_value': 0.0,
                    'peak_to_peak': 0.0,
                    'mean': 0.0,
                    'std': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0,
                    'zero_crossings': 0,
                    'gradient_max': 0.0,
                    'gradient_min': 0.0
                }
                for i in range(8):
                    features.update({
                        f'segment_{i}_mean': 0.0,
                        f'segment_{i}_std': 0.0,
                        f'segment_{i}_max': 0.0,
                        f'segment_{i}_min': 0.0,
                        f'segment_{i}_peak_to_peak': 0.0
                    })
            
            feature_list.append(features)
        
        # Crear DataFrame y manejar cualquier NaN restante
        features_df = pd.DataFrame(feature_list)
        features_df = features_df.fillna(features_df.median())
        
        # Verificación final
        if features_df.isna().any().any():
            print("Advertencia: Rellenando valores NaN restantes con 0")
            features_df = features_df.fillna(0)
        
        return features_df

    def balance_dataset(self, X, y, method='hybrid'):
        """Balance mejorado del dataset con manejo de NaN"""
        # Verificar y limpiar NaN
        if isinstance(X, np.ndarray) and np.isnan(X).any():
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
        
        if method == 'hybrid':
            try:
                over = SMOTE(sampling_strategy=0.7, random_state=42)
                under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
                pipeline = ImbPipeline([('over', over), ('under', under)])
                X_resampled, y_resampled = pipeline.fit_resample(X, y)
                return X_resampled, y_resampled
            except Exception as e:
                print(f"Error en el balanceo: {e}")
                return X, y
        return X, y

    def find_optimal_threshold(self, y_true, y_prob):
        """Encuentra el umbral óptimo usando F1-score"""
        thresholds = np.linspace(0.01, 0.9, 100)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
            
        return thresholds[np.argmax(f1_scores)]

    def train_model(self, df, sensor_type, size):
        print(f"\n=== ENTRENANDO {sensor_type.upper()} (tamaño {size}) ===")
        print(f"Dimensiones DataFrame: {df.shape}")

        # Verificar y limpiar NaN en datos originales
        if df.isna().any().any():
            print("Limpiando valores NaN en datos de entrada...")
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Preparar datos
        print("\nPreprocesando datos...")
        X = self.preprocess_data(df)
        y = df['Label']

        print(f"Características extraídas: {X.shape[1]}")

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        print("\nDistribución de clases:")
        print("Train:", pd.Series(y_train).value_counts(normalize=True))
        print("Test:", pd.Series(y_test).value_counts(normalize=True))

        # Escalado
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[sensor_type][size] = scaler

        # Balanceo
        print("\nBalanceando dataset...")
        X_balanced, y_balanced = self.balance_dataset(X_train_scaled, y_train)

        # Entrenamiento
        print("\nEntrenando modelo...")
        model = XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=200,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            random_state=42
        )

        # Calibración
        print("Calibrando modelo...")
        calibrated_model = CalibratedClassifierCV(
            model, cv=5, method='sigmoid'
        )
        calibrated_model.fit(X_balanced, y_balanced)

        # Evaluación
        print("\nEvaluando modelo...")
        y_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        
        # Encontrar umbral óptimo
        threshold = self.find_optimal_threshold(y_test, y_prob)
        self.optimal_thresholds[sensor_type][size] = threshold
        print(f"\nUmbral óptimo: {threshold:.3f}")

        # Predicciones finales
        y_pred = (y_prob >= threshold).astype(int)

        # Métricas
        print("\n=== MÉTRICAS DE EVALUACIÓN ===")
        print(classification_report(y_test, y_pred))

        # Matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("\nMatriz de confusión:")
        print(f"Verdaderos Negativos: {tn}")
        print(f"Falsos Positivos: {fp}")
        print(f"Falsos Negativos: {fn}")
        print(f"Verdaderos Positivos: {tp}")

        # ROC y PR curves
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        print(f"\nPrecision-Recall AUC: {auc(recall, precision):.3f}")
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

                # Guardar modelo y configuración
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[sensor_type][size], scaler_path)
                
                config = {
                    'optimal_threshold': self.optimal_thresholds[sensor_type][size]
                }
                joblib.dump(config, config_path)

                print(f"[OK] {sensor_type.upper()} tamaño {size}:")
                print(f"   Modelo     : {model_path}")
                print(f"   Escalador  : {scaler_path}")
                print(f"   Config     : {config_path}")

        print("=== GUARDADO COMPLETO ===\n")

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
                    print(f"Cargado {sensor_type.upper()} tamaño {size}")

def main():
    mp = ModelPipeline()
    
    # Cargar datos
    print("Cargando datos...")
    data_dir = 'data'
    
    for sensor_type in ['ELF', 'MAG']:
        for size in [320]:  # Solo procesamos tamaño 320 según los datos disponibles
            csv_path = os.path.join(data_dir, sensor_type, str(size), 
                                  f'passages_export_{sensor_type}.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                mp.train_model(df, sensor_type.lower(), size)
    
    # Guardar modelos
    mp.save_models()

if __name__ == "__main__":
    main()