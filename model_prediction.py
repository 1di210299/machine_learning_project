import os
import pandas as pd
import numpy as np
import traceback
from model_pipeline import ModelPipeline

class ModelPrediction:
    def __init__(self, model_path='model'):
        self.pipeline = ModelPipeline()
        self.pipeline.load_models(model_path)

    def predict(self, df, sensor, size):
        """Realiza predicciones usando el modelo entrenado"""
        model = self.pipeline.models[sensor].get(size)
        if model is None:
            raise ValueError(f"Modelo no encontrado para {sensor.upper()} tamaño {size}")
        
        threshold = self.pipeline.optimal_thresholds[sensor].get(size, 0.5)
        
        # Preprocesar datos usando las mismas funciones del pipeline
        X = self.pipeline.preprocess_data(df)
        
        # Escalar datos
        scaler = self.pipeline.scalers[sensor][size]
        X_scaled = scaler.transform(X)
        
        # Generar predicciones
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Análisis de distribución de probabilidades
        print("\nDistribución de probabilidades:")
        print(f"Media: {probabilities.mean():.3f}")
        print(f"Mediana: {np.median(probabilities):.3f}")
        print(f"Desv. Est.: {probabilities.std():.3f}")
        
        predictions = (probabilities >= threshold).astype(int)
        return predictions, probabilities

    def predict_data(self, data_dir, output_dir='predictions'):
        print("\n=== Iniciando predicción de datos ===\n")
    
        for sensor in ['ELF', 'MAG']:
            print(f"=== Procesando predicciones para {sensor} ===")
            for size in [160, 320, 480]:
                size_path = os.path.join(data_dir, sensor, str(size))
                output_size_path = os.path.join(output_dir, sensor, str(size))
                os.makedirs(output_size_path, exist_ok=True)
                
                if not os.path.exists(size_path):
                    print(f"[OMITIR] Carpeta no encontrada: {size_path}")
                    continue
                
                csv_files = [f for f in os.listdir(size_path) if f.endswith('.csv')]
                if not csv_files:
                    print(f"[OMITIR] No se encontraron archivos CSV en {size_path}")
                    continue
                
                for file in csv_files:
                    file_path = os.path.join(size_path, file)
                    print(f"\n-> Procesando archivo: {file_path}")
                    
                    try:
                        # Leer datos
                        print("   Leyendo archivo CSV...")
                        data = pd.read_csv(file_path)
                        print(f"   Dimensiones del DataFrame: {data.shape}")
                        
                        # Guardar SampleID
                        sample_ids = data['SampleID'].copy() if 'SampleID' in data.columns else pd.Series(range(len(data)))
                        
                        print("   Generando predicciones...")
                        predictions, probabilities = self.predict(
                            data, 
                            sensor.lower(), 
                            size
                        )
                        
                        # Crear DataFrame con resultados
                        results_df = pd.DataFrame({
                            'SampleID': sample_ids,
                            'Prediction': predictions,
                            'Confidence': probabilities
                        })
                        
                        # Guardar predicciones
                        output_file = os.path.join(
                            output_size_path, 
                            f"{os.path.splitext(file)[0]}_predictions.csv"
                        )
                        results_df.to_csv(output_file, index=False)
                        
                        # Mostrar resumen
                        print(f"\n✅ Resumen de predicciones:")
                        print(f"   Total muestras: {len(predictions)}")
                        print(f"   Predicciones positivas: {sum(predictions == 1)} ({sum(predictions == 1)/len(predictions)*100:.2f}%)")
                        print(f"   Confianza media: {probabilities.mean():.2f}")
                        print(f"   Confianza mín/máx: {probabilities.min():.2f}/{probabilities.max():.2f}")
                        print(f"   Archivo guardado: {output_file}\n")
                        
                    except Exception as e:
                        print(f"   ❌ Error procesando {file_path}:")
                        print(traceback.format_exc())

def main():
    predictor = ModelPrediction(model_path='model')
    predictor.predict_data(data_dir='data_predictions', output_dir='predictions')

if __name__ == "__main__":
    main()