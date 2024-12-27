import os
import pandas as pd
import torch
import traceback
from model_pipeline import ModelPipeline

class ModelPrediction:
    def __init__(self, model_path='model'):
        # Cargar el pipeline y los modelos
        self.pipeline = ModelPipeline()
        self.pipeline.load_models(model_path)

    def predict_data(self, data_dir, output_dir='predictions'):
        """
        Procesa los archivos CSV en las carpetas de predicción y genera predicciones.
        """
        print("\n=== Iniciando predicción de datos ===\n")

        for sensor in ['ELF', 'MAG']:
            print(f"=== Procesando predicciones para {sensor} ===")
            sensor_path = os.path.join(data_dir, sensor)
            output_sensor_path = os.path.join(output_dir, sensor)

            for size in [160, 320, 480]:
                size_path = os.path.join(sensor_path, str(size))
                output_size_path = os.path.join(output_sensor_path, str(size))

                # Crear carpetas de salida si no existen
                os.makedirs(output_size_path, exist_ok=True)

                if not os.path.exists(size_path):
                    print(f"[SKIP] Carpeta no encontrada: {size_path}")
                    continue

                csv_files = [f for f in os.listdir(size_path) if f.endswith('.csv')]
                if not csv_files:
                    print(f"[SKIP] No hay archivos CSV en {size_path}")
                    continue

                for file in csv_files:
                    file_path = os.path.join(size_path, file)
                    print(f"-> Procesando archivo: {file_path}")

                    try:
                        data = pd.read_csv(file_path)

                        # Asegúrate de eliminar columnas no necesarias
                        data = data.iloc[:, 2:]  # Elimina las dos primeras columnas: índice y SampleID

                        X, lengths = self.pipeline.preprocess_data(data, is_training=False)
                        predictions = self.predict(X, lengths, sensor.lower(), size)

                        # Guardar las predicciones en la carpeta correspondiente
                        output_file = os.path.join(output_size_path, f"{os.path.splitext(file)[0]}_predictions.csv")
                        pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_file, index=False)
                        print(f"✅ Predicciones guardadas en: {output_file}")

                    except Exception as e:
                        print(f"   ❌ Error procesando {file_path}: {str(e)}")


    def predict(self, X, lengths, sensor, size):
        """
        Genera predicciones para los datos dados utilizando el modelo entrenado.
        """
        print(f"-> Generando predicciones para {sensor.upper()} tamaño {size}")
        
        # Obtener el modelo correspondiente
        model = self.pipeline.models[sensor].get(size, None)
        if model is None:
            raise ValueError(f"No se encontró el modelo para {sensor.upper()} tamaño {size}")
        
        # Realizar predicciones
        X_test = X.detach().cpu().numpy()  # Convertir el tensor a un arreglo numpy
        predictions = model.predict(X_test)  # Usar el método predict() del modelo de Scikit-learn
        
        return predictions


def main():
    # Crear una instancia de ModelPrediction
    predictor = ModelPrediction(model_path='model')

    # Ejecutar predicciones en el directorio especificado
    predictor.predict_data(data_dir='data_predictions', output_dir='predictions')


if __name__ == "__main__":
    main()
