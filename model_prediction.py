import os
import pandas as pd
import torch
import traceback
from model_pipeline import ModelPipeline

class ModelPrediction:
    def __init__(self, model_path='model'):
        # Load pipeline and models
        self.pipeline = ModelPipeline()
        self.pipeline.load_models(model_path)

    def predict_data(self, data_dir, output_dir='predictions'):
        """
        Process CSV files in prediction folders and generate predictions.
        """
        print("\n=== Starting data prediction ===\n")

        for sensor in ['ELF', 'MAG']:
            print(f"=== Processing predictions for {sensor} ===")
            sensor_path = os.path.join(data_dir, sensor)
            output_sensor_path = os.path.join(output_dir, sensor)

            for size in [160, 320, 480]:
                size_path = os.path.join(sensor_path, str(size))
                output_size_path = os.path.join(output_sensor_path, str(size))

                # Create output folders if they don't exist
                os.makedirs(output_size_path, exist_ok=True)

                if not os.path.exists(size_path):
                    print(f"[SKIP] Folder not found: {size_path}")
                    continue

                csv_files = [f for f in os.listdir(size_path) if f.endswith('.csv')]
                if not csv_files:
                    print(f"[SKIP] No CSV files found in {size_path}")
                    continue

                for file in csv_files:
                    file_path = os.path.join(size_path, file)
                    print(f"-> Processing file: {file_path}")

                    try:
                        data = pd.read_csv(file_path)

                        # Make sure to remove unnecessary columns
                        data = data.iloc[:, 2:]  # Remove first two columns: index and SampleID

                        X, lengths = self.pipeline.preprocess_data(data, is_training=False)
                        predictions = self.predict(X, lengths, sensor.lower(), size)

                        # Save predictions in the corresponding folder
                        output_file = os.path.join(output_size_path, f"{os.path.splitext(file)[0]}_predictions.csv")
                        pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_file, index=False)
                        print(f"✅ Predictions saved to: {output_file}")

                    except Exception as e:
                        print(f"   ❌ Error processing {file_path}: {str(e)}")

    def predict(self, X, lengths, sensor, size):
        """
        Generate predictions for given data using the trained model.
        """
        print(f"-> Generating predictions for {sensor.upper()} size {size}")
        
        # Get corresponding model
        model = self.pipeline.models[sensor].get(size, None)
        if model is None:
            raise ValueError(f"Model not found for {sensor.upper()} size {size}")
        
        # Make predictions
        X_test = X.detach().cpu().numpy()  # Convert tensor to numpy array
        predictions = model.predict(X_test)  # Use Scikit-learn model's predict() method
        
        return predictions


def main():
    # Create ModelPrediction instance
    predictor = ModelPrediction(model_path='model')

    # Run predictions on specified directory
    predictor.predict_data(data_dir='data_predictions', output_dir='predictions')


if __name__ == "__main__":
    main()