from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import pandas as pd
import numpy as np
from model_pipeline import ModelPipeline

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base paths - Ajusta esta ruta a tu directorio
BASE_PATH = '/Users/juandiegogutierrezcortez/machine_learning_project'
DATA_PATH = os.path.join(BASE_PATH, 'data')
DATA_PREDICTIONS_PATH = os.path.join(BASE_PATH, 'data_predictions')
MODEL_PATH = os.path.join(BASE_PATH, 'model')
PREDICTIONS_PATH = os.path.join(BASE_PATH, 'predictions')

# Crear estructura de directorios
for path in [DATA_PATH, DATA_PREDICTIONS_PATH, MODEL_PATH, PREDICTIONS_PATH]:
    os.makedirs(path, exist_ok=True)
    for sensor in ['ELF', 'MAG', 'GEO']:
        for size in [160, 320, 480]:
            os.makedirs(os.path.join(path, sensor, str(size)), exist_ok=True)

# Instanciar el pipeline una vez
pipeline = ModelPipeline()
pipeline.load_models(MODEL_PATH)

@app.post("/upload-training/{sensor_type}/{size}")
async def upload_training_file(sensor_type: str, size: int, file: UploadFile = File(...)):
    try:
        # Guardar archivo en la carpeta correcta
        save_path = os.path.join(DATA_PATH, sensor_type.upper(), str(size), file.filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validar archivo
        df = pd.read_csv(save_path)
        
        return {
            "status": "success",
            "file_path": save_path,
            "rows": len(df)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/{sensor_type}")
async def train_model(sensor_type: str, request: dict):
    try:
        size = request.get("size")
        training_dir = os.path.join(DATA_PATH, sensor_type.upper(), str(size))
        
        print(f"\nTraining model for {sensor_type.upper()}/{size}")
        print(f"Loading data from: {training_dir}")
        
        if not os.path.exists(training_dir):
            raise ValueError(f"Training directory not found: {training_dir}")
        
        # Cargar todos los CSV de la carpeta
        all_data = []
        csv_files = [f for f in os.listdir(training_dir) if f.endswith('.csv')]
        
        print(f"Found {len(csv_files)} CSV files")
        
        for file in csv_files:
            file_path = os.path.join(training_dir, file)
            print(f"Loading: {file}")
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
                print(f"Loaded {len(df)} samples from {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError(f"No valid training data found in {training_dir}")
        
        # Combinar todos los datos
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Total combined samples: {len(combined_data)}")
        
        # Entrenar modelo
        pipeline.train_model(combined_data, sensor_type.lower(), size)
        pipeline.save_models(MODEL_PATH)
        
        return {
            "status": "success",
            "message": f"Model trained for {sensor_type.upper()} size {size}",
            "files_used": len(csv_files),
            "total_samples": len(combined_data),
            "files_processed": [f for f in csv_files]
        }
        
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{sensor_type}/{size}")
async def predict(sensor_type: str, size: int, file: UploadFile = File(...)):
    try:
        print(f"\nIniciando predicción para {sensor_type}/{size}")
        
        # Leer el contenido del archivo
        contents = await file.read()
        print(f"Tamaño del contenido recibido: {len(contents)} bytes")
        
        if len(contents) == 0:
            raise ValueError("El archivo está vacío")
            
        # Crear un StringIO objeto para pandas
        import io
        csv_file = io.StringIO(contents.decode('utf-8'))
        
        # Intentar leer el CSV
        try:
            df = pd.read_csv(csv_file)
            print(f"DataFrame leído exitosamente. Dimensiones: {df.shape}")
        except Exception as e:
            print("Error al leer CSV:", str(e))
            print("Contenido del archivo:", contents[:200])
            raise ValueError(f"Error al leer el CSV: {str(e)}")
        
        # Guardar archivo para predicción
        pred_path = os.path.join(DATA_PREDICTIONS_PATH, sensor_type.upper(), str(size), file.filename)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        
        with open(pred_path, "wb") as buffer:
            buffer.write(contents)
            
        if df.empty:
            raise ValueError("El DataFrame está vacío")
            
        # Validar columnas
        required_columns = [f'Data_{i}' for i in range(size)]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas: {missing_columns}")
            
        # Verificar que existe el modelo
        if not pipeline.models.get(sensor_type.lower(), {}).get(size):
            raise ValueError(f"No hay modelo cargado para {sensor_type}/{size}")
        
        # Preprocesar datos
        print("Preprocesando datos...")
        X = pipeline.preprocess_data(df)
        print(f"Dimensiones después del preprocesamiento: {X.shape}")
        
        # Obtener modelo y scaler
        print("Obteniendo modelo y scaler...")
        model = pipeline.models[sensor_type.lower()][size]
        scaler = pipeline.scalers[sensor_type.lower()][size]
        
        # Escalar datos
        print("Escalando datos...")
        X_scaled = scaler.transform(X)
        
        # Predecir
        print("Generando predicciones...")
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Aplicar umbral óptimo
        threshold = pipeline.optimal_thresholds.get(sensor_type.lower(), {}).get(size, 0.5)
        predictions = (probabilities >= threshold).astype(int)
        
        print(f"Predicciones generadas: {len(predictions)}")
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame({
            'SampleID': df.index if 'SampleID' not in df.columns else df['SampleID'],
            'Prediction': predictions,
            'Confidence': probabilities
        })
        
        # Guardar resultados
        output_path = os.path.join(
            PREDICTIONS_PATH,
            sensor_type.upper(),
            str(size),
            f"pred_{os.path.splitext(file.filename)[0]}.csv"
        )
        
        print(f"Guardando resultados en: {output_path}")
        results_df.to_csv(output_path, index=False)
        
        summary = {
            "total_samples": len(predictions),
            "positive_predictions": int(sum(predictions == 1)),
            "positive_percentage": float(sum(predictions == 1)/len(predictions)*100),
            "mean_confidence": float(np.mean(probabilities)),
            "threshold_used": float(threshold)
        }
        
        print("Predicción completada exitosamente")
        return {
            "status": "success",
            "predictions": results_df.to_dict('records'),
            "summary": summary,
            "output_path": output_path
        }
        
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Limpiar archivo si hubo error
        if 'pred_path' in locals() and os.path.exists(pred_path):
            os.remove(pred_path)
            
        if 'csv_file' in locals():
            csv_file.close()
            
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)