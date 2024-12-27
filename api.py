from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from model_pipeline import ModelPipeline
import joblib
import os
import io
from datetime import datetime
import json

app = FastAPI(title="Model Prediction API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    data: List[float]
    sensor_type: str

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    timestamp: str

class BatchPredictionInput(BaseModel):
    data: List[List[float]]
    sensor_type: str

class ModelInfo(BaseModel):
    sensor_type: str
    last_trained: str
    accuracy: float
    total_samples: int
    version: str

# Inicializar pipeline globalmente
pipeline = ModelPipeline()

@app.get("/")
async def root():
    return {"message": "Model Prediction API v1.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": bool(pipeline.models)
    }

@app.get("/model_info/{sensor_type}", response_model=ModelInfo)
async def get_model_info(sensor_type: str):
    try:
        # Cargar información del modelo
        info_path = os.path.join('model', sensor_type.upper(), 'model_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            return info
        else:
            raise HTTPException(status_code=404, detail=f"No info found for {sensor_type} model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Validar tipo de sensor
        sensor_type = input_data.sensor_type.upper()
        if sensor_type not in ['ELF', 'MAG', 'GEO']:
            raise HTTPException(status_code=400, detail="Invalid sensor type")

        # Cargar modelo si no está cargado
        if not pipeline.models.get(sensor_type.lower()):
            pipeline.load_models()

        # Preparar datos
        data = pd.DataFrame([input_data.data])
        processed_data = pipeline.preprocess_data(data, is_training=False)

        # Realizar predicción
        model = pipeline.models[sensor_type.lower()]
        prediction = model.predict(processed_data)[0]
        probability = float(model.predict_proba(processed_data)[0].max())

        return {
            "prediction": int(prediction),
            "probability": probability,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(input_data: BatchPredictionInput):
    try:
        # Validar tipo de sensor
        sensor_type = input_data.sensor_type.upper()
        if sensor_type not in ['ELF', 'MAG', 'GEO']:
            raise HTTPException(status_code=400, detail="Invalid sensor type")

        # Cargar modelo si no está cargado
        if not pipeline.models.get(sensor_type.lower()):
            pipeline.load_models()

        # Preparar datos
        data = pd.DataFrame(input_data.data)
        processed_data = pipeline.preprocess_data(data, is_training=False)

        # Realizar predicciones
        model = pipeline.models[sensor_type.lower()]
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data).max(axis=1)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_data")
async def upload_data(
    file: UploadFile = File(...),
    sensor_type: str = None
):
    try:
        if not sensor_type or sensor_type.upper() not in ['ELF', 'MAG', 'GEO']:
            raise HTTPException(status_code=400, detail="Invalid sensor type")

        # Leer el archivo
        content = await file.read()
        data = pd.read_csv(io.BytesIO(content))

        # Guardar en el directorio correspondiente
        save_path = os.path.join('data', sensor_type.upper(), file.filename)
        data.to_csv(save_path, index=False)

        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "rows": len(data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/{sensor_type}")
async def retrain_model(sensor_type: str):
    try:
        sensor_type = sensor_type.upper()
        if sensor_type not in ['ELF', 'MAG', 'GEO']:
            raise HTTPException(status_code=400, detail="Invalid sensor type")

        # Reentrenar modelo
        pipeline = ModelPipeline()
        success = pipeline.train_model(sensor_type.lower())

        if success:
            return {"message": f"Model {sensor_type} retrained successfully"}
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/{sensor_type}")
async def get_metrics(sensor_type: str):
    try:
        sensor_type = sensor_type.upper()
        if sensor_type not in ['ELF', 'MAG', 'GEO']:
            raise HTTPException(status_code=400, detail="Invalid sensor type")

        # Leer métricas del archivo
        metrics_path = os.path.join('plots', sensor_type, f'metrics_{sensor_type}.txt')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = f.read()
            return {"metrics": metrics}
        else:
            raise HTTPException(status_code=404, detail=f"No metrics found for {sensor_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)