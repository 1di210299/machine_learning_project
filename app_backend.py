# app_backend.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import pandas as pd
from model_pipeline import ModelPipeline
from model_prediction import ModelPrediction
from model_testing import ModelTester

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_PATH = '/Users/juandiegogutierrezcortez/machine_learning_project'

@app.post("/predict/{sensor_type}/{size}")
async def predict_endpoint(
    sensor_type: str,
    size: int,
    file: UploadFile = File(...)
):
    try:
        # Guardar archivo en la estructura correcta
        pred_path = os.path.join(BASE_PATH, 'data_predictions', 
                                sensor_type.upper(), str(size), file.filename)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        
        with open(pred_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Usar tu ModelPrediction para las predicciones
        predictor = ModelPrediction(model_path=os.path.join(BASE_PATH, 'model'))
        
        # Leer datos
        df = pd.read_csv(pred_path)
        
        # Realizar predicción usando tu método
        predictions, probabilities = predictor.predict(
            df, 
            sensor_type.lower(), 
            size
        )
        
        # Crear DataFrame con el formato exacto que usas
        sample_ids = df['SampleID'].copy() if 'SampleID' in df.columns else pd.Series(range(len(df)))
        results_df = pd.DataFrame({
            'SampleID': sample_ids,
            'Prediction': predictions,
            'Confidence': probabilities
        })
        
        # Guardar predicciones en la estructura de carpetas correcta
        output_path = os.path.join(
            BASE_PATH, 'predictions',
            sensor_type.upper(), str(size),
            f"{os.path.splitext(file.filename)[0]}_predictions.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        # Preparar resumen para la respuesta
        summary = {
            "total_samples": len(predictions),
            "positive_predictions": int(sum(predictions == 1)),
            "positive_percentage": float(sum(predictions == 1)/len(predictions)*100),
            "mean_confidence": float(probabilities.mean()),
            "min_confidence": float(probabilities.min()),
            "max_confidence": float(probabilities.max())
        }
        
        return {
            "status": "success",
            "predictions": results_df.to_dict('records'),
            "summary": summary,
            "output_file": output_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))