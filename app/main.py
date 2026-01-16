import os
import joblib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.path.join('models', 'model.pkl')
model = joblib.load(MODEL_PATH)

app = FastAPI(title='DevOps_Exam_Mozhogin_Sergey')
VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')

class PredictRequest(BaseModel):
    X: List[float]

@app.get('/health')
def health():
    return {'status': 'OK', 'version': VERSION}

@app.post('/predict')
def predict(req: PredictRequest):
    try:
        X = [req.X]
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        return {
            'Inference': int(pred),
            'Probabilities': proba.tolist(),
            'Version': VERSION
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))