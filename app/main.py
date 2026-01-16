import os
import joblib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.path.join('models', 'model.pkl')

VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')

app = FastAPI(title='DevOps Exam Mozhogin Sergey', version=VERSION, description='FastAPI сервис для экзаменационного задания по предмету Современные методы DevOps')

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Модель не найдена по пути: {MODEL_PATH}. Сначала обучите ее.')
    return joblib.load(MODEL_PATH)

@app.get('/health')
def health():
    return {'status': 'OK', 'version': VERSION}

@app.post('/predict')
def predict(req: PredictRequest):
    try:
        model = load_model()
        X = [[
            req.sepal_length,
            req.sepal_width,
            req.petal_length,
            req.petal_width
        ]]

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        return {
            'Inference': int(pred),
            'Probabilities': proba.tolist(),
            'Version': VERSION
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))