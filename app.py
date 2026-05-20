from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")


class PredictRequest(BaseModel):
    instances: list


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):

    X = np.array(req.instances)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    return {
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    }