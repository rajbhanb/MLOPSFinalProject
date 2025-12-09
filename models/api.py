from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("flaml_balanced_sentiment_pipeline.joblib")

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
