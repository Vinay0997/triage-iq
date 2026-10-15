from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI(title="TriageIQ")
model = joblib.load("models/baseline.joblib")

class Req(BaseModel):
    description: str
    neighborhood: str = "Unknown"
    channel: str = "Web"
    hour: int = 12
    weekday: int = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(r: Req):
    row = pd.DataFrame([{
        "description": r.description.lower(),
        "neighborhood": r.neighborhood,
        "channel": r.channel,
        "hour": str(r.hour),
        "weekday": str(r.weekday),
    }])
    probs = model.predict_proba(row)[0]
    i = probs.argmax()
    return {"category": model.classes_[i], "confidence": round(float(probs[i]), 3)}