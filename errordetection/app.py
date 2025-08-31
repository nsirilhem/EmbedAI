# app.py – Code Error Detection API using FastAPI + Uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

# Chargement des artefacts
model = load_model("model.h5")
vectorizer = joblib.load("vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Initialisation de l'app FastAPI
app = FastAPI(title="Code Error Detection API")

# (optionnel) Permet CORS si besoin (ex: pour une interface JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schéma d’entrée attendu pour les requêtes
class CodeInput(BaseModel):
    code: str

# Helper
def predict_error_type(code_snippet: str) -> str:
    X = vectorizer.transform([code_snippet]).toarray()
    proba = model.predict(X, verbose=0)
    idx = int(np.argmax(proba, axis=1)[0])
    return label_encoder.inverse_transform([idx])[0]

# Routes
@app.get("/")
def health_check():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(input_data: CodeInput):
    if not input_data.code.strip():
        raise HTTPException(status_code=400, detail="Code is empty.")
    error_type = predict_error_type(input_data.code)
    return {"error_type": error_type}

@app.post("/predict_lines")
def predict_lines(input_data: CodeInput):
    lines = input_data.code.splitlines()
    results = []
    for i, line in enumerate(lines, start=1):
        etype = predict_error_type(line)
        if etype != "aucune erreur":
            results.append({"line": i, "error_type": etype, "code": line})
    return {"errors": results}
