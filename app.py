"""FastAPI model-serving application using the same saved model.pkl pipeline."""
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Serves predictions from the single trained Titanic model pipeline.",
    version="1.0.0",
)

class PassengerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pclass: int = Field(..., ge=1, le=3, description="Passenger class: 1, 2, or 3")
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(default=None, ge=0, le=120)
    SibSp: int = Field(..., ge=0, le=20)
    Parch: int = Field(..., ge=0, le=20)
    Fare: Optional[float] = Field(default=None, ge=0)
    Embarked: Optional[Literal["C", "Q", "S"]] = None

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="model.pkl is missing. Run `python train_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok", "model_file_exists": MODEL_PATH.exists()}

@app.post("/predict")
def predict(passenger: PassengerInput):
    model = get_model()
    row = pd.DataFrame([passenger.model_dump()])[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ]
    try:
        result = int(model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    return {
        "prediction": result,
        "label": "Survived" if result == 1 else "Did not survive",
        "model": "model.pkl",
    }
