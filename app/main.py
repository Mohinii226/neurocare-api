"""NeuroCare migraine-subtype prediction API (FastAPI)."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predictor import ArtifactError, InputError, MigrainePredictor

state = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = os.getenv("ARTIFACT_DIR", "artifacts")
    try:
        state["predictor"] = MigrainePredictor.from_dir(directory)
        state["error"] = None
    except ArtifactError as exc:
        state["predictor"] = None
        state["error"] = str(exc)
    yield
    state["predictor"] = None


app = FastAPI(
    title="NeuroCare API",
    description=(
        "Research prototype: predicts migraine subtype from coded symptom features. "
        "Not a medical device and not a substitute for clinical judgement."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class SymptomRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Feature name -> coded value, using the same codes as the training dataset."
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["predictor"] is not None, "detail": state["error"]}


@app.get("/features")
def features():
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail=state["error"] or "Model not loaded")
    return {"feature_columns": state["predictor"].feature_columns}


@app.post("/predict")
def predict(request: SymptomRequest):
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail=state["error"] or "Model not loaded")
    try:
        return state["predictor"].predict(request.features)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
