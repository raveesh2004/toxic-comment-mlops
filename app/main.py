"""FastAPI service exposing the fine-tuned classifier.

The classifier is a FastAPI dependency (get_classifier), so tests can swap in
a stub without loading the real model.
"""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_DIR = os.getenv("MODEL_DIR", "model")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Toxic Comment Classifier",
    description="DistilBERT fine-tuned for offensive-comment detection, "
    "tracked with MLflow and served via FastAPI.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_classifier():
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_DIR)


class CommentIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class BatchIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class PredictionOut(BaseModel):
    label: str
    score: float


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_dir": MODEL_DIR}


@app.post("/predict", response_model=PredictionOut)
def predict(comment: CommentIn, classifier=Depends(get_classifier)) -> PredictionOut:
    result = classifier(comment.text)[0]
    return PredictionOut(label=result["label"], score=round(result["score"], 4))


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(batch: BatchIn, classifier=Depends(get_classifier)) -> list[PredictionOut]:
    results = classifier(batch.texts)
    return [PredictionOut(label=r["label"], score=round(r["score"], 4)) for r in results]
