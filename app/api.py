"""FastAPI service and web UI.  Run with:  uvicorn app.api:app --reload"""
from __future__ import annotations

from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import pipeline
from .providers import LLMError

app = FastAPI(title="SEC Filings Research Assistant", version="1.0.0")
STATIC_DIR = Path(__file__).parent / "static"


class IngestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, examples=["AAPL"])


class AskRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, examples=["AAPL"])
    question: str = Field(min_length=3, max_length=1000, examples=["What are the main risk factors?"])
    k: int | None = Field(default=None, ge=1, le=15)
    evaluate: bool = True


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/companies")
def companies():
    return pipeline.list_companies()


@app.post("/api/ingest")
def ingest(request: IngestRequest):
    try:
        return pipeline.ingest(request.ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach SEC EDGAR: {exc}")


@app.post("/api/ask")
def ask(request: AskRequest):
    try:
        return pipeline.ask(request.ticker, request.question, k=request.k, evaluate=request.evaluate)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
