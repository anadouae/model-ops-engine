"""FastAPI service: POST /predict for urban congestion score."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_ops_engine import __version__
from model_ops_engine.config import settings
from model_ops_engine.data import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    vehicle_count: float = Field(..., ge=0, description="Vehicles in corridor window")
    pedestrian_count: float = Field(0, ge=0)
    avg_speed_kmh: float = Field(..., ge=0, le=120)
    occupancy_ratio: float = Field(..., ge=0, le=1)
    hour_of_day: float = Field(..., ge=0, le=23)
    is_weekend: float = Field(0, ge=0, le=1)
    rain_mm: float = Field(0, ge=0)


class PredictResponse(BaseModel):
    congestion_score: float
    model_name: str
    model_version: str
    features_used: list[str]


def _load_model(model_dir: Path) -> tuple[Any, dict]:
    path = model_dir / "congestion_gbr.joblib"
    meta_path = model_dir / "model_meta.json"
    if not path.exists():
        from model_ops_engine.train import train

        logger.info("No artifact found — training a fresh model into %s", model_dir)
        meta = train(model_dir=model_dir)
    else:
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    model = joblib.load(path)
    return model, meta


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = (settings.api_key or "").strip()
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key.strip(), expected):
        raise HTTPException(status_code=401, detail="Valid X-API-Key required")


def create_app(model_dir: Path | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        directory = Path(model_dir or settings.model_dir)
        model, meta = _load_model(directory)
        state["model"] = model
        state["meta"] = meta
        if not (settings.api_key or "").strip():
            logger.warning(
                "SECURITY: API_KEY empty — /predict is open (local demo). Set API_KEY for hardened demos."
            )
        else:
            logger.info("SECURITY: X-API-Key required on /predict")
        logger.info("Model ready: %s", meta.get("model_name", settings.model_name))
        yield
        state.clear()

    app = FastAPI(
        title=settings.api_title,
        version=__version__,
        description=(
            "MLOps predict API for Urban Pulse — congestion score from "
            "corridor traffic features. Portfolio project (Douae El Assal)."
        ),
        lifespan=lifespan,
    )

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        meta = state.get("meta") or {}
        return {
            "model_name": meta.get("model_name", settings.model_name),
            "version": meta.get("version", __version__),
            "mae": meta.get("mae"),
            "r2": meta.get("r2"),
            "features": FEATURE_COLUMNS,
            "mlflow_run_id": meta.get("mlflow_run_id"),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        body: PredictRequest,
        _: None = Depends(_require_api_key),
    ) -> PredictResponse:
        import pandas as pd

        model = state.get("model")
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        row = pd.DataFrame([{col: getattr(body, col) for col in FEATURE_COLUMNS}])
        score = float(model.predict(row)[0])
        score = max(0.0, min(1.0, score))
        meta = state.get("meta") or {}
        return PredictResponse(
            congestion_score=round(score, 4),
            model_name=meta.get("model_name", settings.model_name),
            model_version=meta.get("version", __version__),
            features_used=list(FEATURE_COLUMNS),
        )

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "model_ops_engine.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
