"""Tests for training + FastAPI /predict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_ops_engine.data import FEATURE_COLUMNS, generate_corridor_dataset
from model_ops_engine.train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("artifacts")
    mlruns = tmp_path_factory.mktemp("mlruns")
    db = mlruns / "mlflow.db"
    meta = train(
        n_samples=400,
        seed=0,
        tracking_uri=f"sqlite:///{db}",
        model_dir=root,
    )
    assert (root / "congestion_gbr.joblib").exists()
    assert meta["mae"] < 0.15  # synthetic data should fit well
    return {"dir": root, "meta": meta}


def test_dataset_shape():
    X, y = generate_corridor_dataset(n_samples=50, seed=1)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 50
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_train_writes_meta(trained: dict):
    meta = trained["meta"]
    assert "mlflow_run_id" in meta
    assert meta["algorithm"] == "GradientBoostingRegressor"
    assert set(meta["features"]) == set(FEATURE_COLUMNS)


def test_health_and_predict(trained: dict):
    from model_ops_engine.api import create_app

    app = create_app(model_dir=Path(trained["dir"]))
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        info = client.get("/model")
        assert info.status_code == 200
        assert info.json()["features"] == FEATURE_COLUMNS

        payload = {
            "vehicle_count": 28,
            "pedestrian_count": 5,
            "avg_speed_kmh": 18.5,
            "occupancy_ratio": 0.72,
            "hour_of_day": 8,
            "is_weekend": 0,
            "rain_mm": 2.0,
        }
        pred = client.post("/predict", json=payload)
        assert pred.status_code == 200
        body = pred.json()
        assert 0.0 <= body["congestion_score"] <= 1.0
        assert body["model_name"]
        assert body["features_used"] == FEATURE_COLUMNS

        # Free-flow should score lower than rush-hour dense traffic
        free = client.post(
            "/predict",
            json={
                "vehicle_count": 3,
                "pedestrian_count": 1,
                "avg_speed_kmh": 48.0,
                "occupancy_ratio": 0.08,
                "hour_of_day": 14,
                "is_weekend": 1,
                "rain_mm": 0.0,
            },
        ).json()["congestion_score"]
        busy = body["congestion_score"]
        assert busy > free


def test_predict_validation(trained: dict):
    from model_ops_engine.api import create_app

    app = create_app(model_dir=Path(trained["dir"]))
    with TestClient(app) as client:
        bad = client.post("/predict", json={"vehicle_count": -1})
        assert bad.status_code == 422


def test_predict_requires_api_key_when_configured(trained: dict, monkeypatch: pytest.MonkeyPatch):
    from model_ops_engine import config
    from model_ops_engine.api import create_app

    monkeypatch.setattr(config.settings, "api_key", "demo-secret")
    app = create_app(model_dir=Path(trained["dir"]))
    payload = {
        "vehicle_count": 10,
        "pedestrian_count": 1,
        "avg_speed_kmh": 30.0,
        "occupancy_ratio": 0.4,
        "hour_of_day": 12,
        "is_weekend": 0,
        "rain_mm": 0.0,
    }
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 401
        ok = client.post("/predict", json=payload, headers={"X-API-Key": "demo-secret"})
        assert ok.status_code == 200



def test_meta_json_roundtrip(trained: dict):
    meta_path = Path(trained["dir"]) / "model_meta.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["target"] == "congestion_score"
