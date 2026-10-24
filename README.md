# Model Ops Engine

**MLOps** for urban traffic models in the Urban Pulse ecosystem: train a congestion-score regressor with **MLflow** tracking, serve it via **FastAPI** `POST /predict`.

> Portfolio project — PFE 2027 · ENSA Tétouan · Douae El Assal (`anadouae`).  
> Predicts **traffic congestion** from corridor features (vehicle count, speed, occupancy, weather). No secrets required.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train + log to local MLflow (sqlite:///mlflow.db)
model-ops-train --samples 800

# Serve predict API (auto-trains if no artifact yet)
model-ops-serve
# → http://127.0.0.1:8091/docs
```

```bash
curl -s http://127.0.0.1:8091/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "vehicle_count": 28,
    "pedestrian_count": 5,
    "avg_speed_kmh": 18.5,
    "occupancy_ratio": 0.72,
    "hour_of_day": 8,
    "is_weekend": 0,
    "rain_mm": 2.0
  }'
```

## Tests

```bash
pytest -q
```

## Configuration

Copy [`.env.example`](.env.example) → `.env` (optional). Defaults work offline:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow store |
| `MLFLOW_EXPERIMENT` | `urban-congestion` | Experiment name |
| `MODEL_DIR` | `./artifacts` | joblib + meta JSON |
| `HOST` / `PORT` | `0.0.0.0` / `8091` | API bind |

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness |
| `GET` | `/model` | Model metadata (MAE, R², run id) |
| `POST` | `/predict` | Congestion score ∈ [0, 1] |

## Ecosystem

Feeds Urban Pulse scenario / forecast layers. Upstream detections may come from [Perception Lab](https://github.com/anadouae/perception-lab) (`urban.perception.detections`).

## License

MIT
