"""Train GradientBoosting congestion model and log to MLflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from model_ops_engine.config import settings
from model_ops_engine.data import FEATURE_COLUMNS, generate_corridor_dataset

logger = logging.getLogger(__name__)


def train(
    *,
    n_samples: int = 800,
    seed: int = 42,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    model_dir: Path | None = None,
) -> dict:
    """Fit GBR, log metrics/params/model to MLflow, persist artifact locally."""
    tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    experiment = experiment or settings.mlflow_experiment
    model_dir = Path(model_dir or settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_corridor_dataset(n_samples=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X[FEATURE_COLUMNS], y, test_size=0.2, random_state=seed
    )

    model = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        random_state=seed,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="congestion-gbr") as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_params(
            {
                "n_estimators": 120,
                "max_depth": 3,
                "learning_rate": 0.08,
                "n_samples": n_samples,
                "features": ",".join(FEATURE_COLUMNS),
            }
        )
        mlflow.log_metrics({"mae": mae, "r2": r2})
        # pickle: we train this model ourselves; skops needs extra trust for tree models
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )

        artifact_path = model_dir / "congestion_gbr.joblib"
        meta_path = model_dir / "model_meta.json"
        joblib.dump(model, artifact_path)
        meta = {
            "model_name": settings.model_name,
            "version": "0.1.0",
            "algorithm": "GradientBoostingRegressor",
            "features": FEATURE_COLUMNS,
            "mae": mae,
            "r2": r2,
            "mlflow_run_id": run.info.run_id,
            "target": "congestion_score",
            "sample_predict": float(np.mean(preds)),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(meta_path))

        logger.info("MAE=%.4f R2=%.4f run_id=%s", mae, r2, run.info.run_id)
        return meta


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train urban congestion model")
    p.add_argument("--samples", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--model-dir", default=None)
    args = p.parse_args(argv)
    meta = train(
        n_samples=args.samples,
        seed=args.seed,
        tracking_uri=args.tracking_uri,
        model_dir=Path(args.model_dir) if args.model_dir else None,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
