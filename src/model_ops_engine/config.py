"""Runtime configuration (no secrets — use .env.example as template)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "urban-congestion"
    model_name: str = "congestion-gbr"
    model_dir: Path = Path("./artifacts")
    host: str = "0.0.0.0"
    port: int = 8091
    api_title: str = "Model Ops Engine — Urban Congestion"
    # Empty = open local demo; set API_KEY to require X-API-Key on /predict
    api_key: str = ""
    cors_origins: str = "http://127.0.0.1:5178,http://localhost:5178"


settings = Settings()
