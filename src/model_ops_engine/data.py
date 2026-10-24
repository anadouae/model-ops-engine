"""Synthetic urban corridor features for congestion-score training."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "vehicle_count",
    "pedestrian_count",
    "avg_speed_kmh",
    "occupancy_ratio",
    "hour_of_day",
    "is_weekend",
    "rain_mm",
]


def generate_corridor_dataset(
    n_samples: int = 800,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simulate municipal traffic sensor aggregates.

    Target ``congestion_score`` ∈ [0, 1]: higher when many vehicles move slowly
    with high lane occupancy (classic urban congestion signal).
    """
    rng = np.random.default_rng(seed)
    vehicle_count = rng.integers(0, 40, size=n_samples)
    pedestrian_count = rng.integers(0, 25, size=n_samples)
    hour = rng.integers(0, 24, size=n_samples)
    is_weekend = rng.integers(0, 2, size=n_samples)
    rain_mm = np.clip(rng.gamma(1.2, 1.5, size=n_samples) - 1.0, 0, 30)

    # Speed drops with density, peak hours, rain
    peak = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
    base_speed = 45.0 - 0.7 * vehicle_count - 4.0 * peak - 0.35 * rain_mm
    avg_speed = np.clip(base_speed + rng.normal(0, 2.5, n_samples), 5.0, 55.0)
    occupancy = np.clip(
        vehicle_count / 40.0 + 0.05 * pedestrian_count / 25.0 + rng.normal(0, 0.03, n_samples),
        0.0,
        1.0,
    )

    # Ground-truth congestion (noisy)
    congestion = (
        0.45 * (1.0 - avg_speed / 55.0)
        + 0.35 * occupancy
        + 0.10 * (vehicle_count / 40.0)
        + 0.05 * (rain_mm / 30.0)
        + 0.05 * peak.astype(float)
        + rng.normal(0, 0.04, n_samples)
    )
    congestion = np.clip(congestion, 0.0, 1.0)

    # Also expose a derived speed prediction target used in README demos
    X = pd.DataFrame(
        {
            "vehicle_count": vehicle_count.astype(float),
            "pedestrian_count": pedestrian_count.astype(float),
            "avg_speed_kmh": avg_speed,
            "occupancy_ratio": occupancy,
            "hour_of_day": hour.astype(float),
            "is_weekend": is_weekend.astype(float),
            "rain_mm": rain_mm,
        }
    )
    y = pd.Series(congestion, name="congestion_score")
    return X, y
