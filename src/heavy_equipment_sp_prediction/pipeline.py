"""Training and inference services for the selling-price model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .features import FeatureBuilder
from .metrics import rmsle_from_log_target


@dataclass
class ModelArtifact:
    """Serializable model, feature builder, and training metadata."""

    model: HistGradientBoostingRegressor
    features: FeatureBuilder
    target_column: str = "TargetValue"
    id_column: str = "TransactionID"
    target_min: float = 0.0
    target_max: float = 0.0


def train_model(
    train_frame: pd.DataFrame,
    *,
    target_column: str = "TargetValue",
    id_column: str = "TransactionID",
    random_state: int = 42,
) -> tuple[ModelArtifact, dict[str, float]]:
    """Train the production estimator and return it with a holdout score."""
    if target_column not in train_frame:
        raise ValueError(f"Training data must contain {target_column!r}")
    target = pd.to_numeric(train_frame[target_column], errors="coerce")
    valid = target.notna() & (target >= 0)
    raw_features = train_frame.loc[valid].drop(columns=[target_column], errors="ignore")
    y_log = np.log1p(target.loc[valid].to_numpy(dtype=float))
    stratify = pd.qcut(y_log, q=min(10, max(2, len(y_log) // 50)), labels=False, duplicates="drop")
    raw_train, raw_valid, y_train, y_valid = train_test_split(
        raw_features, y_log, test_size=0.2, random_state=random_state, stratify=stratify
    )
    features = FeatureBuilder()
    x_train = features.fit_transform(raw_train)
    x_valid = features.transform(raw_valid)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    score = rmsle_from_log_target(y_valid, model.predict(x_valid))
    artifact = ModelArtifact(
        model=model,
        features=features,
        target_column=target_column,
        id_column=id_column,
        target_min=float(target.min()),
        target_max=float(target.max()),
    )
    return artifact, {
        "validation_rmsle": score,
        "rows": float(valid.sum()),
        "features": float(x_train.shape[1]),
    }


def predict(artifact: ModelArtifact, frame: pd.DataFrame) -> np.ndarray:
    """Predict non-negative selling prices from raw records."""
    values = np.expm1(artifact.model.predict(artifact.features.transform(frame)))
    return np.clip(values, max(1.0, artifact.target_min), artifact.target_max)


def save_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    """Persist a trained artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)


def load_artifact(path: str | Path) -> ModelArtifact:
    """Load a trained artifact from disk."""
    artifact = joblib.load(path)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("Artifact does not contain a heavy-equipment model")
    return artifact
