import numpy as np
import pandas as pd

from heavy_equipment_sp_prediction.metrics import rmsle
from heavy_equipment_sp_prediction.pipeline import (
    load_artifact,
    predict,
    save_artifact,
    train_model,
)


def _sample_data(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(rows),
            "TransactionDate": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "ManufactureYear": rng.integers(1995, 2023, rows),
            "OperationalHoursMeter": rng.integers(0, 5000, rows),
            "RegionCode": rng.choice(["North", "South", "West"], rows),
            "TargetValue": rng.uniform(5_000, 100_000, rows),
        }
    )


def test_training_prediction_and_artifact_round_trip(tmp_path) -> None:
    data = _sample_data()
    artifact, metrics = train_model(data)
    predictions = predict(artifact, data.drop(columns="TargetValue"))
    assert metrics["validation_rmsle"] >= 0
    assert len(predictions) == 80
    assert np.all(predictions >= 1)
    assert rmsle(np.ones(2), np.ones(2)) == 0
    path = tmp_path / "model.joblib"
    save_artifact(artifact, path)
    reloaded = load_artifact(path)
    np.testing.assert_allclose(predictions, predict(reloaded, data.drop(columns="TargetValue")))
