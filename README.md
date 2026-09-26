# Heavy Equipment Selling Price Prediction

Production Python package for predicting used heavy-equipment selling prices from transaction-level tabular data.

## What it does

- Parses dates and mixed-format equipment specifications.
- Creates age, utilization, log, missingness, and categorical-frequency features.
- Trains on `log1p(TargetValue)`, the natural target space for RMSLE.
- Fits preprocessing state only on training data and handles unseen categories at inference time.
- Saves a model artifact containing both the estimator and feature state.
- Produces Kaggle-compatible CSV predictions with `TransactionID` and `TargetValue`.

The package uses scikit-learn's `HistGradientBoostingRegressor` as the dependable default. The notebook's LightGBM/XGBoost ensemble remains available through optional `competition` dependencies for research and comparison.

## Requirements

- Python 3.12-3.14
- [uv](https://docs.astral.sh/uv/)

## Install

```powershell
uv sync --extra dev
```

For the full notebook and competition stack:

```powershell
uv sync --extra dev --extra competition
```

## Train and predict

```powershell
uv run heavy-equipment-sp-prediction train data/train.csv --artifact artifacts/model.joblib
uv run heavy-equipment-sp-prediction predict data/test.csv --artifact artifacts/model.joblib --output artifacts/submission.csv
```

The training command prints a holdout RMSLE and writes the serialized artifact. The prediction command writes a CSV with the identifier and non-negative price prediction.

## Python API

```python
import pandas as pd

from heavy_equipment_sp_prediction.pipeline import predict, train_model

artifact, metrics = train_model(pd.read_csv("data/train.csv"))
predictions = predict(artifact, pd.read_csv("data/test.csv"))
print(metrics["validation_rmsle"])
```

## Quality checks

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests cover domain parsers, unseen categories, RMSLE, training, inference, and artifact round trips. Raw datasets, generated artifacts, and notebook checkpoints are excluded from version control.

## Project layout

```text
src/heavy_equipment_sp_prediction/  Importable production code
tests/                               Unit and integration tests
notebook.ipynb                       Exploratory analysis and ensemble research
pyproject.toml                       Dependencies, CLI, and tool configuration
```
