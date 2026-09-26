# Heavy Equipment Selling Price Prediction

Production Python package for predicting used heavy-equipment selling prices from transaction-level tabular data.

The original Kaggle analysis remains in [`notebook.ipynb`](notebook.ipynb) as an exploratory record. The supported runtime path is the importable package under [`src/heavy_equipment_sp_prediction`](src/heavy_equipment_sp_prediction), which provides deterministic feature engineering, log-target regression, artifact persistence, and a small CLI.

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

Training data must contain `TargetValue`; the default identifier is `TransactionID`.

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

## Data contract

The pipeline accepts arbitrary additional columns. Numeric columns are retained, date-like columns are decomposed, and other fields are normalized into compact categorical and string-shape features. Missing target values and negative targets are excluded from training. Predictions are clipped to the observed training target range.

## Project layout

```text
src/heavy_equipment_sp_prediction/  Importable production code
tests/                               Unit and integration tests
notebook.ipynb                       Exploratory analysis and ensemble research
pyproject.toml                       Dependencies, CLI, and tool configuration
```
