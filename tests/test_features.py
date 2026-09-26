import numpy as np
import pandas as pd

from heavy_equipment_sp_prediction.features import (
    FeatureBuilder,
    normalize_text,
    parse_horsepower,
    parse_length_inches,
)


def test_parsers_handle_domain_formats() -> None:
    assert normalize_text(" None or unspecified ") == "__missing__"
    assert parse_horsepower("100 kW") == 134.1
    assert parse_length_inches("15' 9\"") == 189.0


def test_feature_builder_is_schema_stable_for_unseen_categories() -> None:
    train = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDate": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "ManufactureYear": [2020, 2018, 2019],
            "OperationalHoursMeter": [100, 200, np.nan],
            "RegionCode": ["North", "South", "North"],
        }
    )
    test = train.iloc[[0]].copy()
    test["RegionCode"] = "Unseen"
    builder = FeatureBuilder()
    train_features = builder.fit_transform(train)
    test_features = builder.transform(test)
    assert list(train_features.columns) == list(test_features.columns)
    assert np.isfinite(test_features.to_numpy()).all()
