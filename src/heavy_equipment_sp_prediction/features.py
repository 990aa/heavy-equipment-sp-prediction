"""Leakage-safe, schema-tolerant feature engineering for equipment records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MISSING = "__missing__"
_MISSING_TEXT = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "unspecified",
    "none or unspecified",
}


def normalize_text(value: Any) -> str:
    """Normalize a categorical value into a stable, non-null string."""
    if pd.isna(value):
        return MISSING
    normalized = re.sub(r"\s+", " ", str(value).strip().lower())
    return MISSING if normalized in _MISSING_TEXT else normalized


def parse_number(value: Any) -> float:
    """Extract the first numeric value from a mixed-format field."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def parse_horsepower(value: Any) -> float:
    """Parse horsepower and convert kilowatts to horsepower."""
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", "")
    number = parse_number(text)
    if np.isnan(number):
        return np.nan
    return number * 1.341 if "kw" in text or "kilowatt" in text else number


def parse_length_inches(value: Any) -> float:
    """Parse imperial feet/inches values or a plain numeric length."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)\s*['′]\s*(\d+)\s*[\"″]", str(value))
    if match:
        return float(match.group(1)) * 12 + float(match.group(2))
    return parse_number(value)


@dataclass
class FeatureBuilder:
    """Fit reusable numeric features using training data only."""

    reference_year: int = 2024
    max_categories: int = 500
    feature_columns_: list[str] = field(default_factory=list, init=False)
    medians_: dict[str, float] = field(default_factory=dict, init=False)
    category_maps_: dict[str, dict[str, int]] = field(default_factory=dict, init=False)
    frequencies_: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    active_columns_: list[str] = field(default_factory=list, init=False)

    def _engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        output = pd.DataFrame(index=data.index)

        for column in data.columns:
            if column.lower() in {"targetvalue", "transactionid"}:
                continue
            series = data[column]
            name = column.lower()
            if any(token in name for token in ("date", "timestamp", "datetime")):
                dates = pd.to_datetime(series, errors="coerce")
                if dates.notna().mean() >= 0.2:
                    output[f"{column}_year"] = dates.dt.year
                    output[f"{column}_month"] = dates.dt.month
                    output[f"{column}_quarter"] = dates.dt.quarter
                    output[f"{column}_dow"] = dates.dt.dayofweek
                    output[f"{column}_age"] = self.reference_year - dates.dt.year
                    continue

            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() >= 0.5:
                output[column] = numeric
                if numeric.notna().any() and numeric.min(skipna=True) >= 0:
                    output[f"{column}_log1p"] = np.log1p(numeric)
                continue

            normalized = series.map(normalize_text)
            output[f"{column}_code"] = normalized
            output[f"{column}_length"] = normalized.str.len()
            output[f"{column}_digits"] = normalized.str.count(r"\d")
            output[f"{column}_has_dash"] = normalized.str.contains("-", regex=False).astype(float)
            output[f"{column}_has_slash"] = normalized.str.contains("/", regex=False).astype(float)
            if column == "col9":
                output["Horsepower"] = series.map(parse_horsepower)
            if column == "col22":
                output["BoomLength_inches"] = series.map(parse_length_inches)

        if "ManufactureYear" in output and "TransactionDate_year" in output:
            age = (output["TransactionDate_year"] - output["ManufactureYear"]).clip(lower=0)
            output["AssetAgeAtSale"] = age
            output["AssetAgeAtSale_sq"] = age**2
            output["AssetAgeAtSale_sqrt"] = np.sqrt(age + 1)
        if "OperationalHoursMeter" in output and "AssetAgeAtSale" in output:
            hours = output["OperationalHoursMeter"].clip(lower=0)
            output["HoursPerYear"] = hours / (output["AssetAgeAtSale"] + 1)
            output["HoursPerYear_log"] = np.log1p(output["HoursPerYear"])

        for column in list(output.columns):
            if column.endswith("_code") and self.category_maps_:
                values = output[column].astype(str)
                output[column] = values.map(self.category_maps_.get(column, {})).fillna(-1)
                output[f"{column}_freq"] = values.map(self.frequencies_.get(column, {})).fillna(0)
        return output

    def fit(self, frame: pd.DataFrame) -> FeatureBuilder:
        engineered = self._engineer(frame)
        for column in engineered.columns:
            if engineered[column].dtype == object:
                counts = engineered[column].value_counts()
                keep = counts.nlargest(self.max_categories).index
                values = (
                    engineered[column].where(engineered[column].isin(keep), "__rare__").astype(str)
                )
                self.category_maps_[column] = {
                    value: index for index, value in enumerate(sorted(values.unique()))
                }
                self.frequencies_[column] = values.value_counts(normalize=True).to_dict()
        encoded = self._engineer(frame)
        self.feature_columns_ = list(encoded.columns)
        self.medians_ = {}
        for column in self.feature_columns_:
            numeric = pd.to_numeric(encoded[column], errors="coerce")
            median = numeric.median()
            self.medians_[column] = float(median) if not pd.isna(median) else -1.0
        self.active_columns_ = [
            column for column in self.feature_columns_ if encoded[column].nunique() > 1
        ]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns_:
            raise RuntimeError("FeatureBuilder must be fitted before transform")
        engineered = self._engineer(frame)
        for column in self.feature_columns_:
            if column not in engineered:
                engineered[column] = np.nan
        engineered = engineered[self.feature_columns_].apply(pd.to_numeric, errors="coerce")
        engineered = (
            engineered.replace([np.inf, -np.inf], np.nan).fillna(self.medians_).fillna(-1.0)
        )
        return engineered[self.active_columns_]

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.fit(frame)
        return self.transform(frame)
