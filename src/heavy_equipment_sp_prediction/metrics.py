"""Metrics used by the prediction pipeline."""

from __future__ import annotations

import numpy as np


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSLE for non-negative selling-price predictions."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(actual < 0):
        raise ValueError("y_true must contain non-negative values")
    return float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)))


def rmsle_from_log_target(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Return RMSLE when both arrays are in log1p price space."""
    return rmsle(np.expm1(y_true_log), np.expm1(y_pred_log))
