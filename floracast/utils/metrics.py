"""
Utility functions for computing forecasting metrics.
"""

import numpy as np
from typing import Union
from darts import TimeSeries


def smape(
    actual: Union[TimeSeries, np.ndarray],
    predicted: Union[TimeSeries, np.ndarray],
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        actual: Actual time series values
        predicted: Predicted time series values

    Returns:
        SMAPE value (lower is better)
    """
    # Convert to numpy arrays if TimeSeries
    if isinstance(actual, TimeSeries):
        actual = actual.values().flatten()
    if isinstance(predicted, TimeSeries):
        predicted = predicted.values().flatten()

    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    # Calculate SMAPE with zero-denominator guard
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    errors = np.zeros_like(denominator, dtype=float)
    non_zero_mask = denominator != 0
    if np.any(non_zero_mask):
        errors[non_zero_mask] = (
            np.abs(actual[non_zero_mask] - predicted[non_zero_mask])
            / denominator[non_zero_mask]
        )
    smape_value = np.mean(errors) * 100.0

    return float(smape_value)
