from __future__ import annotations

"""Compute one-step SARIMA forecasts to use as an expert inside the hybrid.

The model is fit only on the training portion of each target series; test/val
forecasts are produced by appending observed values one step at a time
(refit=False), so no future information leaks into the forecast. The returned
predictions are indexed by timestamp so they can be aligned to the windowed
temporal-group samples.
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover - optional dependency
    ConvergenceWarning = Warning
    SARIMAX = None

from utils.baselines import (
    ARIMA_FALLBACK,
    ARIMA_PRIMARY,
    SARIMA_FALLBACK,
    SARIMA_PRIMARY,
    STAT_HISTORY_WINDOW,
)


def _one_step_series(values: np.ndarray, train_end: int, order, seasonal_order) -> np.ndarray:
    """One-step predictions for the whole series, fit on values[:train_end]."""
    history_start = max(0, train_end - STAT_HISTORY_WINDOW)
    train_values = values[history_start:train_end]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            train_values,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        preds = np.empty(len(values), dtype=float)
        # In-sample one-step predictions over the training region.
        in_sample = fitted.get_prediction(start=0, end=len(train_values) - 1).predicted_mean
        preds[history_start:train_end] = in_sample
        if history_start > 0:
            preds[:history_start] = values[:history_start]

        # Walk-forward one-step predictions for val + test using observed values.
        future = values[train_end:]
        if len(future) > 0:
            appended = fitted.append(future, refit=False)
            preds[train_end:] = appended.get_prediction(
                start=len(train_values), end=len(train_values) + len(future) - 1
            ).predicted_mean

    return preds


def compute_statistical_expert(
    prepared_df: pd.DataFrame,
    target_columns,
    train_end: int,
    model: str = "sarima",
) -> pd.DataFrame:
    """Return a timestamp-indexed frame of one-step forecasts per target.

    Falls back to a seasonal-naive (24h) estimate if statsmodels is unavailable
    or a fit fails, so training never crashes.
    """
    timestamps = pd.to_datetime(prepared_df["timestamp"]).reset_index(drop=True)
    out = pd.DataFrame(index=timestamps)

    if model.lower() == "arima":
        primary, fallback = ARIMA_PRIMARY, ARIMA_FALLBACK
    else:
        primary, fallback = SARIMA_PRIMARY, SARIMA_FALLBACK

    for target in target_columns:
        values = prepared_df[target].to_numpy(dtype=float)
        preds = None
        if SARIMAX is not None:
            for order, seasonal_order in (primary, fallback):
                try:
                    preds = _one_step_series(values, train_end, order, seasonal_order)
                    break
                except Exception:
                    preds = None
        if preds is None:
            # Seasonal-naive fallback (previous day, same hour).
            preds = np.empty(len(values), dtype=float)
            preds[24:] = values[:-24]
            preds[:24] = values[:24]
        out[target] = preds

    return out
