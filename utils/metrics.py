from __future__ import annotations

import numpy as np


def _as_2d(values):
    values = np.asarray(values)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    percentage_errors = np.abs((y_true - y_pred) / denominator) * 100.0
    if np.isnan(percentage_errors).all():
        return 0.0
    return float(np.nanmean(percentage_errors))


def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    variance = float(np.var(y_true))
    if variance <= 1e-12:
        return 0.0
    return float(rmse(y_true, y_pred) / np.sqrt(variance))


def urban_prediction_score_from_normalized_error(normalized_error: float) -> float:
    if not np.isfinite(normalized_error):
        return 0.0
    normalized_error = max(0.0, float(normalized_error))
    return float(100.0 / (1.0 + normalized_error))


def compute_urban_prediction_score(y_true, y_pred, target_names):
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    if y_true.shape[1] == 0:
        return 100.0

    return urban_prediction_score_from_normalized_error(nrmse(y_true, y_pred))


def compute_all_metrics(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
    }


def compute_metrics_by_target(y_true, y_pred, target_names):
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    metrics_by_target = {}
    for idx, target_name in enumerate(target_names):
        metrics = compute_all_metrics(y_true[:, idx], y_pred[:, idx])
        metrics["UPS"] = urban_prediction_score_from_normalized_error(metrics["NRMSE"])
        metrics_by_target[target_name] = metrics
    return metrics_by_target
