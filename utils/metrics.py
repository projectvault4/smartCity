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


def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    variance = float(np.var(y_true))
    if variance <= 1e-12:
        return 0.0
    return float(rmse(y_true, y_pred) / np.sqrt(variance))


def compute_urban_prediction_score(y_true, y_pred, target_names):
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    normalized_errors = []
    for idx, _ in enumerate(target_names):
        variance = float(np.var(y_true[:, idx]))
        if variance <= 1e-12:
            continue
        normalized_errors.append(float(rmse(y_true[:, idx], y_pred[:, idx]) / np.sqrt(variance)))

    if not normalized_errors:
        return 100.0

    mean_nrmse = float(np.mean(normalized_errors))
    return float(max(0.0, 100.0 * (1.0 - mean_nrmse)))


def compute_all_metrics(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
    }


def compute_metrics_by_target(y_true, y_pred, target_names):
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    return {
        target_name: compute_all_metrics(y_true[:, idx], y_pred[:, idx])
        for idx, target_name in enumerate(target_names)
    }
