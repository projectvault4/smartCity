from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.recurrent_baselines import PlainGRU, PlainLSTM
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score
from utils.training import predict_model, train_model

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover - optional dependency
    ConvergenceWarning = Warning
    SARIMAX = None


ARIMA_PRIMARY = ((2, 1, 1), (0, 0, 0, 0))
ARIMA_FALLBACK = ((1, 0, 0), (0, 0, 0, 0))
SARIMA_PRIMARY = ((1, 0, 1), (1, 0, 0, 24))
SARIMA_FALLBACK = ((1, 0, 0), (1, 0, 0, 24))
STAT_HISTORY_WINDOW = 24 * 90


def _merge_temporal_inputs(grouped: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([grouped["closeness"], grouped["period"], grouped["trend"]], axis=1).astype(np.float32)


def _checkpoint_path(config, name: str) -> Path:
    return Path(config.checkpoint_dir) / f"{name.lower()}.pt"


def _load_or_train_neural_baselines(datasets, config):
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    output_dim = len(config.target_columns)
    baselines = {
        "LSTM": PlainLSTM(
            input_dim=input_dim,
            hidden_dim=config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            dropout=config.dropout,
            output_dim=output_dim,
        ),
        "GRU": PlainGRU(
            input_dim=input_dim,
            hidden_dim=config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            dropout=config.dropout,
            output_dim=output_dim,
        ),
    }

    x_train = _merge_temporal_inputs(datasets["train_tpt"])
    y_train = datasets["train_tpt"]["target"]
    x_val = _merge_temporal_inputs(datasets["val_tpt"])
    y_val = datasets["val_tpt"]["target"]

    for name, model in baselines.items():
        checkpoint = _checkpoint_path(config, name)
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            continue
        train_model(
            model=model,
            model_name=name.lower(),
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return baselines


def _predict_neural_baselines(datasets, config) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    processor = datasets["processor"]
    models = _load_or_train_neural_baselines(datasets, config)
    x_test = _merge_temporal_inputs(datasets["test_tpt"])
    predictions = {}
    metadata = {}
    for name, model in models.items():
        test_scaled = predict_model(model, x_test, config)
        predictions[name] = processor.inverse_transform_targets(test_scaled)
        metadata[name] = {"family": "recurrent neural network", "status": "evaluated"}
    return predictions, metadata


def _one_step_predictions(train_values: np.ndarray, future_values: np.ndarray, order, seasonal_order) -> np.ndarray:
    if SARIMAX is None:
        raise RuntimeError("statsmodels is required for ARIMA and SARIMA baselines.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        fitted = SARIMAX(
            train_values,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        extended = fitted.extend(future_values)
    return np.asarray(extended.fittedvalues, dtype=float)


def _predict_with_fallback(train_values: np.ndarray, test_values: np.ndarray, primary, fallback):
    for order, seasonal_order in (primary, fallback):
        try:
            preds = _one_step_predictions(train_values, test_values, order, seasonal_order)
            return preds, order, seasonal_order
        except Exception:
            continue
    raise RuntimeError("Unable to fit the configured statistical baseline.")


def _predict_statistical_baselines(datasets, config) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    if SARIMAX is None:
        return {}, {}

    test_timestamps = pd.Index(pd.to_datetime(datasets["test_tpt"]["timestamp"]))
    train_df = datasets["train_df"]
    val_df = datasets["val_df"]
    test_df = datasets["test_df"]

    full_test_predictions = {
        "ARIMA": pd.DataFrame(index=test_df["timestamp"], columns=config.target_columns, dtype=float),
        "SARIMA": pd.DataFrame(index=test_df["timestamp"], columns=config.target_columns, dtype=float),
    }
    metadata = {"ARIMA": {"family": "classical statistical"}, "SARIMA": {"family": "classical statistical"}}

    for target_name in config.target_columns:
        train_values = train_df[target_name].to_numpy(dtype=float)
        val_values = val_df[target_name].to_numpy(dtype=float)
        train_plus_val = np.concatenate([train_values, val_values])[-STAT_HISTORY_WINDOW:]
        test_values = test_df[target_name].to_numpy(dtype=float)

        arima_preds, arima_order, arima_seasonal = _predict_with_fallback(
            train_plus_val,
            test_values,
            ARIMA_PRIMARY,
            ARIMA_FALLBACK,
        )
        sarima_preds, sarima_order, sarima_seasonal = _predict_with_fallback(
            train_plus_val,
            test_values,
            SARIMA_PRIMARY,
            SARIMA_FALLBACK,
        )

        full_test_predictions["ARIMA"][target_name] = arima_preds
        full_test_predictions["SARIMA"][target_name] = sarima_preds

        metadata["ARIMA"].setdefault("orders", {})[target_name] = {"order": arima_order, "seasonal_order": arima_seasonal}
        metadata["SARIMA"].setdefault("orders", {})[target_name] = {"order": sarima_order, "seasonal_order": sarima_seasonal}

    filtered_predictions = {}
    for model_name, frame in full_test_predictions.items():
        aligned = frame.loc[test_timestamps]
        filtered_predictions[model_name] = aligned.to_numpy(dtype=float)
        metadata[model_name]["status"] = "evaluated"

    return filtered_predictions, metadata


def evaluate_baselines(datasets, config):
    processor = datasets["processor"]
    y_true = processor.inverse_transform_targets(datasets["test_tpt"]["target"])

    predictions = {}
    metadata = {}

    recurrent_predictions, recurrent_metadata = _predict_neural_baselines(datasets, config)
    predictions.update(recurrent_predictions)
    metadata.update(recurrent_metadata)

    statistical_predictions, statistical_metadata = _predict_statistical_baselines(datasets, config)
    predictions.update(statistical_predictions)
    metadata.update(statistical_metadata)

    if not predictions:
        empty = pd.DataFrame(columns=["MAE", "MAPE", "RMSE", "NRMSE", "UPS"])
        return empty, {}, {}, {}

    metrics = {name: compute_all_metrics(y_true, pred) for name, pred in predictions.items()}
    metrics_df = pd.DataFrame(metrics).T
    metrics_df["UPS"] = [
        compute_urban_prediction_score(y_true, predictions[name], config.target_columns)
        for name in metrics_df.index
    ]
    per_target_metrics = {
        name: compute_metrics_by_target(y_true, pred, config.target_columns)
        for name, pred in predictions.items()
    }
    return metrics_df, per_target_metrics, predictions, metadata
