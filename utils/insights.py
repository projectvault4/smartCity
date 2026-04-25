from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from engine.adaptive_ensemble import AdaptiveEnsemble
from train import build_models, checkpoint_name_for_model
from utils.data_utils import build_temporal_groups_for_inference, load_input_dataframe
from utils.training import predict_model


def load_trained_models(datasets, config):
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    models = build_models(input_dim, config)
    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / f"{checkpoint_name_for_model(name)}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {checkpoint}. Run `python3 main.py` to train the models on the current dataset."
            )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return models


def build_adaptive_ensemble(models, datasets, config):
    processor = datasets["processor"]
    val_groups = datasets["val_tpt"]
    x_val = {key: val_groups[key] for key in ("closeness", "period", "trend")}
    y_val_scaled = val_groups["target"]
    y_val = processor.inverse_transform_targets(y_val_scaled)
    predictions = {}
    for name, model in models.items():
        predictions[name] = processor.inverse_transform_targets(predict_model(model, x_val, config))
    ensemble = AdaptiveEnsemble(model_names=list(models.keys()), error_window=config.ensemble_error_window)
    ensemble.update_errors(y_val, predictions)
    ensemble.fit_meta_learner(y_val, predictions)
    return ensemble


def prepare_latest_sequence(raw_df: pd.DataFrame, processor, config):
    prepared = processor.prepare_dataframe(raw_df)
    x_all, _ = processor.transform_dataframe(prepared)
    grouped = build_temporal_groups_for_inference(x_all, prepared["timestamp"], config)
    if len(grouped["closeness"]) == 0:
        raise ValueError("Need enough usable rows to create closeness, period, and trend groups.")
    latest_x = {
        "closeness": grouped["closeness"][-1:].astype(np.float32),
        "period": grouped["period"][-1:].astype(np.float32),
        "trend": grouped["trend"][-1:].astype(np.float32),
    }
    latest_timestamp = grouped["timestamp"].iloc[-1]
    return latest_x, latest_timestamp, prepared


def predict_next_hour(models, ensemble, datasets, config):
    processor = datasets["processor"]
    raw_df = load_input_dataframe(config)
    latest_x, latest_timestamp, prepared = prepare_latest_sequence(raw_df, processor, config)

    predictions = {}
    for name, model in models.items():
        pred_scaled = predict_model(model, latest_x, config)
        predictions[name] = processor.inverse_transform_targets(pred_scaled)

    ensemble_pred = ensemble.predict(predictions)[0]
    next_timestamp = latest_timestamp + pd.Timedelta(hours=config.forecast_horizon)
    target_map = dict(zip(config.target_columns, ensemble_pred.tolist()))
    return target_map, next_timestamp, predictions, prepared


def classify_alerts(prediction: Dict[str, float], recent_df: pd.DataFrame) -> Dict[str, object]:
    alerts: List[str] = []
    status = {}

    traffic_threshold = max(250, float(recent_df["traffic_flow"].quantile(0.85)))
    electricity_threshold = max(400, float(recent_df["electricity_demand"].quantile(0.85)))
    recent_aqi = float(recent_df["aqi"].iloc[-1])
    predicted_aqi = float(prediction["aqi"])
    aqi_change = ((predicted_aqi - recent_aqi) / max(abs(recent_aqi), 1e-6)) * 100

    if prediction["traffic_flow"] > traffic_threshold:
        alerts.append("Traffic congestion expected")
        status["traffic"] = "High"
    else:
        status["traffic"] = "Normal"

    if predicted_aqi > 150 or aqi_change > 20:
        alerts.append("Pollution spike detected")
        status["aqi"] = "Unhealthy"
    else:
        status["aqi"] = "Normal"

    if prediction["electricity_demand"] > electricity_threshold:
        alerts.append("High electricity demand")
        status["electricity"] = "High"
    else:
        status["electricity"] = "Normal"

    return {
        "alerts": alerts,
        "status": status,
        "traffic_threshold": traffic_threshold,
        "electricity_threshold": electricity_threshold,
        "aqi_change_pct": aqi_change,
    }


def build_reasoning(prediction: Dict[str, float], recent_df: pd.DataFrame) -> List[str]:
    reasons = []
    last_row = recent_df.iloc[-1]
    if prediction["traffic_flow"] > recent_df["traffic_flow"].quantile(0.85):
        reasons.append("High traffic flow is pushing AQI upward through cross-domain coupling.")
    if prediction["electricity_demand"] > recent_df["electricity_demand"].quantile(0.85):
        reasons.append("Elevated electricity demand suggests stronger urban activity and added pollution load.")
    if float(last_row["temperature"]) > recent_df["temperature"].median():
        reasons.append("Warmer-than-median temperature supports higher cooling demand in the next hour.")
    if float(last_row["humidity"]) > recent_df["humidity"].median():
        reasons.append("Higher humidity can worsen perceived air quality and align with AQI persistence.")
    if not reasons:
        reasons.append("Recent cross-domain signals are stable, so the forecast remains close to the prevailing regime.")
    return reasons
