from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.bilstm import EnhancedBiLSTM
from models.hybrid import TFTGRUResidualHybrid
from models.informer import InformerForecastModel
from models.patchtst import PatchTSTForecastModel
from models.transformer import TemporalFusionTransformer
from utils.data_utils import build_temporal_groups_for_inference, create_datasets, load_input_dataframe
from utils.training import predict_model


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _window(length: int, anchor: int) -> tuple[int, ...]:
    return tuple(range(anchor, anchor + length))


def load_project_best_summary(config) -> dict | None:
    summary_path = Path(config.output_dir) / "project_best_summary.json"
    if not summary_path.exists():
        return None
    return _load_json(summary_path)


def load_final_model_registry(config) -> dict | None:
    registry_path = Path(config.output_dir) / "final_model_registry.json"
    if not registry_path.exists():
        return None
    return _load_json(registry_path)


def project_best_model_config(base_config, model_name: str, best_validation: dict):
    config = copy.deepcopy(base_config)

    if model_name == "BiLSTM":
        config.bilstm_hidden_dim = int(best_validation["hidden_dim"])
        config.seq_len = int(best_validation["seq_len"])
        config.batch_size = int(best_validation["batch_size"])
        config.dropout = float(best_validation["dropout"])
        config.learning_rate = float(best_validation["learning_rate"])
        config.closeness_lags = _window(config.seq_len, 1)
        config.period_lags = _window(config.seq_len, 24)
        config.trend_lags = _window(config.seq_len, 24 * 7)
    elif model_name == "TFT":
        config.seq_len = int(best_validation["seq_len"])
        config.tft_hidden_dim = int(best_validation["hidden_dim"])
        config.tft_heads = int(best_validation["heads"])
        config.tft_layers = int(best_validation["layers"])
        config.tft_ff_dim = int(best_validation["ff_dim"])
        config.batch_size = int(best_validation["batch_size"])
        config.dropout = float(best_validation["dropout"])
        config.learning_rate = float(best_validation["learning_rate"])
        config.closeness_lags = _window(config.seq_len, 1)
        config.period_lags = _window(config.seq_len, 24)
        config.trend_lags = _window(config.seq_len, 24 * 7)
    elif model_name == "Hybrid":
        config.closeness_lags = _window(int(best_validation["closeness_len"]), 1)
        config.period_lags = _window(int(best_validation["period_len"]), 24)
        config.trend_lags = _window(int(best_validation["trend_len"]), 24 * 7)
        config.seq_len = max(
            int(best_validation["closeness_len"]),
            int(best_validation["period_len"]),
            int(best_validation["trend_len"]),
        )
        config.bilstm_hidden_dim = int(best_validation["bilstm_hidden_dim"])
        config.tft_hidden_dim = int(best_validation["tft_hidden_dim"])
        config.tft_heads = int(best_validation["tft_heads"])
        config.tft_layers = int(best_validation["tft_layers"])
        config.tft_ff_dim = int(best_validation["tft_ff_dim"])
        config.dense_hidden_dim = int(best_validation["dense_hidden_dim"])
        config.batch_size = int(best_validation["batch_size"])
        config.dropout = float(best_validation["dropout"])
        config.learning_rate = float(best_validation["learning_rate"])
    elif model_name == "Informer":
        config.seq_len = int(best_validation["seq_len"])
        config.informer_d_model = int(best_validation["d_model"])
        config.informer_heads = int(best_validation["heads"])
        config.informer_layers = int(best_validation["layers"])
        config.informer_ff_dim = int(best_validation["ff_dim"])
        config.batch_size = int(best_validation["batch_size"])
        config.dropout = float(best_validation["dropout"])
        config.learning_rate = float(best_validation["learning_rate"])
        config.closeness_lags = _window(config.seq_len, 1)
        config.period_lags = _window(config.seq_len, 24)
        config.trend_lags = _window(config.seq_len, 24 * 7)
    elif model_name == "PatchTST":
        config.seq_len = int(best_validation["seq_len"])
        config.patchtst_d_model = int(best_validation["d_model"])
        config.patchtst_heads = int(best_validation["heads"])
        config.patchtst_layers = int(best_validation["layers"])
        config.patchtst_ff_dim = int(best_validation["ff_dim"])
        config.patchtst_patch_len = int(best_validation["patch_len"])
        config.patchtst_stride = int(best_validation["stride"])
        config.batch_size = int(best_validation["batch_size"])
        config.dropout = float(best_validation["dropout"])
        config.learning_rate = float(best_validation["learning_rate"])
        config.closeness_lags = _window(config.seq_len, 1)
        config.period_lags = _window(config.seq_len, 24)
        config.trend_lags = _window(config.seq_len, 24 * 7)
    else:
        raise ValueError(f"Unsupported project-best model: {model_name}")

    return config


def _build_model(model_name: str, input_dim: int, config):
    output_dim = len(config.target_columns)
    if model_name == "BiLSTM":
        return EnhancedBiLSTM(
            input_dim=input_dim,
            hidden_dim=config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            dropout=config.dropout,
            output_dim=output_dim,
        )
    if model_name == "TFT":
        return TemporalFusionTransformer(
            input_dim=input_dim,
            hidden_dim=config.tft_hidden_dim,
            nhead=config.tft_heads,
            num_layers=config.tft_layers,
            dim_feedforward=config.tft_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        )
    if model_name == "Hybrid":
        return TFTGRUResidualHybrid(input_dim=input_dim, config=config)
    if model_name == "Informer":
        return InformerForecastModel(
            input_dim=input_dim,
            d_model=config.informer_d_model,
            nhead=config.informer_heads,
            num_layers=config.informer_layers,
            dim_feedforward=config.informer_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        )
    if model_name == "PatchTST":
        return PatchTSTForecastModel(
            input_dim=input_dim,
            d_model=config.patchtst_d_model,
            nhead=config.patchtst_heads,
            num_layers=config.patchtst_layers,
            dim_feedforward=config.patchtst_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
            patch_len=config.patchtst_patch_len,
            stride=config.patchtst_stride,
        )
    raise ValueError(f"Unsupported project-best model: {model_name}")


def select_finalized_forecast_model(config) -> str | None:
    registry = load_final_model_registry(config)
    summary = load_project_best_summary(config)
    if not registry or not summary:
        return None
    project_best_names = set(summary["results"].keys())
    for row in registry.get("top_4_models_by_rmse", []):
        model_name = row.get("model")
        if model_name in project_best_names:
            return model_name
    return None


def load_project_best_model(model_name: str, base_config):
    summary = load_project_best_summary(base_config)
    if not summary or model_name not in summary["results"]:
        raise FileNotFoundError("Project-best summary is missing or does not contain the requested model.")

    best_validation = summary["results"][model_name]["best_validation"]
    model_config = project_best_model_config(base_config, model_name, best_validation)
    raw_df = load_input_dataframe(model_config)
    datasets = create_datasets(model_config, raw_df)
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    model = _build_model(model_name, input_dim, model_config)
    checkpoint = Path(summary["results"][model_name]["best_checkpoint"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model, model_config, datasets, raw_df


def latest_project_best_prediction(model_name: str, base_config):
    model, model_config, datasets, raw_df = load_project_best_model(model_name, base_config)
    processor = datasets["processor"]
    prepared = processor.prepare_dataframe(raw_df)
    x_all, _ = processor.transform_dataframe(prepared)
    grouped = build_temporal_groups_for_inference(x_all, prepared["timestamp"], model_config)
    latest_x = {
        "closeness": grouped["closeness"][-1:].astype(np.float32),
        "period": grouped["period"][-1:].astype(np.float32),
        "trend": grouped["trend"][-1:].astype(np.float32),
    }
    pred_scaled = predict_model(model, latest_x, model_config)
    prediction = processor.inverse_transform_targets(pred_scaled)[0]
    next_timestamp = grouped["timestamp"].iloc[-1] + pd.Timedelta(hours=model_config.forecast_horizon)
    return {
        "model_name": model_name,
        "timestamp": next_timestamp,
        "prediction": dict(zip(model_config.target_columns, prediction.tolist())),
    }
