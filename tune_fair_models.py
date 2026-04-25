from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.hybrid import AdaptiveHybridModel
from models.recurrent_baselines import PlainGRU, PlainLSTM
from train import build_models
from utils.baselines import (
    _merge_temporal_inputs,
    _predict_with_fallback,
    ARIMA_FALLBACK,
    ARIMA_PRIMARY,
    SARIMA_FALLBACK,
    SARIMA_PRIMARY,
    STAT_HISTORY_WINDOW,
)
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics, compute_urban_prediction_score
from utils.training import predict_model, train_model


def _dataset_groups(grouped):
    return (
        {key: grouped[key] for key in ("closeness", "period", "trend")},
        grouped["target"],
    )


def _evaluate_sequence_model(model, datasets, config):
    processor = datasets["processor"]
    x_val = _merge_temporal_inputs(datasets["val_tpt"])
    y_val_scaled = datasets["val_tpt"]["target"]
    y_val = processor.inverse_transform_targets(y_val_scaled)
    y_pred_scaled = predict_model(model, x_val, config)
    y_pred = processor.inverse_transform_targets(y_pred_scaled)
    metrics = compute_all_metrics(y_val, y_pred)
    metrics["UPS"] = compute_urban_prediction_score(y_val, y_pred, config.target_columns)
    return metrics


def _evaluate_grouped_model(model, datasets, config):
    processor = datasets["processor"]
    x_val, y_val_scaled = _dataset_groups(datasets["val_tpt"])
    y_val = processor.inverse_transform_targets(y_val_scaled)
    y_pred_scaled = predict_model(model, x_val, config)
    y_pred = processor.inverse_transform_targets(y_pred_scaled)
    metrics = compute_all_metrics(y_val, y_pred)
    metrics["UPS"] = compute_urban_prediction_score(y_val, y_pred, config.target_columns)
    return metrics


def _search_statistical_model(name: str, datasets, config):
    val_df = datasets["val_df"]
    train_df = datasets["train_df"]
    candidate_pairs = (
        [(ARIMA_PRIMARY, ARIMA_FALLBACK), (ARIMA_FALLBACK, ARIMA_PRIMARY)]
        if name == "ARIMA"
        else [(SARIMA_PRIMARY, SARIMA_FALLBACK), (SARIMA_FALLBACK, SARIMA_PRIMARY)]
    )

    best_row = None
    for idx, (primary, fallback) in enumerate(candidate_pairs, start=1):
        preds_by_target = []
        for target_name in config.target_columns:
            train_values = train_df[target_name].to_numpy(dtype=float)[-STAT_HISTORY_WINDOW:]
            val_values = val_df[target_name].to_numpy(dtype=float)
            preds, order, seasonal_order = _predict_with_fallback(train_values, val_values, primary, fallback)
            preds_by_target.append(preds)
        stacked_pred = np.column_stack(preds_by_target)
        y_true = val_df[list(config.target_columns)].to_numpy(dtype=float)
        metrics = compute_all_metrics(y_true, stacked_pred)
        metrics["UPS"] = compute_urban_prediction_score(y_true, stacked_pred, config.target_columns)
        row = {
            "family": "statistical",
            "model": name,
            "trial": idx,
            "config": {"primary": primary, "fallback": fallback},
            **metrics,
        }
        if best_row is None or (row["RMSE"], row["MAE"]) < (best_row["RMSE"], best_row["MAE"]):
            best_row = row
    return best_row


def _search_recurrent_model(name: str, datasets, base_config):
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    output_dim = len(base_config.target_columns)
    model_cls = PlainLSTM if name == "LSTM" else PlainGRU
    candidates = [
        {"hidden_dim": 32, "num_layers": 1, "dropout": 0.1},
        {"hidden_dim": 48, "num_layers": 2, "dropout": 0.2},
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2},
    ]
    best_row = None
    train_x = _merge_temporal_inputs(datasets["train_tpt"])
    train_y = datasets["train_tpt"]["target"]
    val_x = _merge_temporal_inputs(datasets["val_tpt"])
    val_y = datasets["val_tpt"]["target"]

    for idx, candidate in enumerate(candidates, start=1):
        config = copy.deepcopy(base_config)
        config.epochs = min(base_config.epochs, 3)
        config.patience = min(base_config.patience, 1)
        model = model_cls(
            input_dim=input_dim,
            hidden_dim=candidate["hidden_dim"],
            num_layers=candidate["num_layers"],
            dropout=candidate["dropout"],
            output_dim=output_dim,
        )
        checkpoint_dir = Path(config.output_dir) / "fair_tuning_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{name.lower()}_fair_{idx}"
        train_model(
            model=model,
            model_name=run_name,
            train_data=(train_x, train_y),
            val_data=(val_x, val_y),
            config=config,
            checkpoint_dir=checkpoint_dir,
        )
        metrics = _evaluate_sequence_model(model, datasets, config)
        print(
            f"{name} trial {idx}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, UPS={metrics['UPS']:.4f}",
            flush=True,
        )
        row = {
            "family": "recurrent",
            "model": name,
            "trial": idx,
            "config": candidate,
            **metrics,
        }
        if best_row is None or (row["RMSE"], row["MAE"]) < (best_row["RMSE"], best_row["MAE"]):
            best_row = row
    return best_row


def _search_grouped_model(name: str, datasets, base_config):
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    output_dim = len(base_config.target_columns)
    candidates = [
        {"dropout": 0.1, "hidden_scale": 1.0},
        {"dropout": 0.2, "hidden_scale": 1.0},
        {"dropout": 0.2, "hidden_scale": 1.25},
    ]
    best_row = None

    for idx, candidate in enumerate(candidates, start=1):
        config = copy.deepcopy(base_config)
        config.epochs = min(base_config.epochs, 3)
        config.patience = min(base_config.patience, 1)
        config.dropout = candidate["dropout"]
        config.bilstm_hidden_dim = int(round(base_config.bilstm_hidden_dim * candidate["hidden_scale"]))
        config.tft_hidden_dim = int(round(base_config.tft_hidden_dim * candidate["hidden_scale"]))
        config.dense_hidden_dim = int(round(base_config.dense_hidden_dim * candidate["hidden_scale"]))

        if name == "BiLSTM":
            model = build_models(input_dim, config)["BiLSTM"]
        elif name == "TFT":
            model = build_models(input_dim, config)["TFT"]
        elif name == "Hybrid":
            model = AdaptiveHybridModel(input_dim=input_dim, config=config)
        else:
            raise ValueError(f"Unsupported grouped model {name}")

        checkpoint_dir = Path(config.output_dir) / "fair_tuning_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{name.lower()}_fair_{idx}"
        train_model(
            model=model,
            model_name=run_name,
            train_data=_dataset_groups(datasets["train_tpt"]),
            val_data=_dataset_groups(datasets["val_tpt"]),
            config=config,
            checkpoint_dir=checkpoint_dir,
        )
        metrics = _evaluate_grouped_model(model, datasets, config)
        print(
            f"{name} trial {idx}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, UPS={metrics['UPS']:.4f}",
            flush=True,
        )
        row = {
            "family": "grouped_neural",
            "model": name,
            "trial": idx,
            "config": candidate,
            **metrics,
        }
        if best_row is None or (row["RMSE"], row["MAE"]) < (best_row["RMSE"], best_row["MAE"]):
            best_row = row
    return best_row


def build_tuned_model(name: str, datasets, base_config, tuned_entry: dict):
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    output_dim = len(base_config.target_columns)
    config = copy.deepcopy(base_config)
    candidate = tuned_entry["config"]

    if name in {"LSTM", "GRU"}:
        model_cls = PlainLSTM if name == "LSTM" else PlainGRU
        model = model_cls(
            input_dim=input_dim,
            hidden_dim=int(candidate["hidden_dim"]),
            num_layers=int(candidate["num_layers"]),
            dropout=float(candidate["dropout"]),
            output_dim=output_dim,
        )
        return model, config, "sequence"

    if name in {"BiLSTM", "TFT", "Hybrid"}:
        config.dropout = float(candidate["dropout"])
        scale = float(candidate["hidden_scale"])
        config.bilstm_hidden_dim = int(round(base_config.bilstm_hidden_dim * scale))
        config.tft_hidden_dim = int(round(base_config.tft_hidden_dim * scale))
        config.dense_hidden_dim = int(round(base_config.dense_hidden_dim * scale))
        if name == "BiLSTM":
            model = build_models(input_dim, config)["BiLSTM"]
        elif name == "TFT":
            model = build_models(input_dim, config)["TFT"]
        else:
            model = AdaptiveHybridModel(input_dim=input_dim, config=config)
        return model, config, "grouped"

    raise ValueError(f"Unsupported tuned neural model: {name}")


def _save_outputs(best_rows: list[dict], base_config):
    results_df = pd.DataFrame(best_rows)
    results_path = Path(base_config.output_dir) / "fair_tuning_summary.csv"
    results_df.to_csv(results_path, index=False)

    best_config_path = Path(base_config.output_dir) / "fair_tuning_best_configs.json"
    serializable = {
        row["model"]: {
            "family": row["family"],
            "trial": int(row["trial"]),
            "config": row["config"],
            "MAE": float(row["MAE"]),
            "MAPE": float(row["MAPE"]),
            "RMSE": float(row["RMSE"]),
            "NRMSE": float(row["NRMSE"]),
            "UPS": float(row["UPS"]),
        }
        for row in best_rows
    }
    with open(best_config_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    protocol = {
        "description": "Each implemented model is tuned with a modest and comparable validation-driven budget.",
        "validation_selection_rule": "Lowest validation RMSE, with MAE as the secondary tie-breaker.",
        "trials_per_model": {
            "ARIMA": 2,
            "SARIMA": 2,
            "LSTM": 3,
            "GRU": 3,
            "BiLSTM": 3,
            "TFT": 3,
            "Hybrid": 3,
        },
        "epochs_per_neural_trial": 3,
        "early_stopping_patience": 1,
    }
    protocol_path = Path(base_config.output_dir) / "fair_tuning_protocol.json"
    with open(protocol_path, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    return results_df, results_path, best_config_path, protocol_path


def run_fair_tuning(base_config=CONFIG):
    set_seed(base_config.random_seed)
    raw_df = load_input_dataframe(base_config)
    datasets = create_datasets(base_config, raw_df)

    best_rows = []
    best_rows.append(_search_statistical_model("ARIMA", datasets, base_config))
    _save_outputs(best_rows, base_config)
    best_rows.append(_search_statistical_model("SARIMA", datasets, base_config))
    _save_outputs(best_rows, base_config)
    best_rows.append(_search_recurrent_model("LSTM", datasets, base_config))
    _save_outputs(best_rows, base_config)
    best_rows.append(_search_recurrent_model("GRU", datasets, base_config))
    _save_outputs(best_rows, base_config)
    best_rows.append(_search_grouped_model("BiLSTM", datasets, base_config))
    _save_outputs(best_rows, base_config)
    best_rows.append(_search_grouped_model("TFT", datasets, base_config))
    _save_outputs(best_rows, base_config)
    best_rows.append(_search_grouped_model("Hybrid", datasets, base_config))
    return _save_outputs(best_rows, base_config)


def load_best_configs(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    results_df, results_path, best_config_path, protocol_path = run_fair_tuning(CONFIG)
    print(results_df[["model", "family", "trial", "MAE", "RMSE", "NRMSE", "UPS"]].round(4).to_string(index=False))
    print(f"\nSaved fair tuning summary to {results_path}")
    print(f"Saved fair tuning configs to {best_config_path}")
    print(f"Saved fair tuning protocol to {protocol_path}")


if __name__ == "__main__":
    main()
