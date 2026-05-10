from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch

from models.bilstm import EnhancedBiLSTM
from models.hybrid import TFTGRUResidualHybrid
from models.transformer import TemporalFusionTransformer
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score
from utils.training import predict_model, train_model


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _window(length: int, anchor: int) -> tuple[int, ...]:
    return tuple(range(anchor, anchor + length))


def _dataset_groups(grouped):
    return (
        {key: grouped[key] for key in ("closeness", "period", "trend")},
        grouped["target"],
    )


def _model_config(base_config, model_name: str, best_validation: dict, seed: int):
    config = copy.deepcopy(base_config)
    config.random_seed = seed
    config.epochs = 16
    config.patience = 5

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
    else:
        raise ValueError(f"Unsupported model for reliability: {model_name}")

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
    raise ValueError(f"Unsupported model for reliability: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed reliability for project-best core models.")
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated random seeds.")
    args = parser.parse_args()

    summary_path = Path(CONFIG.output_dir) / "project_best_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Missing outputs/project_best_summary.json. Finish tuning first.")

    project_best_summary = _load_json(summary_path)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    raw_df = load_input_dataframe(CONFIG)

    rows = []
    per_target_rows = []

    for model_name, payload in project_best_summary["results"].items():
        best_validation = payload["best_validation"]
        for seed in seeds:
            config = _model_config(CONFIG, model_name, best_validation, seed)
            set_seed(seed)
            datasets = create_datasets(config, raw_df)
            input_dim = datasets["train_tpt"]["closeness"].shape[-1]
            model = _build_model(model_name, input_dim, config)
            checkpoint_dir = Path(CONFIG.output_dir) / f"reliability_project_best_seed_{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            train_model(
                model=model,
                model_name=f"{model_name.lower()}_seed_{seed}",
                train_data=_dataset_groups(datasets["train_tpt"]),
                val_data=_dataset_groups(datasets["val_tpt"]),
                config=config,
                checkpoint_dir=checkpoint_dir,
            )
            x_test, y_test_scaled = _dataset_groups(datasets["test_tpt"])
            preds_scaled = predict_model(model, x_test, config)
            processor = datasets["processor"]
            y_true = processor.inverse_transform_targets(y_test_scaled)
            preds = processor.inverse_transform_targets(preds_scaled)

            metrics = compute_all_metrics(y_true, preds)
            metrics["UPS"] = compute_urban_prediction_score(y_true, preds, config.target_columns)
            rows.append({"model": model_name, "seed": seed, **metrics})

            per_target = compute_metrics_by_target(y_true, preds, config.target_columns)
            for target_name, values in per_target.items():
                per_target_rows.append({"model": model_name, "seed": seed, "target": target_name, **values})

    by_seed_df = pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)
    summary_df = (
        by_seed_df.groupby("model")[["MAE", "MAPE", "RMSE", "NRMSE", "UPS"]]
        .agg(["mean", "std"])
        .round(6)
    )
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
    summary_df = summary_df.reset_index()

    per_target_df = pd.DataFrame(per_target_rows).sort_values(["model", "target", "seed"]).reset_index(drop=True)
    per_target_summary_df = (
        per_target_df.groupby(["model", "target"])[["MAE", "MAPE", "RMSE", "NRMSE"]]
        .agg(["mean", "std"])
        .round(6)
    )
    per_target_summary_df.columns = [f"{metric}_{stat}" for metric, stat in per_target_summary_df.columns]
    per_target_summary_df = per_target_summary_df.reset_index()

    output_dir = Path(CONFIG.output_dir)
    by_seed_df.to_csv(output_dir / "project_best_reliability_by_seed.csv", index=False)
    summary_df.to_csv(output_dir / "project_best_reliability_summary.csv", index=False)
    per_target_summary_df.to_csv(output_dir / "project_best_reliability_per_target_summary.csv", index=False)

    protocol = {
        "description": "Multi-seed reliability study for finalized project-best neural models.",
        "seeds": seeds,
        "models": list(project_best_summary["results"].keys()),
    }
    with open(output_dir / "project_best_reliability_protocol.json", "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)

    print("\nProject-Best Reliability Summary")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
