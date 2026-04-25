from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate_models
from tune_fair_models import build_tuned_model, load_best_configs, run_fair_tuning
from utils.baselines import _predict_with_fallback
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score
from utils.training import predict_model, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed reliability using tuned model configurations.")
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated random seeds.")
    return parser.parse_args()


def _dataset_groups(grouped):
    return (
        {key: grouped[key] for key in ("closeness", "period", "trend")},
        grouped["target"],
    )


def _evaluate_statistical_from_tuned_entry(name: str, tuned_entry: dict, datasets, config):
    processor = datasets["processor"]
    y_true = processor.inverse_transform_targets(datasets["test_tpt"]["target"])
    train_df = datasets["train_df"]
    val_df = datasets["val_df"]
    test_df = datasets["test_df"]

    train_plus_val = {
        target_name: np.concatenate(
            [
                train_df[target_name].to_numpy(dtype=float),
                val_df[target_name].to_numpy(dtype=float),
            ]
        )
        for target_name in config.target_columns
    }
    preds_by_target = []
    primary = tuple(tuned_entry["config"]["primary"])
    fallback = tuple(tuned_entry["config"]["fallback"])
    for target_name in config.target_columns:
        preds, _, _ = _predict_with_fallback(
            train_plus_val[target_name],
            test_df[target_name].to_numpy(dtype=float),
            primary,
            fallback,
        )
        preds_by_target.append(preds)
    y_pred = np.column_stack(preds_by_target)
    metrics = compute_all_metrics(y_true, y_pred)
    metrics["UPS"] = compute_urban_prediction_score(y_true, y_pred, config.target_columns)
    per_target = compute_metrics_by_target(y_true, y_pred, config.target_columns)
    return metrics, per_target


def _train_and_evaluate_tuned_neural_model(name: str, tuned_entry: dict, datasets, base_config, seed: int):
    model, config, input_style = build_tuned_model(name, datasets, base_config, tuned_entry)
    config.random_seed = seed
    checkpoint_dir = Path(base_config.output_dir) / f"reliability_tuned_checkpoints_seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{name.lower()}_tuned_seed_{seed}"

    if input_style == "sequence":
        train_x = np.concatenate(
            [datasets["train_tpt"]["closeness"], datasets["train_tpt"]["period"], datasets["train_tpt"]["trend"]],
            axis=1,
        ).astype(np.float32)
        val_x = np.concatenate(
            [datasets["val_tpt"]["closeness"], datasets["val_tpt"]["period"], datasets["val_tpt"]["trend"]],
            axis=1,
        ).astype(np.float32)
        train_data = (train_x, datasets["train_tpt"]["target"])
        val_data = (val_x, datasets["val_tpt"]["target"])
        x_test = np.concatenate(
            [datasets["test_tpt"]["closeness"], datasets["test_tpt"]["period"], datasets["test_tpt"]["trend"]],
            axis=1,
        ).astype(np.float32)
    else:
        train_data = _dataset_groups(datasets["train_tpt"])
        val_data = _dataset_groups(datasets["val_tpt"])
        x_test, _ = _dataset_groups(datasets["test_tpt"])

    train_model(
        model=model,
        model_name=model_name,
        train_data=train_data,
        val_data=val_data,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )
    processor = datasets["processor"]
    y_true = processor.inverse_transform_targets(datasets["test_tpt"]["target"])
    y_pred = processor.inverse_transform_targets(predict_model(model, x_test, config))
    metrics = compute_all_metrics(y_true, y_pred)
    metrics["UPS"] = compute_urban_prediction_score(y_true, y_pred, config.target_columns)
    per_target = compute_metrics_by_target(y_true, y_pred, config.target_columns)
    return metrics, per_target


def main():
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    base_config = copy.deepcopy(CONFIG)
    best_config_path = Path(base_config.output_dir) / "fair_tuning_best_configs.json"
    if not best_config_path.exists():
        print("Fair tuning configs not found. Running fair tuner first.", flush=True)
        run_fair_tuning(base_config)

    tuned_configs = load_best_configs(best_config_path)
    raw_df = load_input_dataframe(base_config)

    rows = []
    per_target_rows = []
    for seed in seeds:
        print(f"Running tuned reliability seed {seed}", flush=True)
        config = copy.deepcopy(base_config)
        config.random_seed = seed
        set_seed(seed)
        datasets = create_datasets(config, raw_df)

        for model_name, tuned_entry in tuned_configs.items():
            if tuned_entry["family"] == "statistical":
                metrics, per_target = _evaluate_statistical_from_tuned_entry(model_name, tuned_entry, datasets, config)
            else:
                metrics, per_target = _train_and_evaluate_tuned_neural_model(model_name, tuned_entry, datasets, config, seed)

            rows.append({"seed": seed, "model": model_name, **metrics})
            for target_name, target_metrics in per_target.items():
                per_target_rows.append({"seed": seed, "model": model_name, "target": target_name, **target_metrics})

    long_df = pd.DataFrame(rows)
    summary_df = long_df.groupby("model")[["MAE", "MAPE", "RMSE", "NRMSE", "UPS"]].agg(["mean", "std"]).reset_index()
    summary_df.columns = ["model"] + [f"{metric}_{stat}" for metric, stat in summary_df.columns.tolist()[1:]]

    per_target_long = pd.DataFrame(per_target_rows)
    per_target_summary = (
        per_target_long.groupby(["model", "target"])[["MAE", "MAPE", "RMSE", "NRMSE"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_target_summary.columns = ["model", "target"] + [f"{metric}_{stat}" for metric, stat in per_target_summary.columns.tolist()[2:]]

    output_dir = Path(base_config.output_dir)
    long_path = output_dir / "reliability_tuned_by_seed.csv"
    summary_path = output_dir / "reliability_tuned_summary.csv"
    per_target_path = output_dir / "reliability_tuned_per_target_summary.csv"
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    per_target_summary.to_csv(per_target_path, index=False)

    protocol = {
        "seeds": seeds,
        "config_source": str(best_config_path),
        "selection_rule": "Configs were chosen from the fair tuning stage using validation RMSE and then retrained/evaluated across multiple seeds.",
    }
    protocol_path = output_dir / "reliability_tuned_protocol.json"
    with open(protocol_path, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)

    print(summary_df.round(4).to_string(index=False))
    print(f"\nSaved reliability runs to {long_path}")
    print(f"Saved reliability summary to {summary_path}")
    print(f"Saved per-target reliability summary to {per_target_path}")


if __name__ == "__main__":
    main()
