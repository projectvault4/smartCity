from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from evaluate import evaluate_models
from train import build_models
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score
from utils.training import predict_model


def _dataset_groups(grouped):
    return (
        {key: grouped[key] for key in ("closeness", "period", "trend")},
        grouped["target"],
    )


def _load_trained_models(datasets, config):
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    models = build_models(input_dim, config)
    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / f"{name.lower()}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint {checkpoint}. Run `python3 main.py` first.")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return models


def _build_long_table(metric_map: dict, split_name: str) -> pd.DataFrame:
    rows = []
    for model_name, target_metrics in metric_map.items():
        for target_name, values in target_metrics.items():
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "target": target_name,
                    "MAE": float(values["MAE"]),
                    "RMSE": float(values["RMSE"]),
                    "NRMSE": float(values["NRMSE"]),
                }
            )
    return pd.DataFrame(rows)


def run_per_target_tuning(config=CONFIG):
    set_seed(config.random_seed)
    raw_df = load_input_dataframe(config)
    datasets = create_datasets(config, raw_df)
    processor = datasets["processor"]
    models = _load_trained_models(datasets, config)

    metrics_df, per_target_test_metrics, test_predictions, _ = evaluate_models(models, datasets, config)

    val_groups = datasets["val_tpt"]
    test_groups = datasets["test_tpt"]
    x_val, y_val_scaled = _dataset_groups(val_groups)
    x_test, y_test_scaled = _dataset_groups(test_groups)
    y_val = processor.inverse_transform_targets(y_val_scaled)
    y_test = processor.inverse_transform_targets(y_test_scaled)

    base_val_predictions = {}
    base_test_predictions = {}
    for name, model in models.items():
        val_scaled = predict_model(model, x_val, config)
        test_scaled = predict_model(model, x_test, config)
        base_val_predictions[name] = processor.inverse_transform_targets(val_scaled)
        base_test_predictions[name] = processor.inverse_transform_targets(test_scaled)

    per_target_val_metrics = {
        name: compute_metrics_by_target(y_val, pred, config.target_columns)
        for name, pred in base_val_predictions.items()
    }

    best_by_target = {}
    for target_name in config.target_columns:
        best_model = min(
            per_target_val_metrics,
            key=lambda model_name: per_target_val_metrics[model_name][target_name]["NRMSE"],
        )
        best_by_target[target_name] = {
            "model": best_model,
            **{
                metric_name: float(metric_value)
                for metric_name, metric_value in per_target_val_metrics[best_model][target_name].items()
            },
        }

    tuned_test_prediction = y_test.copy()
    for target_idx, target_name in enumerate(config.target_columns):
        model_name = best_by_target[target_name]["model"]
        tuned_test_prediction[:, target_idx] = base_test_predictions[model_name][:, target_idx]

    tuned_metrics = compute_all_metrics(y_test, tuned_test_prediction)
    tuned_metrics["UPS"] = compute_urban_prediction_score(y_test, tuned_test_prediction, config.target_columns)
    tuned_per_target_metrics = compute_metrics_by_target(y_test, tuned_test_prediction, config.target_columns)

    val_table = _build_long_table(per_target_val_metrics, "validation")
    test_table = _build_long_table(per_target_test_metrics, "test")
    combined_table = pd.concat([val_table, test_table], ignore_index=True)

    combined_path = Path(config.output_dir) / "per_target_metrics.csv"
    combined_table.to_csv(combined_path, index=False)

    best_path = Path(config.output_dir) / "best_model_per_target.json"
    with open(best_path, "w", encoding="utf-8") as handle:
        json.dump(best_by_target, handle, indent=2)

    tuned_summary = {
        "selected_models": {target: values["model"] for target, values in best_by_target.items()},
        "validation_best_models": best_by_target,
        "test_metrics_overall": tuned_metrics,
        "test_metrics_by_target": tuned_per_target_metrics,
        "existing_offline_metrics": metrics_df.to_dict(orient="index"),
    }
    tuned_path = Path(config.output_dir) / "target_tuned_summary.json"
    with open(tuned_path, "w", encoding="utf-8") as handle:
        json.dump(tuned_summary, handle, indent=2)

    return {
        "metrics_df": metrics_df,
        "per_target_val_metrics": per_target_val_metrics,
        "per_target_test_metrics": per_target_test_metrics,
        "best_by_target": best_by_target,
        "tuned_metrics": tuned_metrics,
        "tuned_per_target_metrics": tuned_per_target_metrics,
        "combined_path": combined_path,
        "best_path": best_path,
        "tuned_path": tuned_path,
    }


def main():
    results = run_per_target_tuning(CONFIG)
    best_by_target = results["best_by_target"]
    print("\nBest Model Per Target (Validation Split)")
    for target_name, info in best_by_target.items():
        print(
            f"{target_name}: {info['model']} | "
            f"MAE={info['MAE']:.4f} RMSE={info['RMSE']:.4f} NRMSE={info['NRMSE']:.4f}"
        )

    print("\nTarget-Tuned Test Metrics")
    for metric_name, metric_value in results["tuned_metrics"].items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nTarget-Tuned Test Metrics By Target")
    print(pd.DataFrame(results["tuned_per_target_metrics"]).T.round(4).to_string())

    print(f"\nSaved per-target metrics to {results['combined_path']}")
    print(f"Saved best-model mapping to {results['best_path']}")
    print(f"Saved target-tuned summary to {results['tuned_path']}")


if __name__ == "__main__":
    main()
