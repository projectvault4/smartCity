from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from engine.adaptive_ensemble import AdaptiveDomainSwitcher
from evaluate import evaluate_models
from models.hybrid_ablation import StaticFusionHybridModel
from models.recurrent_baselines import PlainGRU, PlainLSTM
from train import build_models, checkpoint_name_for_model, train_selected_models
from utils.baselines import _load_or_train_neural_baselines, evaluate_baselines
from utils.metrics import (
    compute_all_metrics,
    compute_metrics_by_target,
    compute_urban_prediction_score,
    urban_prediction_score_from_normalized_error,
)
from utils.training import predict_model, train_model
from utils.visualization import (
    plot_forecast_windows,
    plot_residual_boxplot,
    plot_residual_histograms,
)


def _merge_temporal_inputs(grouped: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([grouped["closeness"], grouped["period"], grouped["trend"]], axis=1).astype(np.float32)


def _predict_any_model(model, grouped_x, config):
    if isinstance(model, (PlainLSTM, PlainGRU)):
        return predict_model(model, _merge_temporal_inputs(grouped_x), config)
    return predict_model(model, grouped_x, config)


def ordered_model_names(config, metrics_df: pd.DataFrame, literature_rows: list[dict]) -> list[str]:
    literature_names = [item["model"] for item in literature_rows]
    present_names = list(metrics_df.index)
    ordered = [name for name in literature_names if name in present_names]
    ordered.extend(name for name in present_names if name not in ordered)
    return ordered


def save_per_target_tables(
    per_target_metrics: dict,
    config,
    output_dir: Path,
    literature_rows: list[dict],
    prefix: str = "",
) -> dict[str, Path]:
    rows = []
    for model_name, target_map in per_target_metrics.items():
        for target_name, metrics in target_map.items():
            metric_values = {metric_name: float(metric_value) for metric_name, metric_value in metrics.items()}
            if "NRMSE" in metric_values:
                metric_values["UPS"] = urban_prediction_score_from_normalized_error(metric_values["NRMSE"])
            rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    **metric_values,
                }
            )

    long_df = pd.DataFrame(rows)
    order = ordered_model_names(config, long_df[["model"]].drop_duplicates().set_index("model"), literature_rows)
    target_order = list(config.target_columns)
    long_df["model"] = pd.Categorical(long_df["model"], categories=order, ordered=True)
    long_df["target"] = pd.Categorical(long_df["target"], categories=target_order, ordered=True)
    long_df = long_df.sort_values(["target", "model"]).reset_index(drop=True)

    stem = f"{prefix}_" if prefix else ""
    long_path = output_dir / f"{stem}per_target_results_long.csv"
    long_df.to_csv(long_path, index=False)

    metric_paths = {}
    for metric_name in ("MAE", "MAPE", "RMSE", "NRMSE", "UPS"):
        pivot = long_df.pivot(index="model", columns="target", values=metric_name)
        pivot = pivot.reindex(order)
        metric_path = output_dir / f"{stem}per_target_{metric_name.lower()}.csv"
        pivot.to_csv(metric_path)
        metric_paths[metric_name] = metric_path

    return {"long": long_path, **metric_paths}


def save_fair_baseline_note(
    literature_rows: list[dict],
    evaluated_models: Iterable[str],
    output_dir: Path,
) -> Path:
    evaluated_models = set(evaluated_models)
    lines = [
        "# Fair Baseline Note",
        "",
        "Evaluated in this project:",
    ]
    implemented = sorted([item["model"] for item in literature_rows if item["model"] in evaluated_models])
    lines.extend(f"- {name}" for name in implemented)
    lines.append("")
    lines.append("Literature-only references:")
    literature_only = sorted([item["model"] for item in literature_rows if item["model"] not in evaluated_models])
    lines.extend(f"- {name}" for name in literature_only)
    lines.append("")
    lines.append(
        "Only models listed under 'Evaluated in this project' have metrics produced from this repository's training and evaluation pipeline."
    )
    note_path = output_dir / "fair_baseline_note.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


def save_split_protocol(datasets, config, output_dir: Path) -> Path:
    payload = {
        "sequence_length": int(config.seq_len),
        "forecast_horizon": int(config.forecast_horizon),
        "targets": list(config.target_columns),
        "train": {
            "rows": int(len(datasets["train_df"])),
            "start": str(datasets["train_df"]["timestamp"].iloc[0]),
            "end": str(datasets["train_df"]["timestamp"].iloc[-1]),
        },
        "validation": {
            "rows": int(len(datasets["val_df"])),
            "start": str(datasets["val_df"]["timestamp"].iloc[0]),
            "end": str(datasets["val_df"]["timestamp"].iloc[-1]),
        },
        "test": {
            "rows": int(len(datasets["test_df"])),
            "start": str(datasets["test_df"]["timestamp"].iloc[0]),
            "end": str(datasets["test_df"]["timestamp"].iloc[-1]),
        },
        "leakage_prevention": [
            "The dataset is split chronologically into train, validation, and test segments.",
            "Lag and rolling-window features only use past observations because they are created with trailing shifts/rolling windows.",
            "Rows with insufficient history are dropped after feature engineering.",
            "Feature and target scalers are fitted on the training split only, then applied to validation and test splits.",
            "Forecast evaluation uses future timestamps that are strictly later than the input context window.",
        ],
    }
    protocol_path = output_dir / "split_protocol.json"
    with open(protocol_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return protocol_path


def train_or_load_static_hybrid_ablation(datasets, config):
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    model = StaticFusionHybridModel(input_dim=input_dim, config=config)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "hybrid_no_gate.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return model, 0.0

    bilstm_checkpoint = checkpoint_dir / "bilstm.pt"
    tft_checkpoint = checkpoint_dir / "tft.pt"
    if bilstm_checkpoint.exists():
        model.bilstm_branch.load_state_dict(torch.load(bilstm_checkpoint, map_location="cpu"))
    if tft_checkpoint.exists():
        model.tft_branch.load_state_dict(torch.load(tft_checkpoint, map_location="cpu"))

    for parameter in model.bilstm_branch.parameters():
        parameter.requires_grad = False
    for parameter in model.tft_branch.parameters():
        parameter.requires_grad = False

    x_train = {key: datasets["train_tpt"][key] for key in ("closeness", "period", "trend")}
    y_train = datasets["train_tpt"]["target"]
    x_val = {key: datasets["val_tpt"][key] for key in ("closeness", "period", "trend")}
    y_val = datasets["val_tpt"]["target"]
    ablation_config = copy.deepcopy(config)
    ablation_config.epochs = min(config.epochs, 4)
    ablation_config.patience = min(config.patience, 1)
    start = time.perf_counter()
    train_model(
        model=model,
        model_name="hybrid_no_gate",
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        config=ablation_config,
        checkpoint_dir=checkpoint_dir,
    )
    elapsed = time.perf_counter() - start
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model, elapsed


def evaluate_ablation_study(core_metrics_df, core_per_target_metrics, datasets, config, output_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    model, training_time = train_or_load_static_hybrid_ablation(datasets, config)
    processor = datasets["processor"]
    x_test = {key: datasets["test_tpt"][key] for key in ("closeness", "period", "trend")}
    y_true = processor.inverse_transform_targets(datasets["test_tpt"]["target"])
    predictions = processor.inverse_transform_targets(predict_model(model, x_test, config))

    no_gate_metrics = compute_all_metrics(y_true, predictions)
    no_gate_metrics["UPS"] = compute_urban_prediction_score(y_true, predictions, config.target_columns)
    no_gate_per_target = compute_metrics_by_target(y_true, predictions, config.target_columns)

    ablation_rows = [
        {
            "model": "BiLSTM branch only",
            **{metric: float(core_metrics_df.loc["BiLSTM", metric]) for metric in ("MAE", "MAPE", "RMSE", "NRMSE", "UPS")},
        },
        {
            "model": "TFT branch only",
            **{metric: float(core_metrics_df.loc["TFT", metric]) for metric in ("MAE", "MAPE", "RMSE", "NRMSE", "UPS")},
        },
        {"model": "Hybrid without gating", **no_gate_metrics},
        {
            "model": "Hybrid with gating",
            **{metric: float(core_metrics_df.loc["Hybrid", metric]) for metric in ("MAE", "MAPE", "RMSE", "NRMSE", "UPS")},
        },
        {
            "model": "Adaptive switcher",
            **{
                metric: float(core_metrics_df.loc["AdaptiveSwitcher", metric])
                for metric in ("MAE", "MAPE", "RMSE", "NRMSE", "UPS")
            },
        },
    ]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = output_dir / "ablation_study.csv"
    ablation_df.to_csv(ablation_path, index=False)

    metadata = {
        "hybrid_without_gating_training_seconds": training_time,
        "path": str(ablation_path),
    }
    per_target = {
        "BiLSTM branch only": core_per_target_metrics["BiLSTM"],
        "TFT branch only": core_per_target_metrics["TFT"],
        "Hybrid without gating": no_gate_per_target,
        "Hybrid with gating": core_per_target_metrics["Hybrid"],
        "Adaptive switcher": core_per_target_metrics["AdaptiveSwitcher"],
    }
    return ablation_df, per_target, metadata


def _measure_inference_latency(model, x, config, repeats: int = 2) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _predict_any_model(model, x, config)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(timings))


def save_efficiency_table(models: dict, datasets, config, output_dir: Path, extra_training_times: dict | None = None) -> Path:
    extra_training_times = extra_training_times or {}
    x_test = {
        key: datasets["test_tpt"][key][: min(512, len(datasets["test_tpt"][key]))]
        for key in ("closeness", "period", "trend")
    }
    rows = []
    for name, model in models.items():
        param_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        checkpoint_name = "hybrid_no_gate.pt" if name == "HybridNoGate" else f"{checkpoint_name_for_model(name)}.pt"
        checkpoint_path = Path(config.checkpoint_dir) / checkpoint_name
        rows.append(
            {
                "model": name,
                "parameters": int(param_count),
                "trainable_parameters": int(trainable_count),
                "parameter_memory_mb": round(param_count * 4 / (1024**2), 4),
                "checkpoint_size_mb": round(checkpoint_path.stat().st_size / (1024**2), 4) if checkpoint_path.exists() else None,
                "inference_latency_ms": round(_measure_inference_latency(model, x_test, config), 4),
                "training_time_seconds": round(float(extra_training_times.get(name)), 4) if name in extra_training_times else None,
            }
        )

    efficiency_df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    efficiency_path = output_dir / "efficiency_metrics.csv"
    efficiency_df.to_csv(efficiency_path, index=False)
    return efficiency_path


def save_forecast_and_residual_plots(y_true, predictions: dict[str, np.ndarray], config, plot_dir: Path) -> list[Path]:
    selected_models = [name for name in ("SARIMA", "GRU", "Informer", "PatchTST", "Hybrid", "AdaptiveSwitcher") if name in predictions]
    selected_predictions = {name: predictions[name] for name in selected_models}
    saved_paths = []
    for target_idx, target_name in enumerate(config.target_columns):
        forecast_path = plot_dir / f"forecast_windows_{target_name}.png"
        residual_hist_path = plot_dir / f"residual_hist_{target_name}.png"
        residual_box_path = plot_dir / f"residual_box_{target_name}.png"
        plot_forecast_windows(y_true[:, target_idx], {name: pred[:, target_idx] for name, pred in selected_predictions.items()}, forecast_path)
        plot_residual_histograms(y_true[:, target_idx], {name: pred[:, target_idx] for name, pred in selected_predictions.items()}, residual_hist_path)
        plot_residual_boxplot(y_true[:, target_idx], {name: pred[:, target_idx] for name, pred in selected_predictions.items()}, residual_box_path)
        saved_paths.extend([forecast_path, residual_hist_path, residual_box_path])
    return saved_paths


def _apply_noise(group_array: np.ndarray, scale: float) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=scale, size=group_array.shape).astype(np.float32)
    return group_array + noise


def _apply_missing(group_array: np.ndarray, ratio: float, fill_value: float) -> np.ndarray:
    masked = group_array.copy()
    missing_mask = np.random.rand(*group_array.shape) < ratio
    masked[missing_mask] = fill_value
    return masked


def save_robustness_results(models: dict, datasets, config, output_dir: Path) -> Path:
    processor = datasets["processor"]
    subset_end = min(1024, len(datasets["test_tpt"]["target"]))
    base_x_test = {key: datasets["test_tpt"][key][:subset_end] for key in ("closeness", "period", "trend")}
    base_x_val = {key: datasets["val_tpt"][key] for key in ("closeness", "period", "trend")}
    y_true = processor.inverse_transform_targets(datasets["test_tpt"]["target"][:subset_end])
    y_val = processor.inverse_transform_targets(datasets["val_tpt"]["target"])

    fill_values = {
        key: float(np.mean(datasets["train_tpt"][key]))
        for key in ("closeness", "period", "trend")
    }

    selected_model_names = [name for name in ("BiLSTM", "TFT", "Informer", "PatchTST", "Hybrid", "GRU", "HybridNoGate") if name in models]
    selected_models = {name: models[name] for name in selected_model_names}

    scenarios = {
        "clean": base_x_test,
        "noise_0.02": {key: _apply_noise(value, 0.02) for key, value in base_x_test.items()},
        "noise_0.05": {key: _apply_noise(value, 0.05) for key, value in base_x_test.items()},
        "missing_05": {key: _apply_missing(value, 0.05, fill_values[key]) for key, value in base_x_test.items()},
        "missing_10": {key: _apply_missing(value, 0.10, fill_values[key]) for key, value in base_x_test.items()},
        "drift_tail": {key: value[len(value) // 2 :] for key, value in base_x_test.items()},
    }

    val_predictions = {
        name: processor.inverse_transform_targets(_predict_any_model(model, base_x_val, config))
        for name, model in selected_models.items()
    }
    switcher = AdaptiveDomainSwitcher(
        model_names=list(selected_models.keys()),
        target_names=config.target_columns,
        switch_window=config.adaptive_switch_window,
    )
    switcher.update(y_val, val_predictions)

    rows = []
    for scenario_name, scenario_x in scenarios.items():
        scenario_y = y_true if scenario_name != "drift_tail" else y_true[len(y_true) // 2 :]
        scenario_predictions = {}
        for name, model in selected_models.items():
            pred = processor.inverse_transform_targets(_predict_any_model(model, scenario_x, config))
            scenario_predictions[name] = pred
            metrics = compute_all_metrics(scenario_y, pred)
            metrics["UPS"] = compute_urban_prediction_score(scenario_y, pred, config.target_columns)
            rows.append({"scenario": scenario_name, "model": name, **metrics})

        switched = switcher.predict(scenario_predictions)
        switched_metrics = compute_all_metrics(scenario_y, switched)
        switched_metrics["UPS"] = compute_urban_prediction_score(scenario_y, switched, config.target_columns)
        rows.append({"scenario": scenario_name, "model": "AdaptiveSwitcher", **switched_metrics})

    robustness_df = pd.DataFrame(rows)
    robustness_path = output_dir / "robustness_results.csv"
    robustness_df.to_csv(robustness_path, index=False)
    return robustness_path


def run_reliability_study(raw_df: pd.DataFrame, base_config, output_dir: Path, seeds: Iterable[int]) -> Path:
    from utils.data_utils import create_datasets, set_seed

    seed_rows = []
    seeds = list(seeds)
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config.random_seed = int(seed)
        config.checkpoint_dir = output_dir / f"reliability_checkpoints_seed_{seed}"
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        set_seed(config.random_seed)
        datasets = create_datasets(config, raw_df)
        models = build_models(datasets["train_tpt"]["closeness"].shape[-1], config)
        train_selected_models(models, datasets, config, model_names=list(models.keys()))
        metrics_df, _, _, _, _ = evaluate_models(models, datasets, config)

        for model_name, row in metrics_df.iterrows():
            seed_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "MAE": float(row["MAE"]),
                    "MAPE": float(row["MAPE"]),
                    "RMSE": float(row["RMSE"]),
                    "NRMSE": float(row["NRMSE"]),
                    "UPS": float(row["UPS"]),
                }
            )

    reliability_long = pd.DataFrame(seed_rows)
    summary = (
        reliability_long.groupby("model")[["MAE", "MAPE", "RMSE", "NRMSE", "UPS"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["model"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[1:]]
    long_path = output_dir / "reliability_by_seed.csv"
    summary_path = output_dir / "reliability_summary.csv"
    reliability_long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    return summary_path


def load_neural_models_for_robustness(datasets, config) -> dict:
    models = build_models(datasets["train_tpt"]["closeness"].shape[-1], config)
    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / f"{checkpoint_name_for_model(name)}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    recurrent = _load_or_train_neural_baselines(datasets, config)
    return {**models, **recurrent}
