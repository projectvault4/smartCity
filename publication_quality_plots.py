from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / "cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.bilstm import EnhancedBiLSTM
from models.hybrid import TFTGRUResidualHybrid
from models.transformer import TemporalFusionTransformer
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import mae, mape, rmse
from utils.training import predict_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality figures from real model outputs.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where PNG files will be saved.",
    )
    parser.add_argument(
        "--prediction-model",
        default="Hybrid",
        choices=["BiLSTM", "TFT", "Hybrid"],
        help="Model to use for the prediction-vs-actual figure.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=250,
        help="Number of test time steps to display in the prediction plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
        }
    )


def _window(length: int, anchor: int) -> tuple[int, ...]:
    return tuple(range(anchor, anchor + length))


def _load_model_artifacts() -> dict[str, dict]:
    summary = _load_json(Path(CONFIG.output_dir) / "project_best_summary.json")
    artifacts = {}
    for model_name, filename in {
        "BiLSTM": "project_best_bilstm_best.json",
        "TFT": "project_best_tft_best.json",
        "Hybrid": "project_best_hybrid_best.json",
    }.items():
        artifacts[model_name] = {
            "best": _load_json(Path(CONFIG.output_dir) / filename),
            "checkpoint": Path(summary["results"][model_name]["best_checkpoint"]),
        }
    return artifacts


def _config_for_model(model_name: str, best: dict) -> object:
    config = copy.deepcopy(CONFIG)
    config.device = "cpu"

    if model_name == "BiLSTM":
        seq_len = int(best["seq_len"])
        config.seq_len = seq_len
        config.batch_size = int(best["batch_size"])
        config.dropout = float(best["dropout"])
        config.learning_rate = float(best["learning_rate"])
        config.bilstm_hidden_dim = int(best["hidden_dim"])
        config.closeness_lags = _window(seq_len, 1)
        config.period_lags = _window(seq_len, 24)
        config.trend_lags = _window(seq_len, 24 * 7)
        return config

    if model_name == "TFT":
        seq_len = int(best["seq_len"])
        config.seq_len = seq_len
        config.batch_size = int(best["batch_size"])
        config.dropout = float(best["dropout"])
        config.learning_rate = float(best["learning_rate"])
        config.tft_hidden_dim = int(best["hidden_dim"])
        config.tft_heads = int(best["heads"])
        config.tft_layers = int(best["layers"])
        config.tft_ff_dim = int(best["ff_dim"])
        config.closeness_lags = _window(seq_len, 1)
        config.period_lags = _window(seq_len, 24)
        config.trend_lags = _window(seq_len, 24 * 7)
        return config

    if model_name == "Hybrid":
        config.batch_size = int(best["batch_size"])
        config.dropout = float(best["dropout"])
        config.learning_rate = float(best["learning_rate"])
        config.bilstm_hidden_dim = int(best["bilstm_hidden_dim"])
        config.tft_hidden_dim = int(best["tft_hidden_dim"])
        config.tft_heads = int(best["tft_heads"])
        config.tft_layers = int(best["tft_layers"])
        config.tft_ff_dim = int(best["tft_ff_dim"])
        config.dense_hidden_dim = int(best["dense_hidden_dim"])
        config.closeness_lags = _window(int(best["closeness_len"]), 1)
        config.period_lags = _window(int(best["period_len"]), 24)
        config.trend_lags = _window(int(best["trend_len"]), 24 * 7)
        return config

    raise ValueError(f"Unsupported model: {model_name}")


def _build_model(model_name: str, input_dim: int, config) -> torch.nn.Module:
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
    raise ValueError(f"Unsupported model: {model_name}")


def _collect_real_outputs(raw_df: pd.DataFrame, prediction_model: str) -> dict[str, dict]:
    artifacts = _load_model_artifacts()
    results: dict[str, dict] = {}

    for model_name, artifact in artifacts.items():
        print(f"Loading real outputs for {model_name}...", flush=True)
        if model_name == "Hybrid" and (Path(CONFIG.output_dir) / "tft_gru_residual_hybrid_metrics.json").exists():
            metrics_doc = _load_json(Path(CONFIG.output_dir) / "tft_gru_residual_hybrid_metrics.json")
            entry = {
                "metrics": {
                    "RMSE": float(metrics_doc["metrics"]["RMSE"]),
                    "MAE": float(metrics_doc["metrics"]["MAE"]),
                    "MAPE": float(metrics_doc["metrics"]["MAPE"]),
                }
            }
            prediction_path = Path(metrics_doc["prediction_path"])
            if prediction_model == "Hybrid" and prediction_path.exists():
                prediction_df = pd.read_csv(prediction_path)
                actual_columns = [f"actual_{target}" for target in CONFIG.target_columns]
                predicted_columns = [f"predicted_{target}" for target in CONFIG.target_columns]
                entry["y_true"] = prediction_df[actual_columns].to_numpy(dtype=float)
                entry["y_pred"] = prediction_df[predicted_columns].to_numpy(dtype=float)
                entry["timestamps"] = pd.to_datetime(prediction_df["timestamp"]).reset_index(drop=True)
            results[model_name] = entry
            print(
                f"{model_name} metrics: "
                f"RMSE={entry['metrics']['RMSE']:.2f}, "
                f"MAE={entry['metrics']['MAE']:.2f}, "
                f"MAPE={entry['metrics']['MAPE']:.2f}%",
                flush=True,
            )
            continue

        config = _config_for_model(model_name, artifact["best"])
        datasets = create_datasets(config, raw_df)
        input_dim = datasets["train_tpt"]["closeness"].shape[-1]
        model = _build_model(model_name, input_dim, config)
        model.load_state_dict(torch.load(artifact["checkpoint"], map_location="cpu"))

        test_groups = datasets["test_tpt"]
        x_test = {key: test_groups[key] for key in ("closeness", "period", "trend")}
        processor = datasets["processor"]
        y_true = processor.inverse_transform_targets(test_groups["target"])
        y_pred = processor.inverse_transform_targets(predict_model(model, x_test, config))

        entry = {
            "metrics": {
                "RMSE": rmse(y_true, y_pred),
                "MAE": mae(y_true, y_pred),
                "MAPE": mape(y_true, y_pred),
            }
        }
        if model_name == prediction_model:
            entry["y_true"] = y_true
            entry["y_pred"] = y_pred
            entry["timestamps"] = pd.to_datetime(test_groups["timestamp"]).reset_index(drop=True)
        results[model_name] = entry
        print(
            f"{model_name} metrics: "
            f"RMSE={entry['metrics']['RMSE']:.2f}, "
            f"MAE={entry['metrics']['MAE']:.2f}, "
            f"MAPE={entry['metrics']['MAPE']:.2f}%",
            flush=True,
        )

        del model
        del datasets
        del x_test
        del y_true
        del y_pred
        gc.collect()

    return results


def _save_prediction_plot(model_outputs: dict, output_dir: Path, max_points: int, keep_open: bool) -> None:
    target_name = CONFIG.target_col
    target_idx = list(CONFIG.target_columns).index(target_name)
    y_true = model_outputs["y_true"][:, target_idx]
    y_pred = model_outputs["y_pred"][:, target_idx]
    timestamps = model_outputs["timestamps"]

    end = min(max_points, len(y_true))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(timestamps.iloc[:end], y_true[:end], label="Actual")
    ax.plot(timestamps.iloc[:end], y_pred[:end], label="Predicted")
    ax.set_title("Prediction vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "prediction.png", bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def _save_bar_chart(
    models: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
    keep_open: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(models, values)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / filename, bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def _save_heatmap(raw_df: pd.DataFrame, output_dir: Path, keep_open: bool) -> None:
    corr = raw_df.drop(columns=["timestamp"]).corr()
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(corr.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Heatmap")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Correlation")

    for row_idx in range(corr.shape[0]):
        for col_idx in range(corr.shape[1]):
            ax.text(col_idx, row_idx, f"{corr.iat[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "heatmap.png", bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def _save_time_series_trends(raw_df: pd.DataFrame, output_dir: Path, keep_open: bool) -> None:
    series_to_plot = [
        ("aqi", "AQI over Time"),
        ("traffic_flow", "Traffic over Time"),
        ("electricity_demand", "Electricity over Time"),
    ]
    available = [(column, title) for column, title in series_to_plot if column in raw_df.columns]
    fig, axes = plt.subplots(len(available), 1, figsize=(10, 7), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (column, title) in zip(axes, available):
        ax.plot(raw_df["timestamp"], raw_df[column])
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(output_dir / "time_series_trends.png", bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def _print_real_metrics(model_outputs: dict[str, dict]) -> None:
    print("\nReal test metrics from saved project-best checkpoints")
    for model_name in ("BiLSTM", "TFT", "Hybrid"):
        metrics = model_outputs[model_name]["metrics"]
        print(
            f"{model_name}: "
            f"RMSE={metrics['RMSE']:.2f}, "
            f"MAE={metrics['MAE']:.2f}, "
            f"MAPE={metrics['MAPE']:.2f}%"
        )


def main() -> None:
    args = parse_args()
    _set_publication_style()
    set_seed(CONFIG.random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_input_dataframe(CONFIG)
    model_outputs = _collect_real_outputs(raw_df, args.prediction_model)
    summary_metrics = _load_json(Path(CONFIG.output_dir) / "summary.json")["offline_metrics"]
    keep_open = args.show

    _save_prediction_plot(model_outputs[args.prediction_model], output_dir, args.max_points, keep_open)
    print("Saved prediction.png", flush=True)
    _save_bar_chart(
        models=["BiLSTM", "TFT", "Hybrid"],
        values=[model_outputs[name]["metrics"]["MAE"] for name in ("BiLSTM", "TFT", "Hybrid")],
        ylabel="MAE",
        title="MAE Comparison",
        filename="mae.png",
        output_dir=output_dir,
        keep_open=keep_open,
    )
    print("Saved mae.png", flush=True)
    _save_bar_chart(
        models=["BiLSTM", "TFT", "Hybrid"],
        values=[model_outputs[name]["metrics"]["RMSE"] for name in ("BiLSTM", "TFT", "Hybrid")],
        ylabel="RMSE",
        title="RMSE Comparison",
        filename="rmse.png",
        output_dir=output_dir,
        keep_open=keep_open,
    )
    print("Saved rmse.png", flush=True)
    _save_bar_chart(
        models=["BiLSTM", "TFT", "Hybrid"],
        values=[summary_metrics[name]["UPS"] for name in ("BiLSTM", "TFT", "Hybrid")],
        ylabel="UPS",
        title="UPS Comparison",
        filename="ups.png",
        output_dir=output_dir,
        keep_open=keep_open,
    )
    print("Saved ups.png", flush=True)
    _save_heatmap(raw_df, output_dir, keep_open)
    print("Saved heatmap.png", flush=True)
    _save_time_series_trends(raw_df, output_dir, keep_open)
    print("Saved time_series_trends.png", flush=True)
    _print_real_metrics(model_outputs)

    print(f"\nSaved figures to {output_dir.resolve()}")
    print("Files: prediction.png, rmse.png, mae.png, ups.png, heatmap.png, time_series_trends.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
