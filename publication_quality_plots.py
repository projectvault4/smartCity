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
from models.recurrent_baselines import PlainGRU
from models.transformer import TemporalFusionTransformer
from utils.config import CONFIG, apply_city_config
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_urban_prediction_score, mae, mape, nrmse, rmse
from utils.training import predict_model

COMPARISON_MODELS = ["Hybrid", "SARIMA", "TFT", "GRU"]
METRIC_COLUMNS = ("MAE", "MAPE", "RMSE", "NRMSE", "UPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality figures from real model outputs.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where PNG files will be saved.",
    )
    parser.add_argument("--city", default=None, help="Use city-specific data and outputs, e.g. bangalore or delhi.")
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


def _city_graph_folder_name() -> str:
    if CONFIG.city == "bangalore":
        return "banglore"
    return CONFIG.city


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir == "outputs" and args.city:
        return Path("outputs") / "graph_outputs" / _city_graph_folder_name()
    return Path(args.output_dir)


def _load_finalized_metric_rows() -> dict[str, dict[str, float]]:
    metrics_path = Path(CONFIG.output_dir) / "metrics.csv"
    if not metrics_path.exists():
        return {}

    metrics_df = pd.read_csv(metrics_path, index_col=0)
    finalized_rows: dict[str, dict[str, float]] = {}
    for model_name in COMPARISON_MODELS:
        if model_name not in metrics_df.index:
            continue
        row = metrics_df.loc[model_name]
        if not all(column in row.index and pd.notna(row[column]) for column in METRIC_COLUMNS):
            continue
        finalized_rows[model_name] = {column: float(row[column]) for column in METRIC_COLUMNS}
    return finalized_rows


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


def _trend_anchor(config) -> int:
    return int(config.trend_lags[0])


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
    artifacts["GRU"] = {
        "best": {},
        "checkpoint": Path(CONFIG.checkpoint_dir) / "gru.pt",
    }
    return artifacts


def _config_for_model(model_name: str, best: dict) -> object:
    config = copy.deepcopy(CONFIG)
    config.device = "cpu"

    if model_name == "GRU":
        return config

    if model_name == "BiLSTM":
        seq_len = int(best["seq_len"])
        config.seq_len = seq_len
        config.batch_size = int(best["batch_size"])
        config.dropout = float(best["dropout"])
        config.learning_rate = float(best["learning_rate"])
        config.bilstm_hidden_dim = int(best["hidden_dim"])
        config.closeness_lags = _window(seq_len, 1)
        config.period_lags = _window(seq_len, 24)
        config.trend_lags = _window(seq_len, _trend_anchor(config))
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
        config.trend_lags = _window(seq_len, _trend_anchor(config))
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
        config.trend_lags = _window(int(best["trend_len"]), _trend_anchor(config))
        return config

    raise ValueError(f"Unsupported model: {model_name}")


def _build_model(model_name: str, input_dim: int, config) -> torch.nn.Module:
    output_dim = len(config.target_columns)
    if model_name == "GRU":
        return PlainGRU(
            input_dim=input_dim,
            hidden_dim=config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            dropout=config.dropout,
            output_dim=output_dim,
        )
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
    finalized_metric_rows = _load_finalized_metric_rows()
    hybrid_metrics_path = Path(CONFIG.output_dir) / "tft_gru_residual_hybrid_lag_stabilized_metrics.json"
    if not hybrid_metrics_path.exists():
        hybrid_metrics_path = Path(CONFIG.output_dir) / "tft_gru_residual_hybrid_metrics.json"

    for model_name in COMPARISON_MODELS:
        print(f"Loading real outputs for {model_name}...", flush=True)
        # SARIMA is a classical model with no torch checkpoint; use its fair
        # metrics from metrics.csv for the comparison bar charts (no overlay).
        if model_name == "SARIMA":
            if model_name in finalized_metric_rows:
                results[model_name] = {"metrics": finalized_metric_rows[model_name]}
                m = finalized_metric_rows[model_name]
                print(
                    f"{model_name} metrics: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, MAPE={m['MAPE']:.2f}%",
                    flush=True,
                )
            continue
        artifact = artifacts[model_name]
        if model_name == "Hybrid" and hybrid_metrics_path.exists():
            metrics_doc = _load_json(hybrid_metrics_path)
            entry = {
                "metrics": {
                    "RMSE": float(metrics_doc["metrics"]["RMSE"]),
                    "MAE": float(metrics_doc["metrics"]["MAE"]),
                    "MAPE": float(metrics_doc["metrics"]["MAPE"]),
                    "NRMSE": float(metrics_doc["metrics"]["NRMSE"]),
                    "UPS": float(metrics_doc["metrics"]["UPS"]),
                }
            }
            prediction_path = Path(metrics_doc["prediction_path"])
            if prediction_path.exists():
                prediction_df = pd.read_csv(prediction_path)
                actual_columns = [f"actual_{target}" for target in CONFIG.target_columns]
                predicted_columns = [f"predicted_{target}" for target in CONFIG.target_columns]
                entry["y_true"] = prediction_df[actual_columns].to_numpy(dtype=float)
                entry["y_pred"] = prediction_df[predicted_columns].to_numpy(dtype=float)
                entry["timestamps"] = pd.to_datetime(prediction_df["timestamp"]).reset_index(drop=True)
            if model_name in finalized_metric_rows:
                entry["metrics"] = finalized_metric_rows[model_name]
                print(f"Using finalized {model_name} metrics from {Path(CONFIG.output_dir) / 'metrics.csv'}.", flush=True)
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
        if model_name == "GRU":
            x_test = np.concatenate(
                [test_groups["closeness"], test_groups["period"], test_groups["trend"]],
                axis=1,
            ).astype(np.float32)
        else:
            x_test = {key: test_groups[key] for key in ("closeness", "period", "trend")}
        processor = datasets["processor"]
        y_true = processor.inverse_transform_targets(test_groups["target"])
        y_pred = processor.inverse_transform_targets(predict_model(model, x_test, config))

        entry = {
            "metrics": {
                "RMSE": rmse(y_true, y_pred),
                "MAE": mae(y_true, y_pred),
                "MAPE": mape(y_true, y_pred),
                "NRMSE": nrmse(y_true, y_pred),
                "UPS": compute_urban_prediction_score(y_true, y_pred, config.target_columns),
            },
            "y_true": y_true,
            "y_pred": y_pred,
            "timestamps": pd.to_datetime(test_groups["timestamp"]).reset_index(drop=True),
        }
        if model_name in finalized_metric_rows:
            entry["metrics"] = finalized_metric_rows[model_name]
            print(f"Using finalized {model_name} metrics from {Path(CONFIG.output_dir) / 'metrics.csv'}.", flush=True)
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


def _save_prediction_plot(model_outputs: dict[str, dict], output_dir: Path, max_points: int, keep_open: bool) -> None:
    target_name = CONFIG.target_col
    target_idx = list(CONFIG.target_columns).index(target_name)
    reference = next(output for output in model_outputs.values() if "y_true" in output)
    y_true = reference["y_true"][:, target_idx]
    timestamps = reference["timestamps"]

    end = min(max_points, len(y_true))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(timestamps.iloc[:end], y_true[:end], label="Actual")

    for model_name in COMPARISON_MODELS:
        output = model_outputs.get(model_name, {})
        if "y_pred" not in output:
            continue
        model_end = min(max_points, len(output["y_pred"]))
        ax.plot(
            output["timestamps"].iloc[:model_end],
            output["y_pred"][:model_end, target_idx],
            label=f"{model_name} Predicted",
        )

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
    y_limits: tuple[float, float] | None = None,
) -> None:
    sorted_pairs = sorted(zip(models, values), key=lambda item: item[1])
    models = [model for model, _ in sorted_pairs]
    values = [value for _, value in sorted_pairs]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(models, values)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        lower, upper = y_limits
        value_min = min(values)
        value_max = max(values)
        if value_min <= lower:
            padding = max((value_max - value_min) * 0.12, 1.0)
            lower = max(0.0, value_min - padding)
        if value_max >= upper:
            upper = value_max + max((value_max - lower) * 0.12, 1.0)
        ax.set_ylim(lower, upper)
    else:
        ax.margins(y=0.12)
    ax.grid(True, axis="y", alpha=0.3)

    y_min, y_max = ax.get_ylim()
    label_offset = (y_max - y_min) * 0.018
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + label_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )
    fig.tight_layout()
    fig.savefig(output_dir / filename, bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def _metrics_for_models(model_outputs: dict[str, dict], models: list[str], metric_name: str) -> list[float]:
    return [float(model_outputs[model_name]["metrics"][metric_name]) for model_name in models]


def _save_heatmap(raw_df: pd.DataFrame, output_dir: Path, keep_open: bool) -> None:
    corr = raw_df.drop(columns=["timestamp"]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    image = ax.imshow(np.ma.array(corr.to_numpy(), mask=mask), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Correlation Heatmap")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Correlation")

    for row_idx in range(corr.shape[0]):
        for col_idx in range(row_idx + 1):
            ax.text(col_idx, row_idx, f"{corr.iat[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "heatmap.png", bbox_inches="tight", dpi=300)
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
    for model_name in COMPARISON_MODELS:
        metrics = model_outputs[model_name]["metrics"]
        print(
            f"{model_name}: "
            f"RMSE={metrics['RMSE']:.2f}, "
            f"MAE={metrics['MAE']:.2f}, "
            f"MAPE={metrics['MAPE']:.2f}%, "
            f"NRMSE={metrics['NRMSE']:.4f}, "
            f"UPS={metrics['UPS']:.2f}"
        )


def main() -> None:
    global CONFIG
    args = parse_args()
    CONFIG = apply_city_config(copy.deepcopy(CONFIG), args.city)
    _set_publication_style()
    set_seed(CONFIG.random_seed)

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_input_dataframe(CONFIG)
    model_outputs = _collect_real_outputs(raw_df, args.prediction_model)
    keep_open = args.show

    _save_prediction_plot(model_outputs, output_dir, args.max_points, keep_open)
    print("Saved prediction.png", flush=True)
    _save_bar_chart(
        models=COMPARISON_MODELS,
        values=_metrics_for_models(model_outputs, COMPARISON_MODELS, "MAE"),
        ylabel="MAE",
        title="MAE Comparison",
        filename="mae.png",
        output_dir=output_dir,
        keep_open=keep_open,
    )
    print("Saved mae.png", flush=True)
    _save_bar_chart(
        models=COMPARISON_MODELS,
        values=_metrics_for_models(model_outputs, COMPARISON_MODELS, "MAPE"),
        ylabel="MAPE (%)",
        title="MAPE Comparison",
        filename="mape.png",
        output_dir=output_dir,
        keep_open=keep_open,
    )
    print("Saved mape.png", flush=True)
    _save_bar_chart(
        models=COMPARISON_MODELS,
        values=_metrics_for_models(model_outputs, COMPARISON_MODELS, "RMSE"),
        ylabel="RMSE",
        title="RMSE Comparison",
        filename="rmse.png",
        output_dir=output_dir,
        keep_open=keep_open,
    )
    print("Saved rmse.png", flush=True)
    _save_bar_chart(
        models=COMPARISON_MODELS,
        values=_metrics_for_models(model_outputs, COMPARISON_MODELS, "NRMSE"),
        ylabel="NRMSE",
        title="NRMSE Comparison",
        filename="nrmse.png",
        output_dir=output_dir,
        keep_open=keep_open,
    )
    print("Saved nrmse.png", flush=True)
    _save_bar_chart(
        models=COMPARISON_MODELS,
        values=_metrics_for_models(model_outputs, COMPARISON_MODELS, "UPS"),
        ylabel="UPS (zoomed scale)",
        title="UPS Comparison",
        filename="ups.png",
        output_dir=output_dir,
        keep_open=keep_open,
        y_limits=(70.0, 93.0),
    )
    print("Saved ups.png", flush=True)
    _save_heatmap(raw_df, output_dir, keep_open)
    print("Saved heatmap.png", flush=True)
    _save_time_series_trends(raw_df, output_dir, keep_open)
    print("Saved time_series_trends.png", flush=True)
    _print_real_metrics(model_outputs)

    print(f"\nSaved figures to {output_dir.resolve()}")
    print("Files: prediction.png, rmse.png, mae.png, mape.png, nrmse.png, ups.png, heatmap.png, time_series_trends.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
