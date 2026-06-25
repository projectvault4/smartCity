from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / "cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("GRU", "TFT", "Hybrid")
METRICS = ("UPS", "MAE", "RMSE")
CITY_OUTPUTS = {
    "Delhi": Path("outputs") / "delhi" / "metrics.csv",
    "Bangalore": Path("outputs") / "bangalore" / "metrics.csv",
}
GRAPH_OUTPUT_DIR = Path("outputs") / "graph_outputs"


def _load_city_metrics(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path, index_col=0)
    missing_models = [model for model in MODELS if model not in metrics_df.index]
    missing_metrics = [metric for metric in METRICS if metric not in metrics_df.columns]
    if missing_models or missing_metrics:
        raise ValueError(
            f"{metrics_path} is missing models={missing_models} metrics={missing_metrics}"
        )

    return metrics_df.loc[list(MODELS), list(METRICS)].astype(float)


def _format_bar_label(value: float) -> str:
    if abs(value) >= 10000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _save_city_metric_graph(city: str, metrics_df: pd.DataFrame) -> Path:
    GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "UPS": "#2ca02c",
        "MAE": "#1f77b4",
        "RMSE": "#ff7f0e",
    }

    fig, axes = plt.subplots(1, len(METRICS), figsize=(12.4, 4.4))
    fig.suptitle(f"{city} GRU, TFT, and Hybrid Metrics", fontsize=14)

    for ax, metric in zip(axes, METRICS):
        values = metrics_df[metric].to_numpy()
        bars = ax.bar(MODELS, values, color=colors[metric])
        ax.set_title(metric)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score" if metric == "UPS" else metric)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.margins(y=0.18)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                _format_bar_label(float(value)),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if value >= 10000 else 0,
            )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = GRAPH_OUTPUT_DIR / f"{city.lower()}_gru_tft_hybrid_ups_mae_rmse.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    saved_paths = []
    for city, metrics_path in CITY_OUTPUTS.items():
        metrics_df = _load_city_metrics(metrics_path)
        csv_path = GRAPH_OUTPUT_DIR / f"{city.lower()}_gru_tft_hybrid_ups_mae_rmse.csv"
        GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(csv_path)
        saved_paths.append(_save_city_metric_graph(city, metrics_df))
        saved_paths.append(csv_path)

    print("Saved city model metric comparison outputs:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
