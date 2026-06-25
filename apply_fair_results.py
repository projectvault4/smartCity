from __future__ import annotations

"""Propagate the fair single-window comparison to every canonical output.

Source of truth: outputs/<city>/fair_comparison.csv (produced by
fair_comparison.py, where every model is scored on one identical test window).

This script updates the metric CSV/JSON files that downstream tables, the
finalized-best-model registry, and the next-hour reporting read from, and
regenerates the metric comparison graphs so they match exactly.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import CONFIG, apply_city_config

METRIC_BAR_COLOR = "#1f77b4"
# Models shown in the comparison graphs: proposed hybrid, its TFT/GRU branches,
# and the strongest classical baseline (SARIMA).
GRAPH_MODELS = ["Hybrid", "SARIMA", "TFT", "GRU"]


def _load_json(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {}


def _save_json(path: Path, data: dict) -> None:
    json.dump(data, open(path, "w"), indent=2)


def _bar_chart(series: pd.Series, metric: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(series.index, series.values, color=METRIC_BAR_COLOR)
    ax.set_title(f"{metric} Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAPE (%)" if metric == "MAPE" else metric)
    ax.margins(y=0.12)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=30)
    y_min, y_max = ax.get_ylim()
    off = (y_max - y_min) * 0.018
    for bar, v in zip(bars, series.values):
        fmt = f"{v:.4f}" if metric == "NRMSE" else f"{v:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off, fmt,
                ha="center", va="bottom", fontsize=8, clip_on=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric.lower()}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _all_metrics_chart(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["MAE", "MAPE", "RMSE", "NRMSE", "UPS"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    for ax, metric in zip(axes, metrics):
        s = df[metric]
        ax.bar(s.index, s.values, color=METRIC_BAR_COLOR)
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=70, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle("Model Metric Comparison (fair single-window)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "all_model_metrics.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply fair comparison to all outputs.")
    parser.add_argument("--city", default="bangalore")
    args = parser.parse_args()
    config = apply_city_config(copy.deepcopy(CONFIG), args.city)
    out = Path(config.output_dir)

    fair_path = out / "fair_comparison.csv"
    fair = pd.read_csv(fair_path, index_col=0)
    fair_records = {m: {k: float(v) for k, v in row.items()} for m, row in fair.iterrows()}
    print(f"Loaded fair comparison for {len(fair)} models from {fair_path}")

    # 1) metrics.csv (ordered by the original file order if present).
    fair.to_csv(out / "metrics.csv")
    fair.to_csv(out / "research_performance_comparison.csv")
    print("Updated metrics.csv and research_performance_comparison.csv")

    # 2) summary.json offline_metrics.
    summary_path = out / "summary.json"
    summary = _load_json(summary_path)
    if summary:
        summary.setdefault("offline_metrics", {})
        for m, vals in fair_records.items():
            summary["offline_metrics"][m] = vals
        _save_json(summary_path, summary)
        print("Updated summary.json offline_metrics")

    # 3) project_best_summary.json test_metrics for each present model.
    pbs_path = out / "project_best_summary.json"
    pbs = _load_json(pbs_path)
    if pbs and "results" in pbs:
        for model, payload in pbs["results"].items():
            if model in fair_records and isinstance(payload, dict) and "test_metrics" in payload:
                payload["test_metrics"].update(fair_records[model])
        _save_json(pbs_path, pbs)
        print("Updated project_best_summary.json test_metrics")

    # 4) final_model_registry.json ranking + hybrid override metrics.
    reg_path = out / "final_model_registry.json"
    reg = _load_json(reg_path)
    if reg:
        ranked = fair.sort_values("RMSE")
        reg["top_4_models_by_rmse"] = [
            {"model": name, "RMSE": float(ranked.loc[name, "RMSE"]), "UPS": float(ranked.loc[name, "UPS"])}
            for name in ranked.index[:4]
        ]
        if isinstance(reg.get("hybrid_override"), dict) and "Hybrid" in fair_records:
            reg["hybrid_override"]["final_metrics"] = fair_records["Hybrid"]
        reg["fair_comparison"] = fair_records
        _save_json(reg_path, reg)
        print(f"Updated final_model_registry.json (best by RMSE: {ranked.index[0]})")

    # 5) Regenerate metric comparison graphs from the fair table.
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    graph_models = [m for m in GRAPH_MODELS if m in fair.index]
    graph_df = fair.loc[graph_models]
    for metric in ("MAE", "MAPE", "RMSE", "NRMSE", "UPS"):
        _bar_chart(graph_df[metric], metric, plot_dir)
    _all_metrics_chart(graph_df, plot_dir)
    print(f"Regenerated metric graphs ({', '.join(graph_models)}) in {plot_dir}")

    best = fair.sort_values("RMSE").index[0]
    print(f"\nFinalized best model by RMSE: {best} "
          f"(RMSE={fair.loc[best,'RMSE']:.2f}, UPS={fair.loc[best,'UPS']:.2f})")


if __name__ == "__main__":
    main()
