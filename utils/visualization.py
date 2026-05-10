from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_predictions(y_true, predictions, save_path: Path, max_points: int = 250):
    plt.figure(figsize=(14, 6))
    view = slice(0, min(max_points, len(y_true)))
    plt.plot(y_true[view], label="Actual", linewidth=2)
    for name, pred in predictions.items():
        plt.plot(pred[view], label=name, alpha=0.8)
    plt.title("Predictions vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error_bars(metric_df: pd.DataFrame, save_path: Path):
    metrics = [metric for metric in ("MAE", "RMSE", "NRMSE", "UPS") if metric in metric_df.columns]
    if not metrics:
        return

    plot_df = metric_df[metrics].reset_index().rename(columns={"index": "Model"})
    melted = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
    melted = melted.dropna(subset=["Value"])

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    palette = dict(zip(plot_df["Model"], sns.color_palette(n_colors=len(plot_df))))
    for axis, metric in zip(axes, metrics):
        metric_values = melted[melted["Metric"] == metric]
        sns.barplot(data=metric_values, x="Model", y="Value", hue="Model", palette=palette, dodge=False, ax=axis)
        axis.set_title(metric)
        axis.set_xlabel("")
        axis.set_ylabel("Value")
        axis.tick_params(axis="x", rotation=35)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

        max_value = float(metric_values["Value"].max()) if not metric_values.empty else 0.0
        axis.set_ylim(0, max_value * 1.15 if max_value > 0 else 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette[model_name])
            for model_name in plot_df["Model"]
        ]
        labels = list(plot_df["Model"])
    fig.suptitle("Error Comparison", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=min(len(labels), 4), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_error(error_traces, save_path: Path):
    plt.figure(figsize=(14, 6))
    for name, errors in error_traces.items():
        plt.plot(errors, label=name)
    plt.title("Rolling Absolute Error")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_named_correlation_heatmap(corr: pd.DataFrame, save_path: Path, title: str):
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_attention(weights, feature_names, save_path: Path, top_k: int = 20):
    if weights is None:
        return
    series = pd.Series(weights, index=feature_names).sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=series.values, y=series.index, orient="h")
    plt.title("Top Feature Attention Weights")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_forecast_windows(y_true, predictions, save_path: Path, window: int = 96):
    total_points = len(y_true)
    if total_points == 0:
        return
    anchors = [0, max(0, total_points // 2 - window // 2), max(0, total_points - window)]
    unique_anchors = []
    for anchor in anchors:
        if anchor not in unique_anchors:
            unique_anchors.append(anchor)

    fig, axes = plt.subplots(len(unique_anchors), 1, figsize=(14, 4 * len(unique_anchors)), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for axis, start in zip(axes, unique_anchors):
        end = min(total_points, start + window)
        axis.plot(y_true[start:end], label="Actual", linewidth=2)
        for name, pred in predictions.items():
            axis.plot(pred[start:end], label=name, alpha=0.85)
        axis.set_title(f"Representative Forecast Window: {start} to {end}")
        axis.set_xlabel("Time Step")
        axis.set_ylabel("Value")
        axis.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_residual_histograms(y_true, predictions, save_path: Path):
    residual_frame = pd.DataFrame({name: np.asarray(pred) - np.asarray(y_true) for name, pred in predictions.items()})
    plt.figure(figsize=(14, 6))
    for column in residual_frame.columns:
        sns.histplot(residual_frame[column], label=column, kde=True, stat="density", element="step", fill=False)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residual_boxplot(y_true, predictions, save_path: Path):
    residual_frame = pd.DataFrame({name: np.asarray(pred) - np.asarray(y_true) for name, pred in predictions.items()})
    melted = residual_frame.melt(var_name="Model", value_name="Residual")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="Model", y="Residual")
    plt.title("Residual Boxplot")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
