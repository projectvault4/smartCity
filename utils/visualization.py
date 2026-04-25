from __future__ import annotations

from pathlib import Path

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
    melted = metric_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Value")
    melted = melted.rename(columns={"index": "Model"})
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model")
    plt.title("Error Comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
