from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / "cache"))

from evaluate import evaluate_models
from train import build_models
from utils.analytics import compute_correlation_matrices, granger_causality_table, save_analysis_tables
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, rolling_mean, set_seed
from utils.explainable_forecasting import ExplainableTimeSeriesForecaster
from utils.visualization import (
    plot_correlation_heatmap,
    plot_error_bars,
    plot_feature_attention,
    plot_named_correlation_heatmap,
    plot_predictions,
    plot_rolling_error,
)
from utils.xai import save_explainability_report


def main():
    config = CONFIG
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.random_seed)
    raw_df = load_input_dataframe(config)
    raw_df.to_csv(Path(config.data_dir) / "prepared_input_snapshot.csv", index=False)
    datasets = create_datasets(config, raw_df)
    explainable_model = ExplainableTimeSeriesForecaster(config)
    explainable_model.fit(datasets["prepared_df"])
    xai_reports = save_explainability_report(explainable_model, datasets["prepared_df"], Path(config.output_dir))

    input_dim = datasets["train_seq"][0].shape[-1]
    models = build_models(input_dim, config)
    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / f"{name.lower()}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}. Run `python3 main.py` first to train models on the current dataset."
            )
        try:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture does not match the current dataset features. "
                "This usually means the checkpoints were trained on an older schema. "
                "Run `python3 main.py` to retrain models on `urban_multivariate_timeseries.csv`, "
                "then rerun `python3 generate_artifacts.py`."
            ) from exc

    metrics_df, per_target_metrics, predictions, feature_weights, ensemble_weights, switcher_models = evaluate_models(models, datasets, config)
    y_true = datasets["processor"].inverse_transform_targets(datasets["test_seq"][1])
    error_traces = {
        name: rolling_mean(np.mean(np.abs(y_true - pred), axis=1).tolist(), 24)
        for name, pred in predictions.items()
    }

    aqi_idx = list(config.target_columns).index("aqi")
    plot_predictions(y_true[:, aqi_idx], {name: pred[:, aqi_idx] for name, pred in predictions.items()}, Path(config.plot_dir) / "predictions_vs_actual_aqi.png")
    plot_error_bars(metrics_df, Path(config.plot_dir) / "error_comparison.png")
    plot_rolling_error(error_traces, Path(config.plot_dir) / "rolling_error.png")
    plot_correlation_heatmap(datasets["prepared_df"].drop(columns=["timestamp"]), Path(config.plot_dir) / "correlation_heatmap.png")
    plot_feature_attention(feature_weights.get("Hybrid"), datasets["processor"].feature_columns, Path(config.plot_dir) / "feature_attention.png")
    correlations = compute_correlation_matrices(datasets["prepared_df"].drop(columns=["timestamp"]))
    granger_df = granger_causality_table(datasets["prepared_df"], list(config.domain_columns), max_lag=config.granger_max_lag)
    save_analysis_tables(correlations, granger_df, Path(config.output_dir))
    plot_named_correlation_heatmap(correlations["pearson"], Path(config.plot_dir) / "pearson_correlation_heatmap.png", "Pearson Correlation Heatmap")
    plot_named_correlation_heatmap(correlations["spearman"], Path(config.plot_dir) / "spearman_correlation_heatmap.png", "Spearman Correlation Heatmap")

    metrics_df.to_csv(Path(config.output_dir) / "metrics.csv")
    with open(Path(config.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "offline_metrics": metrics_df.to_dict(orient="index"),
                "per_target_metrics": per_target_metrics,
                "ensemble_weights": {k: float(v) for k, v in ensemble_weights.items()},
                "adaptive_switcher": switcher_models,
                "xai_reports": xai_reports,
            },
            f,
            indent=2,
        )

    print(metrics_df.round(4).to_string())


if __name__ == "__main__":
    main()
