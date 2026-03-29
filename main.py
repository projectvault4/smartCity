from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / "cache"))

from engine.adaptive_ensemble import AdaptiveEnsemble, DriftDetector
from evaluate import evaluate_models
from train import train_all_models
from utils.analytics import compute_correlation_matrices, granger_causality_table, save_analysis_tables
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, rolling_mean, set_seed
from utils.explainable_forecasting import ExplainableTimeSeriesForecaster
from utils.metrics import compute_all_metrics
from utils.training import fine_tune_model, predict_model
from utils.visualization import (
    plot_correlation_heatmap,
    plot_error_bars,
    plot_feature_attention,
    plot_named_correlation_heatmap,
    plot_predictions,
    plot_rolling_error,
)
from utils.xai import save_explainability_report


def _print_table(title: str, frame: pd.DataFrame, columns: list[str] | None = None):
    print(f"\n{title}")
    if columns is not None:
        available = [col for col in columns if col in frame.columns]
        print(frame[available].round(4).to_string())
    else:
        print(frame.round(4).to_string())


def _format_switcher_choices(switcher_models: dict, config) -> str:
    parts = []
    for target in config.target_columns:
        model_name = switcher_models.get(target, "Unavailable")
        parts.append(f"{target}={model_name}")
    return " | ".join(parts)


def _print_correlation_summary(correlations: dict, top_k: int = 8):
    print("\nPearson Correlation Matrix Summary")
    pearson_pairs = []
    pearson_df = correlations["pearson"]
    for left in pearson_df.index:
        for right in pearson_df.columns:
            if left >= right:
                continue
            pearson_pairs.append((left, right, float(pearson_df.loc[left, right])))
    pearson_pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    for left, right, value in pearson_pairs[:top_k]:
        print(f"{left} <-> {right}: {value:.4f}")

    print("\nSpearman Correlation Matrix Summary")
    spearman_pairs = []
    spearman_df = correlations["spearman"]
    for left in spearman_df.index:
        for right in spearman_df.columns:
            if left >= right:
                continue
            spearman_pairs.append((left, right, float(spearman_df.loc[left, right])))
    spearman_pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    for left, right, value in spearman_pairs[:top_k]:
        print(f"{left} <-> {right}: {value:.4f}")


def _print_granger_summary(granger_df: pd.DataFrame, top_k: int = 8):
    print("\nGranger Causality Test Summary")
    if granger_df.empty:
        print("No Granger causality results available")
        return

    significant = granger_df[granger_df["significant_5pct"]]
    view = significant if not significant.empty else granger_df
    for _, row in view.head(top_k).iterrows():
        lag_value = "NA" if pd.isna(row["best_lag"]) else int(row["best_lag"])
        print(
            f"{row['cause']} -> {row['effect']}: "
            f"lag={lag_value}, p_value={float(row['p_value']):.4f}, "
            f"significant={bool(row['significant_5pct'])}"
        )


def run_streaming_simulation(models, datasets, config):
    processor = datasets["processor"]
    x_test, y_test_scaled = datasets["test_seq"]
    y_true = processor.inverse_transform_targets(y_test_scaled)
    online_models = {name: copy.deepcopy(model) for name, model in models.items()}
    ensemble = AdaptiveEnsemble(list(online_models.keys()), error_window=config.ensemble_error_window)
    detector = DriftDetector(error_window=config.drift_error_window, threshold=config.drift_threshold)

    warmup_end = min(config.streaming_window, len(y_true) // 2)
    warmup_predictions = {}
    for name, model in online_models.items():
        warmup_scaled = predict_model(model, x_test[:warmup_end], config)
        warmup_predictions[name] = processor.inverse_transform_targets(warmup_scaled)
    ensemble.update_errors(y_true[:warmup_end], warmup_predictions)
    ensemble.fit_meta_learner(y_true[:warmup_end], warmup_predictions)

    traces = {name: [] for name in list(online_models.keys()) + ["AdaptiveEnsemble"]}
    drift_points = []
    dynamic_preds = []

    chunk_start = warmup_end
    while chunk_start < len(y_true):
        chunk_end = min(chunk_start + config.streaming_step, len(y_true))
        current_x = x_test[chunk_start:chunk_end]
        current_y_scaled = y_test_scaled[chunk_start:chunk_end]
        current_y = y_true[chunk_start:chunk_end]

        current_preds = {}
        for name, model in online_models.items():
            pred_scaled = predict_model(model, current_x, config)
            current_preds[name] = processor.inverse_transform_targets(pred_scaled)

        ensemble_pred = ensemble.predict(current_preds)
        dynamic_preds.extend(ensemble_pred.tolist())

        for name, preds in current_preds.items():
            traces[name].extend(np.mean(np.abs(current_y - preds), axis=1).tolist())
        traces["AdaptiveEnsemble"].extend(np.mean(np.abs(current_y - ensemble_pred), axis=1).tolist())
        ensemble.update_errors(current_y, current_preds)

        drift_detected = False
        for offset, err in enumerate(np.mean(np.abs(current_y - ensemble_pred), axis=1)):
            if detector.update(float(err)):
                drift_points.append(chunk_start + offset)
                drift_detected = True

        recent_start = max(0, chunk_end - config.streaming_window)
        recent_x = x_test[recent_start:chunk_end]
        recent_y = y_test_scaled[recent_start:chunk_end]
        if len(recent_x) > 1 and (drift_detected or chunk_end % (config.streaming_step * 2) == 0):
            for name, model in online_models.items():
                online_models[name] = fine_tune_model(model, (recent_x, recent_y), config, epochs=2 if drift_detected else 1)
            refreshed_predictions = {}
            recent_y_true = processor.inverse_transform_targets(recent_y)
            for name, model in online_models.items():
                refreshed_scaled = predict_model(model, recent_x, config)
                refreshed_predictions[name] = processor.inverse_transform_targets(refreshed_scaled)
            ensemble.update_errors(recent_y_true, refreshed_predictions)
            ensemble.fit_meta_learner(recent_y_true, refreshed_predictions)

        if drift_detected:
            weights = ensemble.weights.copy()
            weights["Hybrid"] *= 1.2
            total = sum(weights.values())
            ensemble.weights = {k: v / total for k, v in weights.items()}

        chunk_start = chunk_end

    rolling_errors = {name: rolling_mean(values, window=24) for name, values in traces.items()}
    streaming_truth = y_true[warmup_end : warmup_end + len(dynamic_preds)]
    streaming_metrics = compute_all_metrics(streaming_truth, np.array(dynamic_preds))
    return np.array(dynamic_preds), rolling_errors, streaming_metrics, drift_points, ensemble.weights


def save_metrics(
    metrics_df: pd.DataFrame,
    per_target_metrics: dict,
    streaming_metrics: dict,
    ensemble_weights: dict,
    switcher_models: dict,
    xai_reports: dict,
    config,
):
    metrics_path = Path(config.output_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_path)

    summary = {
        "offline_metrics": metrics_df.to_dict(orient="index"),
        "per_target_metrics": per_target_metrics,
        "streaming_metrics": streaming_metrics,
        "final_ensemble_weights": {k: float(v) for k, v in ensemble_weights.items()},
        "adaptive_switcher": switcher_models,
        "xai_reports": xai_reports,
    }
    with open(Path(config.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


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
    explainable_artifacts = explainable_model.fit(datasets["prepared_df"])
    explainable_model.save(Path(config.checkpoint_dir) / "explainable_forecaster.pkl")
    xai_reports = save_explainability_report(explainable_model, datasets["prepared_df"], Path(config.output_dir))

    models, _ = train_all_models(datasets, config)
    metrics_df, per_target_metrics, predictions, feature_weights, ensemble_weights, switcher_models = evaluate_models(models, datasets, config)

    y_true = datasets["processor"].inverse_transform_targets(datasets["test_seq"][1])
    streaming_pred, rolling_errors, streaming_metrics, drift_points, final_streaming_weights = run_streaming_simulation(models, datasets, config)

    aqi_idx = list(config.target_columns).index("aqi")
    plot_predictions(y_true[:, aqi_idx], {name: pred[:, aqi_idx] for name, pred in predictions.items()}, Path(config.plot_dir) / "predictions_vs_actual_aqi.png")
    plot_error_bars(metrics_df, Path(config.plot_dir) / "error_comparison.png")
    plot_rolling_error(rolling_errors, Path(config.plot_dir) / "rolling_error.png")
    plot_correlation_heatmap(datasets["prepared_df"].drop(columns=["timestamp"]), Path(config.plot_dir) / "correlation_heatmap.png")
    plot_feature_attention(feature_weights.get("Hybrid"), datasets["processor"].feature_columns, Path(config.plot_dir) / "feature_attention.png")
    correlations = compute_correlation_matrices(datasets["prepared_df"].drop(columns=["timestamp"]))
    granger_df = granger_causality_table(datasets["prepared_df"], list(config.domain_columns), max_lag=config.granger_max_lag)
    save_analysis_tables(correlations, granger_df, Path(config.output_dir))
    plot_named_correlation_heatmap(correlations["pearson"], Path(config.plot_dir) / "pearson_correlation_heatmap.png", "Pearson Correlation Heatmap")
    plot_named_correlation_heatmap(correlations["spearman"], Path(config.plot_dir) / "spearman_correlation_heatmap.png", "Spearman Correlation Heatmap")

    save_metrics(metrics_df, per_target_metrics, streaming_metrics, final_streaming_weights, switcher_models, xai_reports, config)

    _print_table("Explainable Next-Hour Model Metrics", explainable_artifacts.metrics, ["MAE", "RMSE", "NRMSE"])
    _print_table("Offline Evaluation Metrics", metrics_df, ["MAE", "RMSE", "NRMSE", "UPS"])
    print(f"AdaptiveSwitcher chose: {_format_switcher_choices(switcher_models, config)}")
    _print_table("Per-Target Adaptive Ensemble Metrics", pd.DataFrame(per_target_metrics["AdaptiveEnsemble"]).T, ["MAE", "RMSE", "NRMSE"])
    print("\nStreaming Adaptive Ensemble Metrics")
    for metric_name, value in streaming_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\nFinal Dynamic Ensemble Weights")
    for name, weight in final_streaming_weights.items():
        print(f"{name}: {weight:.4f}")
    print("\nAdaptive Switcher Selected Base Models")
    for target, model_name in switcher_models.items():
        print(f"{target}: {model_name}")
    _print_correlation_summary(correlations)
    _print_granger_summary(granger_df)
    print(f"\nDetected drift points: {drift_points[:10]}{'...' if len(drift_points) > 10 else ''}")


if __name__ == "__main__":
    main()
