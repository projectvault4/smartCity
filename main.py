from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / "cache"))

from engine.adaptive_ensemble import AdaptiveDomainSwitcher, DriftDetector
from evaluate import evaluate_models
from train import build_models, checkpoint_name_for_model, train_all_models, train_selected_models
from utils.analytics import compute_correlation_matrices, granger_causality_table, save_analysis_tables
from utils.baselines import evaluate_baselines
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, rolling_mean, set_seed
from utils.explainable_forecasting import ExplainableTimeSeriesForecaster
from utils.metrics import compute_all_metrics, mape
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


def _print_literature_table(config):
    literature_path = Path(config.data_dir) / "literature_models.json"
    print("\nLiterature Reference Models")
    if not literature_path.exists():
        print("No literature model table found.")
        return

    with open(literature_path, "r", encoding="utf-8") as handle:
        literature_rows = json.load(handle)

    literature_df = pd.DataFrame(literature_rows)
    visible_columns = ["model", "family", "year", "comparison_scope", "source_name"]
    print(literature_df[visible_columns].to_string(index=False))


def _build_paper_style_comparison(config, metrics_df: pd.DataFrame, predictions: dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    literature_path = Path(config.data_dir) / "literature_models.json"
    literature_rows = []
    if literature_path.exists():
        with open(literature_path, "r", encoding="utf-8") as handle:
            literature_rows = json.load(handle)

    rows = []
    seen_models = set()
    for item in literature_rows:
        model_name = item["model"]
        seen_models.add(model_name)
        if model_name in metrics_df.index and model_name in predictions:
            rows.append(
                {
                    "Model": model_name,
                    "MAE": f"{float(metrics_df.loc[model_name, 'MAE']):.2f}",
                    "MAPE": f"{mape(y_true, predictions[model_name]):.2f}%",
                    "RMSE": f"{float(metrics_df.loc[model_name, 'RMSE']):.2f}",
                }
            )
        else:
            rows.append({"Model": model_name, "MAE": "-", "MAPE": "-", "RMSE": "-"})

    for model_name in metrics_df.index:
        if model_name in seen_models or model_name not in predictions:
            continue
        rows.append(
            {
                "Model": model_name,
                "MAE": f"{float(metrics_df.loc[model_name, 'MAE']):.2f}",
                "MAPE": f"{mape(y_true, predictions[model_name]):.2f}%",
                "RMSE": f"{float(metrics_df.loc[model_name, 'RMSE']):.2f}",
            }
        )

    return pd.DataFrame(rows)


def _ordered_model_names(config, metrics_df: pd.DataFrame) -> list[str]:
    literature_path = Path(config.data_dir) / "literature_models.json"
    literature_names = []
    if literature_path.exists():
        with open(literature_path, "r", encoding="utf-8") as handle:
            literature_names = [item["model"] for item in json.load(handle)]
    present_names = list(metrics_df.index)
    ordered = [name for name in literature_names if name in present_names]
    ordered.extend(name for name in present_names if name not in ordered)
    return ordered


def _sort_metrics_frame(config, metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.loc[_ordered_model_names(config, metrics_df)]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the adaptive time-series forecasting pipeline.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the slowest reporting stages such as streaming simulation, plots, and analytics tables.",
    )
    parser.add_argument("--skip-streaming", action="store_true", help="Skip the streaming drift simulation.")
    parser.add_argument("--skip-analytics", action="store_true", help="Skip correlation and Granger analytics.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore saved checkpoints and retrain all models.")
    return parser.parse_args()


def run_streaming_simulation(models, datasets, config):
    processor = datasets["processor"]
    test_groups = datasets["test_tpt"]
    x_test = {key: test_groups[key] for key in ("closeness", "period", "trend")}
    y_test_scaled = test_groups["target"]
    y_true = processor.inverse_transform_targets(y_test_scaled)
    online_models = {name: copy.deepcopy(model) for name, model in models.items()}
    detector = DriftDetector(error_window=config.drift_error_window, threshold=config.drift_threshold)
    switcher = AdaptiveDomainSwitcher(
        model_names=list(online_models.keys()),
        target_names=config.target_columns,
        switch_window=config.adaptive_switch_window,
    )

    warmup_end = min(config.streaming_window, len(y_true) // 2)
    traces = {name: [] for name in online_models.keys()}
    traces["AdaptiveSwitcher"] = []
    drift_points = []
    switched_preds = []

    warmup_x = {key: value[:warmup_end] for key, value in x_test.items()}
    warmup_predictions = {}
    for name, model in online_models.items():
        warmup_scaled = predict_model(model, warmup_x, config)
        warmup_predictions[name] = processor.inverse_transform_targets(warmup_scaled)
    switcher.update(y_true[:warmup_end], warmup_predictions)

    chunk_start = warmup_end
    while chunk_start < len(y_true):
        chunk_end = min(chunk_start + config.streaming_step, len(y_true))
        current_x = {key: value[chunk_start:chunk_end] for key, value in x_test.items()}
        current_y_scaled = y_test_scaled[chunk_start:chunk_end]
        current_y = y_true[chunk_start:chunk_end]

        current_preds = {}
        for name, model in online_models.items():
            pred_scaled = predict_model(model, current_x, config)
            current_preds[name] = processor.inverse_transform_targets(pred_scaled)

        switched_pred = switcher.predict(current_preds)
        switched_preds.extend(switched_pred.tolist())

        for name, preds in current_preds.items():
            traces[name].extend(np.mean(np.abs(current_y - preds), axis=1).tolist())
        traces["AdaptiveSwitcher"].extend(np.mean(np.abs(current_y - switched_pred), axis=1).tolist())

        drift_detected = False
        for offset, err in enumerate(np.mean(np.abs(current_y - switched_pred), axis=1)):
            if detector.update(float(err)):
                drift_points.append(chunk_start + offset)
                drift_detected = True

        switcher.update(current_y, current_preds)

        recent_start = max(0, chunk_end - config.streaming_window)
        recent_x = {key: value[recent_start:chunk_end] for key, value in x_test.items()}
        recent_y = y_test_scaled[recent_start:chunk_end]
        if len(recent_y) > 1 and (drift_detected or chunk_end % (config.streaming_step * 2) == 0):
            for name, model in online_models.items():
                online_models[name] = fine_tune_model(model, (recent_x, recent_y), config, epochs=2 if drift_detected else 1)

        chunk_start = chunk_end

    rolling_errors = {name: rolling_mean(values, window=24) for name, values in traces.items()}
    streaming_truth = y_true[warmup_end : warmup_end + len(switched_preds)]
    streaming_metrics = compute_all_metrics(streaming_truth, np.array(switched_preds))
    return np.array(switched_preds), rolling_errors, streaming_metrics, drift_points, switcher.selected_models


def save_metrics(
    metrics_df: pd.DataFrame,
    per_target_metrics: dict,
    streaming_metrics: dict,
    switcher_models: dict,
    xai_reports: dict,
    baseline_metadata: dict,
    config,
):
    metrics_path = Path(config.output_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_path)

    summary = {
        "offline_metrics": metrics_df.to_dict(orient="index"),
        "per_target_metrics": per_target_metrics,
        "streaming_metrics": streaming_metrics,
        "streaming_model": "AdaptiveSwitcher",
        "adaptive_switcher": switcher_models,
        "xai_reports": xai_reports,
        "baseline_metadata": baseline_metadata,
    }
    with open(Path(config.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _load_explainable_forecaster(prepared_df: pd.DataFrame, config):
    checkpoint_path = Path(config.checkpoint_dir) / "explainable_forecaster.pkl"
    if checkpoint_path.exists():
        try:
            explainable_model = ExplainableTimeSeriesForecaster.load(checkpoint_path)
            explainable_artifacts = explainable_model.fit(prepared_df)
            return explainable_model, explainable_artifacts
        except (OSError, pickle.PickleError, AttributeError, ValueError):
            pass

    explainable_model = ExplainableTimeSeriesForecaster(config)
    explainable_artifacts = explainable_model.fit(prepared_df)
    explainable_model.save(checkpoint_path)
    return explainable_model, explainable_artifacts


def _load_or_train_models(datasets, config, force_retrain: bool = False):
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    models = build_models(input_dim, config)
    if force_retrain:
        train_selected_models(models, datasets, config, model_names=list(models.keys()))
        return models, "trained(all)"
    models_to_train = []

    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / f"{checkpoint_name_for_model(name)}.pt"
        if not checkpoint.exists():
            models_to_train.append(name)
            continue
        try:
            model.load_state_dict(torch.load(checkpoint, map_location=config.device))
        except (OSError, RuntimeError):
            models_to_train.append(name)

    if not models_to_train:
        return models, "checkpoints"

    train_selected_models(models, datasets, config, model_names=models_to_train)
    return models, f"checkpoints+trained({', '.join(models_to_train)})"


def _comparison_feature_weights(feature_weights: dict, baseline_metadata: dict) -> dict:
    return {
        "GRU": baseline_metadata.get("GRU", {}).get("feature_weights"),
        "TFT": feature_weights.get("TFT"),
        "Hybrid": feature_weights.get("Hybrid"),
    }


def main():
    args = parse_args()
    config = copy.deepcopy(CONFIG)
    if args.fast:
        config.epochs = min(config.epochs, 3)
        config.patience = min(config.patience, 1)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.random_seed)

    raw_df = load_input_dataframe(config)
    raw_df.to_csv(Path(config.data_dir) / "prepared_input_snapshot.csv", index=False)
    datasets = create_datasets(config, raw_df)
    explainable_model, explainable_artifacts = _load_explainable_forecaster(datasets["prepared_df"], config)
    xai_reports = save_explainability_report(explainable_model, datasets["prepared_df"], Path(config.output_dir))

    models, model_source = _load_or_train_models(datasets, config, force_retrain=args.force_retrain)
    core_metrics_df, per_target_metrics, predictions, feature_weights, switcher_models = evaluate_models(models, datasets, config)
    baseline_metrics_df, baseline_per_target_metrics, baseline_predictions, baseline_metadata = evaluate_baselines(datasets, config)
    predictions = {**predictions, **baseline_predictions}
    per_target_metrics = {**per_target_metrics, **baseline_per_target_metrics}
    metrics_df = pd.concat([core_metrics_df, baseline_metrics_df], axis=0)
    metrics_df = _sort_metrics_frame(config, metrics_df)
    y_true = datasets["processor"].inverse_transform_targets(datasets["test_tpt"]["target"])
    research_table_df = _build_paper_style_comparison(config, metrics_df, predictions, y_true)
    skip_streaming = args.fast or args.skip_streaming
    skip_analytics = args.fast or args.skip_analytics
    skip_plots = args.fast or args.skip_plots

    streaming_metrics = {}
    drift_points = []
    streaming_switcher_models = switcher_models
    rolling_errors = {}
    if not skip_streaming:
        _, rolling_errors, streaming_metrics, drift_points, streaming_switcher_models = run_streaming_simulation(models, datasets, config)

    correlations = None
    granger_df = pd.DataFrame()
    if not skip_analytics:
        correlations = compute_correlation_matrices(datasets["prepared_df"].drop(columns=["timestamp"]))
        granger_df = granger_causality_table(datasets["prepared_df"], list(config.domain_columns), max_lag=config.granger_max_lag)
        save_analysis_tables(correlations, granger_df, Path(config.output_dir))

    if not skip_plots:
        aqi_idx = list(config.target_columns).index("aqi")
        plot_predictions(y_true[:, aqi_idx], {name: pred[:, aqi_idx] for name, pred in predictions.items()}, Path(config.plot_dir) / "predictions_vs_actual_aqi.png")
        plot_error_bars(metrics_df, Path(config.plot_dir) / "error_comparison.png")
        if rolling_errors:
            plot_rolling_error(rolling_errors, Path(config.plot_dir) / "rolling_error.png")
        if correlations is not None:
            plot_correlation_heatmap(datasets["prepared_df"].drop(columns=["timestamp"]), Path(config.plot_dir) / "correlation_heatmap.png")
            plot_named_correlation_heatmap(correlations["pearson"], Path(config.plot_dir) / "pearson_correlation_heatmap.png", "Pearson Correlation Heatmap")
            plot_named_correlation_heatmap(correlations["spearman"], Path(config.plot_dir) / "spearman_correlation_heatmap.png", "Spearman Correlation Heatmap")
        plot_feature_attention(
            _comparison_feature_weights(feature_weights, baseline_metadata),
            datasets["processor"].feature_columns,
            Path(config.plot_dir) / "feature_attention.png",
        )

    save_metrics(metrics_df, per_target_metrics, streaming_metrics, streaming_switcher_models, xai_reports, baseline_metadata, config)

    _print_table("Explainable Next-Hour Model Metrics", explainable_artifacts.metrics, ["MAE", "RMSE", "NRMSE"])
    _print_table("Offline Evaluation Metrics", metrics_df, ["MAE", "RMSE", "NRMSE", "UPS"])
    print("\nPERFORMANCE COMPARISON WITH BASELINES")
    print(research_table_df.to_string(index=False))
    print(f"\nModel source: {model_source}")
    print(f"\nAdaptiveSwitcher chose: {_format_switcher_choices(switcher_models, config)}")
    _print_table("Per-Target Hybrid Metrics", pd.DataFrame(per_target_metrics["Hybrid"]).T, ["MAE", "RMSE", "NRMSE"])
    _print_table("Per-Target Adaptive Switcher Metrics", pd.DataFrame(per_target_metrics["AdaptiveSwitcher"]).T, ["MAE", "RMSE", "NRMSE"])
    if streaming_metrics:
        print("\nStreaming Adaptive Switcher Metrics")
        for metric_name, value in streaming_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print("\nStreaming Adaptive Switcher Choices")
        for target, model_name in streaming_switcher_models.items():
            print(f"{target}: {model_name}")
        print(f"\nDetected drift points: {drift_points[:10]}{'...' if len(drift_points) > 10 else ''}")
    else:
        print("\nStreaming simulation skipped")
    if correlations is not None:
        _print_correlation_summary(correlations)
        _print_granger_summary(granger_df)
    else:
        print("\nAnalytics skipped")
    _print_literature_table(config)


if __name__ == "__main__":
    main()
