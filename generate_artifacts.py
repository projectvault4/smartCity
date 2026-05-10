from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / "cache"))

from evaluate import evaluate_models
from train import build_models
from utils.analytics import compute_correlation_matrices, granger_causality_table, save_analysis_tables
from utils.baselines import evaluate_baselines
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, rolling_mean, set_seed
from utils.explainable_forecasting import ExplainableTimeSeriesForecaster
from utils.paper_artifacts import (
    evaluate_ablation_study,
    load_neural_models_for_robustness,
    run_reliability_study,
    save_efficiency_table,
    save_fair_baseline_note,
    save_forecast_and_residual_plots,
    save_per_target_tables,
    save_robustness_results,
    save_split_protocol,
    train_or_load_static_hybrid_ablation,
)
from utils.visualization import (
    plot_correlation_heatmap,
    plot_error_bars,
    plot_feature_attention,
    plot_named_correlation_heatmap,
    plot_predictions,
    plot_rolling_error,
)
from utils.xai import save_explainability_report


def parse_args():
    parser = argparse.ArgumentParser(description="Generate core and paper-ready forecasting artifacts.")
    parser.add_argument(
        "--with-reliability",
        action="store_true",
        help="Run the slower multi-seed reliability study and save mean/std summaries.",
    )
    parser.add_argument(
        "--reliability-seeds",
        default="42,52,62",
        help="Comma-separated seeds for the reliability study.",
    )
    return parser.parse_args()


def _load_literature_models(config) -> list[dict]:
    literature_path = Path(config.data_dir) / "literature_models.json"
    if not literature_path.exists():
        return []
    with open(literature_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _print_literature_table(config) -> None:
    literature_rows = _load_literature_models(config)
    if not literature_rows:
        print("\nLiterature Reference Models")
        print("No literature model table found.")
        return

    literature_df = pd.DataFrame(literature_rows)
    visible_columns = ["model", "family", "year", "comparison_scope", "source_name"]
    print("\nLiterature Reference Models")
    print(literature_df[visible_columns].to_string(index=False))


def _ordered_model_names(config, metrics_df: pd.DataFrame) -> list[str]:
    literature_rows = _load_literature_models(config)
    literature_names = [item["model"] for item in literature_rows]
    present_names = list(metrics_df.index)
    ordered = [name for name in literature_names if name in present_names]
    ordered.extend(name for name in present_names if name not in ordered)
    return ordered


def _sort_metrics_frame(config, metrics_df: pd.DataFrame) -> pd.DataFrame:
    ordered_names = _ordered_model_names(config, metrics_df)
    return metrics_df.loc[ordered_names]


def _build_paper_style_comparison(config, metrics_df: pd.DataFrame, predictions: dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    literature_rows = _load_literature_models(config)
    rows = []
    seen_models = set()

    for item in literature_rows:
        model_name = item["model"]
        seen_models.add(model_name)
        if model_name not in metrics_df.index:
            continue

        row = metrics_df.loc[model_name]
        rows.append(
            {
                "Model": model_name,
                "MAE": f"{float(row['MAE']):.2f}",
                "MAPE": f"{float(row['MAPE']):.2f}%",
                "RMSE": f"{float(row['RMSE']):.2f}",
            }
        )

    for model_name in metrics_df.index:
        if model_name in seen_models:
            continue
        row = metrics_df.loc[model_name]
        rows.append(
            {
                "Model": model_name,
                "MAE": f"{float(row['MAE']):.2f}",
                "MAPE": f"{float(row['MAPE']):.2f}%",
                "RMSE": f"{float(row['RMSE']):.2f}",
            }
        )

    return pd.DataFrame(rows)


def _plot_friendly_predictions(all_predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    preferred = ["ARIMA", "SARIMA", "LSTM", "GRU", "BiLSTM", "TFT", "Informer", "PatchTST", "Hybrid", "AdaptiveSwitcher"]
    selected = [name for name in preferred if name in all_predictions]
    return {name: all_predictions[name] for name in selected}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tft_gru_hybrid_artifact(config) -> tuple[dict | None, dict | None, np.ndarray | None]:
    metrics_path = Path(config.output_dir) / "tft_gru_residual_hybrid_metrics.json"
    if not metrics_path.exists():
        return None, None, None

    artifact = _load_json(metrics_path)
    predictions = None
    prediction_path = Path(artifact.get("prediction_path", Path(config.output_dir) / "tft_gru_residual_hybrid_predictions.csv"))
    if prediction_path.exists():
        prediction_df = pd.read_csv(prediction_path)
        prediction_columns = [f"predicted_{target}" for target in config.target_columns]
        if all(column in prediction_df.columns for column in prediction_columns):
            predictions = prediction_df[prediction_columns].to_numpy(dtype=float)

    return artifact.get("metrics"), artifact.get("per_target_metrics"), predictions


def _overlay_tft_gru_hybrid_artifact(
    config,
    core_metrics_df: pd.DataFrame,
    all_metrics_df: pd.DataFrame,
    all_per_target_metrics: dict,
    all_predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, np.ndarray], bool]:
    metrics, per_target_metrics, predictions = _load_tft_gru_hybrid_artifact(config)
    if metrics is None or per_target_metrics is None:
        return core_metrics_df, all_metrics_df, all_per_target_metrics, all_predictions, False

    for frame in (core_metrics_df, all_metrics_df):
        for metric_name, metric_value in metrics.items():
            frame.loc["Hybrid", metric_name] = metric_value
    all_per_target_metrics["Hybrid"] = per_target_metrics
    if predictions is not None:
        all_predictions["Hybrid"] = predictions
    return core_metrics_df, all_metrics_df, all_per_target_metrics, all_predictions, True


def main():
    args = parse_args()
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

    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    models = build_models(input_dim, config)
    can_use_legacy_checkpoints = True
    legacy_checkpoint_error = None
    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / ("hybrid.pt" if name == "Hybrid" else f"{name.lower()}.pt")
        if not checkpoint.exists():
            can_use_legacy_checkpoints = False
            legacy_checkpoint_error = (
                f"Missing checkpoint: {checkpoint}. Run `python3 main.py` first to train models on the current dataset."
            )
            break
        try:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        except RuntimeError as exc:
            can_use_legacy_checkpoints = False
            legacy_checkpoint_error = str(exc)
            break

    baseline_metrics_df, baseline_per_target_metrics, baseline_predictions, baseline_metadata = evaluate_baselines(datasets, config)
    literature_rows = _load_literature_models(config)
    used_finalized_fallback = False

    if can_use_legacy_checkpoints:
        core_metrics_df, per_target_metrics, predictions, feature_weights, switcher_models = evaluate_models(models, datasets, config)
        all_predictions = {**predictions, **baseline_predictions}
        all_per_target_metrics = {**per_target_metrics, **baseline_per_target_metrics}
        all_metrics_df = pd.concat([core_metrics_df, baseline_metrics_df], axis=0)
        all_metrics_df = _sort_metrics_frame(config, all_metrics_df)
    else:
        summary_path = Path(config.output_dir) / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(
                "Checkpoint architecture does not match the current Hybrid model, and no finalized summary is available. "
                "Run `python3 finalize_project_best_outputs.py` first, or retrain legacy checkpoints with `python3 main.py`."
            )
        finalized_summary = _load_json(summary_path)
        offline_metrics = finalized_summary.get("offline_metrics")
        per_target_metrics = finalized_summary.get("per_target_metrics")
        if not offline_metrics or not per_target_metrics:
            raise RuntimeError(
                "outputs/summary.json does not contain finalized metrics needed for artifact fallback. "
                "Run `python3 finalize_project_best_outputs.py` first."
            )
        core_model_names = ["BiLSTM", "TFT", "Hybrid", "Informer", "PatchTST"]
        all_metrics_df = pd.DataFrame(offline_metrics).T
        all_metrics_df = _sort_metrics_frame(config, all_metrics_df)
        core_metrics_df = all_metrics_df.loc[[name for name in core_model_names if name in all_metrics_df.index]].copy()
        all_per_target_metrics = per_target_metrics
        predictions = {}
        all_predictions = dict(baseline_predictions)
        feature_weights = {}
        switcher_models = finalized_summary.get("adaptive_switcher", {})
        used_finalized_fallback = True

    y_true = datasets["processor"].inverse_transform_targets(datasets["test_tpt"]["target"])
    core_metrics_df, all_metrics_df, all_per_target_metrics, all_predictions, used_tft_gru_hybrid = (
        _overlay_tft_gru_hybrid_artifact(
            config,
            core_metrics_df,
            all_metrics_df,
            all_per_target_metrics,
            all_predictions,
        )
    )
    all_metrics_df = _sort_metrics_frame(config, all_metrics_df)
    research_table_df = _build_paper_style_comparison(config, all_metrics_df, all_predictions, y_true)
    error_traces = (
        {
            name: rolling_mean(np.mean(np.abs(y_true - pred), axis=1).tolist(), 24)
            for name, pred in all_predictions.items()
        }
        if all_predictions
        else {}
    )

    core_metrics_df.to_csv(Path(config.output_dir) / "core_model_metrics.csv")
    baseline_metrics_df.to_csv(Path(config.output_dir) / "baseline_metrics.csv")
    all_metrics_df.to_csv(Path(config.output_dir) / "metrics.csv")
    final_best_df = (
        all_metrics_df.sort_values(["RMSE", "MAE", "UPS"], ascending=[True, True, False])
        .reset_index()
        .rename(columns={"index": "model"})
        .head(4)
    )
    final_best_df.to_csv(Path(config.output_dir) / "final_best_models.csv", index=False)
    research_table_df.to_csv(Path(config.output_dir) / "research_performance_comparison.csv", index=False)
    save_per_target_tables(all_per_target_metrics, config, Path(config.output_dir), literature_rows)
    save_split_protocol(datasets, config, Path(config.output_dir))
    save_fair_baseline_note(literature_rows, all_metrics_df.index.tolist(), Path(config.output_dir))

    if not used_finalized_fallback:
        ablation_df, ablation_per_target, ablation_metadata = evaluate_ablation_study(
            core_metrics_df,
            per_target_metrics,
            datasets,
            config,
            Path(config.output_dir),
        )
        save_per_target_tables(ablation_per_target, config, Path(config.output_dir), [], prefix="ablation")

        robustness_models = load_neural_models_for_robustness(datasets, config)
        hybrid_no_gate_model, hybrid_no_gate_training_seconds = train_or_load_static_hybrid_ablation(datasets, config)
        robustness_models["HybridNoGate"] = hybrid_no_gate_model
        save_robustness_results(robustness_models, datasets, config, Path(config.output_dir))

        efficiency_models = dict(robustness_models)
        save_efficiency_table(
            efficiency_models,
            datasets,
            config,
            Path(config.output_dir),
            extra_training_times={"HybridNoGate": hybrid_no_gate_training_seconds},
        )
    else:
        ablation_df = pd.DataFrame()
        ablation_metadata = {
            "skipped": True,
            "reason": "Legacy checkpoints are incompatible with the current Hybrid architecture.",
            "legacy_checkpoint_error": legacy_checkpoint_error,
        }

    with open(Path(config.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "offline_metrics": all_metrics_df.to_dict(orient="index"),
                "per_target_metrics": all_per_target_metrics,
                "streaming_model": "AdaptiveSwitcher",
                "adaptive_switcher": switcher_models,
                "xai_reports": xai_reports,
                "baseline_metadata": baseline_metadata,
                "ablation_metadata": ablation_metadata,
                "hybrid_artifact_override": {
                    "used": used_tft_gru_hybrid,
                    "source": "outputs/tft_gru_residual_hybrid_metrics.json",
                    "model": "TFTGRUResidualHybrid",
                },
            },
            f,
            indent=2,
        )

    aqi_idx = list(config.target_columns).index("aqi")
    if all_predictions:
        plot_predictions(
            y_true[:, aqi_idx],
            {name: pred[:, aqi_idx] for name, pred in _plot_friendly_predictions(all_predictions).items()},
            Path(config.plot_dir) / "predictions_vs_actual_aqi.png",
        )
    plot_error_bars(all_metrics_df, Path(config.plot_dir) / "error_comparison.png")
    if error_traces:
        plot_rolling_error(error_traces, Path(config.plot_dir) / "rolling_error.png")
    plot_correlation_heatmap(datasets["prepared_df"].drop(columns=["timestamp"]), Path(config.plot_dir) / "correlation_heatmap.png")
    if feature_weights.get("Hybrid") is not None:
        plot_feature_attention(feature_weights.get("Hybrid"), datasets["processor"].feature_columns, Path(config.plot_dir) / "feature_attention.png")
    correlations = compute_correlation_matrices(datasets["prepared_df"].drop(columns=["timestamp"]))
    granger_df = granger_causality_table(datasets["prepared_df"], list(config.domain_columns), max_lag=config.granger_max_lag)
    save_analysis_tables(correlations, granger_df, Path(config.output_dir))
    plot_named_correlation_heatmap(correlations["pearson"], Path(config.plot_dir) / "pearson_correlation_heatmap.png", "Pearson Correlation Heatmap")
    plot_named_correlation_heatmap(correlations["spearman"], Path(config.plot_dir) / "spearman_correlation_heatmap.png", "Spearman Correlation Heatmap")
    if all_predictions:
        save_forecast_and_residual_plots(y_true, all_predictions, config, Path(config.plot_dir))

    if args.with_reliability:
        reliability_seeds = [int(value.strip()) for value in args.reliability_seeds.split(",") if value.strip()]
        run_reliability_study(raw_df, config, Path(config.output_dir), reliability_seeds)

    print(all_metrics_df.round(4).to_string())
    print("\nPERFORMANCE COMPARISON WITH BASELINES")
    print(research_table_df.to_string(index=False))
    if used_finalized_fallback:
        print("\nArtifact Fallback Mode")
        print("Used finalized project-best metrics because legacy checkpoints do not match the current Hybrid architecture.")
        print("Skipped ablation, robustness, and efficiency regeneration that depends on legacy-compatible checkpoints.")
    if used_tft_gru_hybrid:
        print("\nHybrid Artifact Override")
        print("Used outputs/tft_gru_residual_hybrid_metrics.json as the canonical Hybrid row.")
    if not used_finalized_fallback:
        print("\nABLATION STUDY")
        print(ablation_df.round(4).to_string(index=False))
    _print_literature_table(config)


if __name__ == "__main__":
    main()
