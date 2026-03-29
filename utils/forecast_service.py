from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from evaluate import evaluate_models
from train import build_models
from utils.analytics import compute_correlation_matrices, granger_causality_table
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.explainable_forecasting import (
    ExplainableTimeSeriesForecaster,
    build_plain_language_summary,
    classify_forecast_levels,
    humanize_target_name,
)
def load_or_train_forecaster(datasets, config=CONFIG) -> ExplainableTimeSeriesForecaster:
    checkpoint_path = Path(config.checkpoint_dir) / "explainable_forecaster.pkl"
    if checkpoint_path.exists():
        forecaster = ExplainableTimeSeriesForecaster.load(checkpoint_path)
        if tuple(forecaster.config.target_columns) == tuple(config.target_columns):
            return forecaster

    forecaster = ExplainableTimeSeriesForecaster(config)
    forecaster.fit(datasets["prepared_df"])
    forecaster.save(checkpoint_path)
    return forecaster


def _base_context(config=CONFIG):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.random_seed)

    raw_df = load_input_dataframe(config)
    datasets = create_datasets(config, raw_df)
    prepared_df = datasets["prepared_df"]
    recent_df = prepared_df.tail(config.seq_len)
    return datasets, prepared_df, recent_df


def _round_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 4) for key, value in metrics.items()}


def _domain_pair_summaries(domain_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    correlations = compute_correlation_matrices(domain_df)
    summaries = {}
    for method, corr_df in correlations.items():
        records = []
        for left in corr_df.index:
            for right in corr_df.columns:
                if left >= right:
                    continue
                records.append(
                    {
                        "left": humanize_target_name(left) if left in CONFIG.target_columns else left.replace("_", " ").title(),
                        "right": humanize_target_name(right) if right in CONFIG.target_columns else right.replace("_", " ").title(),
                        "value": float(corr_df.loc[left, right]),
                    }
                )
        summaries[method] = sorted(records, key=lambda item: abs(item["value"]), reverse=True)[:6]
    return summaries


def _granger_summary(prepared_df: pd.DataFrame, config=CONFIG) -> List[Dict[str, Any]]:
    granger_df = granger_causality_table(prepared_df, list(config.domain_columns), max_lag=config.granger_max_lag)
    if granger_df.empty:
        return []
    significant = granger_df[granger_df["significant_5pct"]].head(6)
    if significant.empty:
        significant = granger_df.head(6)
    rows = []
    for _, row in significant.iterrows():
        rows.append(
            {
                "cause": row["cause"].replace("_", " ").title(),
                "effect": row["effect"].replace("_", " ").title(),
                "lag": None if pd.isna(row["best_lag"]) else int(row["best_lag"]),
                "p_value": round(float(row["p_value"]), 4),
                "significant": bool(row["significant_5pct"]),
            }
        )
    return rows


def _load_saved_summary(config=CONFIG) -> Dict[str, Any] | None:
    summary_path = Path(config.output_dir) / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _build_live_comparison(datasets, config=CONFIG) -> Dict[str, Any] | None:
    input_dim = datasets["train_seq"][0].shape[-1]
    models = build_models(input_dim, config)
    loaded_any = False

    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / f"{name.lower()}.pt"
        if not checkpoint.exists():
            return None
        try:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            loaded_any = True
        except RuntimeError:
            return None

    if not loaded_any:
        return None

    metrics_df, per_target_metrics, _, _, ensemble_weights, switcher_models = evaluate_models(models, datasets, config)
    return {
        "offline_metrics": metrics_df.to_dict(orient="index"),
        "per_target_metrics": per_target_metrics,
        "ensemble_weights": {key: float(value) for key, value in ensemble_weights.items()},
        "adaptive_switcher": switcher_models,
        "source": "live_checkpoints",
    }


def _build_comparison_payload(datasets, config=CONFIG) -> Dict[str, Any]:
    summary = _build_live_comparison(datasets, config) or _load_saved_summary(config)
    if not summary or "offline_metrics" not in summary:
        return {
            "available": False,
            "message": "Model comparison will appear after `python3 main.py` or `python3 generate_artifacts.py` finishes with the current 4-domain setup.",
            "models": [],
            "per_target_best": [],
            "per_target_adaptive_ensemble": [],
            "streaming_metrics": [],
            "ensemble_weights": [],
            "adaptive_switcher": [],
            "analytics": {
                "pearson_top_pairs": [],
                "spearman_top_pairs": [],
                "granger_top_links": [],
            },
        }

    metrics_df = pd.DataFrame(summary["offline_metrics"]).T
    wanted_models = ["BiLSTM", "Transformer", "Hybrid", "AdaptiveSwitcher", "AdaptiveEnsemble"]
    present_models = [name for name in wanted_models if name in metrics_df.index]
    comparison_rows = []
    for model_name in present_models:
        row = metrics_df.loc[model_name]
        comparison_rows.append(
            {
                "model": model_name,
                "mae": round(float(row.get("MAE", 0.0)), 4),
                "rmse": round(float(row.get("RMSE", 0.0)), 4),
                "nrmse": round(float(row.get("NRMSE", 0.0)), 4),
                "ups": round(float(row.get("UPS", 0.0)), 2),
            }
        )

    per_target_metrics = summary.get("per_target_metrics", {})
    best_by_target = []
    for target_name in config.target_columns:
        best_model = None
        best_score = float("inf")
        for model_name in present_models:
            target_metric = per_target_metrics.get(model_name, {}).get(target_name)
            if not target_metric:
                continue
            candidate = float(target_metric.get("NRMSE", target_metric.get("RMSE", float("inf"))))
            if candidate < best_score:
                best_score = candidate
                best_model = model_name
        best_by_target.append(
            {
                "target": humanize_target_name(target_name),
                "best_model": best_model or "Unavailable",
                "best_nrmse": None if best_score == float("inf") else round(best_score, 4),
            }
        )

    ensemble_weights = summary.get("ensemble_weights") or summary.get("final_ensemble_weights") or {}
    switcher_models = summary.get("adaptive_switcher", {})
    switcher_label = " | ".join(
        f"{humanize_target_name(target)}={model_name}"
        for target, model_name in switcher_models.items()
    )
    adaptive_ensemble_target_metrics = []
    adaptive_ensemble_metrics = summary.get("per_target_metrics", {}).get("AdaptiveEnsemble", {})
    for target_name in config.target_columns:
        target_metrics = adaptive_ensemble_metrics.get(target_name, {})
        adaptive_ensemble_target_metrics.append(
            {
                "target": humanize_target_name(target_name),
                "mae": round(float(target_metrics.get("MAE", 0.0)), 4),
                "rmse": round(float(target_metrics.get("RMSE", 0.0)), 4),
                "nrmse": round(float(target_metrics.get("NRMSE", 0.0)), 4),
            }
        )

    streaming_metrics = [
        {"metric": key, "value": round(float(value), 4)}
        for key, value in (summary.get("streaming_metrics") or {}).items()
    ]
    domain_df = datasets["prepared_df"][list(config.domain_columns)]
    correlation_pairs = _domain_pair_summaries(domain_df)
    return {
        "available": True,
        "message": "Comparison is built from the latest saved training artifacts." if summary.get("source") != "live_checkpoints" else "Comparison is built from the current checkpoints.",
        "models": comparison_rows,
        "per_target_best": best_by_target,
        "per_target_adaptive_ensemble": adaptive_ensemble_target_metrics,
        "streaming_metrics": streaming_metrics,
        "ensemble_weights": [{"model": key, "weight": round(float(value), 4)} for key, value in ensemble_weights.items()],
        "adaptive_switcher_label": switcher_label,
        "adaptive_switcher": [
            {
                "target": humanize_target_name(target),
                "model": model_name,
                "description": f"Selected base model: {model_name}",
            }
            for target, model_name in switcher_models.items()
        ],
        "analytics": {
            "pearson_top_pairs": correlation_pairs["pearson"],
            "spearman_top_pairs": correlation_pairs["spearman"],
            "granger_top_links": _granger_summary(datasets["prepared_df"], config),
        },
    }


def build_forecast_payload(config=CONFIG) -> Dict[str, Any]:
    datasets, prepared_df, recent_df = _base_context(config)
    latest = recent_df.iloc[-1]

    forecaster = load_or_train_forecaster(datasets, config)
    explainable_artifacts = forecaster.fit(prepared_df)
    forecast = forecaster.predict_from_prepared(prepared_df)
    explanations = forecaster.explain_latest_prediction(prepared_df, top_k=2)

    point_prediction = {name: values["prediction"] for name, values in forecast.items()}
    labels = classify_forecast_levels(point_prediction, recent_df)
    summary_lines = build_plain_language_summary(point_prediction, recent_df)

    time_step = recent_df["timestamp"].diff().dropna().mode().iloc[0]
    next_timestamp = prepared_df["timestamp"].iloc[-1] + time_step

    metrics = []
    for key in ("traffic_flow", "aqi", "temperature", "electricity_demand"):
        metric_forecast = forecast[key]
        current_value = float(latest[key])
        predicted_value = float(metric_forecast["prediction"])
        change_value = predicted_value - current_value
        metrics.append(
            {
                "key": key,
                "label": humanize_target_name(key),
                "current": round(current_value, 2),
                "prediction": round(predicted_value, 2),
                "lower": round(float(metric_forecast["lower"]), 2),
                "upper": round(float(metric_forecast["upper"]), 2),
                "change": round(change_value, 2),
                "status": labels[key],
                "explanations": explanations[key],
            }
        )

    explainable_rows = []
    for target_name, row in explainable_artifacts.metrics.iterrows():
        explainable_rows.append(
            {
                "target": humanize_target_name(target_name),
                **_round_metrics(row.to_dict()),
            }
        )

    domain_df = prepared_df[list(config.domain_columns)]
    correlation_pairs = _domain_pair_summaries(domain_df)

    return {
        "forecast_for": next_timestamp.isoformat(),
        "last_updated": prepared_df["timestamp"].iloc[-1].isoformat(),
        "summary": summary_lines,
        "metrics": metrics,
        "explainable_metrics": explainable_rows,
        "analytics": {
            "pearson_top_pairs": correlation_pairs["pearson"],
            "spearman_top_pairs": correlation_pairs["spearman"],
            "granger_top_links": _granger_summary(prepared_df, config),
        },
    }


def build_comparison_payload(config=CONFIG) -> Dict[str, Any]:
    datasets, _, _ = _base_context(config)
    return _build_comparison_payload(datasets, config)
