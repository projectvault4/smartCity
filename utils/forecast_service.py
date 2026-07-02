from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate_models
from train import build_models, checkpoint_name_for_model
from utils.analytics import compute_correlation_matrices, granger_causality_table
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.explainable_forecasting import (
    ExplainableTimeSeriesForecaster,
    build_plain_language_summary,
    classify_forecast_levels,
    humanize_target_name,
)
from utils.project_best import latest_project_best_prediction, select_finalized_forecast_model


def _load_literature_models(config=CONFIG) -> List[Dict[str, Any]]:
    literature_path = Path(config.data_dir) / "literature_models.json"
    if not literature_path.exists():
        return []
    try:
        with open(literature_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []


def _safe_iso_date(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).date().isoformat()


def _latest_bangalore_aqi_date(dataset_dir: Path) -> str | None:
    try:
        from utils.data_utils import _load_bangalore_aqi_daily

        aqi_df = _load_bangalore_aqi_daily(dataset_dir)
    except Exception:
        return None
    if aqi_df.empty:
        return None
    return _safe_iso_date(aqi_df["timestamp"].max())


def _latest_bangalore_aqi_filename_date(dataset_dir: Path) -> str | None:
    latest = None
    month_pattern = (
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"[-\s]+(\d{4})"
    )
    for path in [*dataset_dir.glob("*AQI*.xls"), *dataset_dir.glob("*data for Bengaluru*.xls")]:
        match = re.search(month_pattern, path.name, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = pd.to_datetime(f"{match.group(1)} {match.group(2)}", errors="coerce")
        if pd.isna(parsed):
            continue
        month_end = pd.Timestamp(parsed) + pd.offsets.MonthEnd(0)
        latest = month_end if latest is None else max(latest, month_end)
    return _safe_iso_date(latest)


def _latest_bangalore_electricity_date(power_dir: Path) -> str | None:
    latest = None
    for path in power_dir.glob("ALLOCATIONVSACTUAL*.xlsx"):
        match = re.search(r"(\d{2}-\d{2}-\d{4})", path.name)
        if not match:
            continue
        parsed = pd.to_datetime(match.group(1), dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            continue
        latest = parsed if latest is None else max(latest, parsed)
    return _safe_iso_date(latest)


def _find_latest_power_dir(dataset_dir: Path) -> Path | None:
    latest_year = -1
    latest_dir = None
    for path in dataset_dir.glob("BESCOM_*_LoadCurves"):
        match = re.search(r"BESCOM_(\d{4})_LoadCurves", path.name)
        if match:
            year = int(match.group(1))
            if year > latest_year:
                latest_year = year
                latest_dir = path
    return latest_dir


def _build_data_freshness(config=CONFIG, prepared_df: pd.DataFrame | None = None) -> Dict[str, Any]:
    latest_prepared = None
    if prepared_df is not None and not prepared_df.empty:
        latest_prepared = _safe_iso_date(prepared_df["timestamp"].max())

    if getattr(config, "city", "default") != "bangalore":
        return {
            "latest_prepared": latest_prepared,
            "sources": [],
            "limiting_source": None,
            "note": None,
        }

    dataset_dir = Path(config.dataset_dir)
    traffic_path = dataset_dir / "Banglore_traffic_Dataset.csv"
    weather_path = dataset_dir / "export.csv"
    power_dir = _find_latest_power_dir(dataset_dir)
    sources: list[dict[str, str | None]] = []

    traffic_latest = None
    if traffic_path.exists():
        try:
            traffic_latest = _safe_iso_date(pd.read_csv(traffic_path, usecols=["Date"])["Date"].max())
        except Exception:
            traffic_latest = None
    sources.append({"name": "Traffic", "latest": traffic_latest})

    weather_latest = None
    if weather_path.exists():
        try:
            weather_latest = _safe_iso_date(pd.read_csv(weather_path, usecols=["date"])["date"].max())
        except Exception:
            weather_latest = None
    sources.append({"name": "Weather", "latest": weather_latest})
    sources.append({"name": "AQI", "latest": _latest_bangalore_aqi_date(dataset_dir) or _latest_bangalore_aqi_filename_date(dataset_dir)})
    sources.append({"name": "Electricity", "latest": _latest_bangalore_electricity_date(power_dir) if power_dir else None})

    dated_sources = [source for source in sources if source["latest"]]
    limiting_source = min(dated_sources, key=lambda source: source["latest"]) if dated_sources else None
    note = None
    if latest_prepared:
        note = (
            f"Bangalore forecast data is historical and currently stops at {latest_prepared}. "
            "The next-hour forecast is generated one step after this loaded data point."
        )
        if limiting_source:
            note += f" The merged dataset is limited by the {limiting_source['name']} source."

    return {
        "latest_prepared": latest_prepared,
        "sources": sources,
        "limiting_source": limiting_source["name"] if limiting_source else None,
        "note": note,
    }


def _ordered_model_names(metrics_df: pd.DataFrame, literature_rows: List[Dict[str, Any]]) -> List[str]:
    literature_names = [item["model"] for item in literature_rows]
    present_names = list(metrics_df.index)
    ordered = [name for name in literature_names if name in present_names]
    ordered.extend(name for name in present_names if name not in ordered)
    return ordered


def _annotate_literature_rows(literature_rows: List[Dict[str, Any]], metrics_df: pd.DataFrame) -> List[Dict[str, Any]]:
    evaluated_models = set(metrics_df.index)
    enriched = []
    for row in literature_rows:
        updated = dict(row)
        if updated.get("model") in evaluated_models:
            updated["comparison_scope"] = "Evaluated in this project"
        enriched.append(updated)
    return enriched


def _direction_word(delta: float, tolerance: float = 1.0) -> str:
    if delta > tolerance:
        return "increase"
    if delta < -tolerance:
        return "decrease"
    return "stay nearly stable"


def _build_interconnected_summary(metrics: list[dict]) -> list[str]:
    metric_map = {metric["key"]: metric for metric in metrics}
    traffic_change = float(metric_map["traffic_flow"]["change"])
    aqi_change = float(metric_map["aqi"]["change"])
    temperature_change = float(metric_map["temperature"]["change"])
    electricity_change = float(metric_map["electricity_demand"]["change"])

    return [
        (
            f"Traffic Flow is expected to {_direction_word(traffic_change)}, "
            f"which can make AQI {_direction_word(aqi_change)} through higher road emissions and congestion."
        ),
        (
            f"AQI is expected to {_direction_word(aqi_change)}, "
            f"which can make Temperature {_direction_word(temperature_change)} in the model's learned cross-domain pattern."
        ),
        (
            f"Temperature is expected to {_direction_word(temperature_change)}, "
            f"which can make Electricity Demand {_direction_word(electricity_change)} because cooling needs usually change with temperature."
        ),
        (
            f"Overall chain: Traffic Flow -> AQI -> Temperature -> Electricity Demand, "
            f"with the next-hour forecast reflecting how these signals move together in the dataset."
        ),
    ]


def _estimate_next_humidity(raw_df: pd.DataFrame, next_temperature: float) -> float:
    recent = raw_df["humidity"].tail(24)
    latest_humidity = float(recent.iloc[-1])
    recent_mean = float(recent.mean())
    latest_temperature = float(raw_df["temperature"].iloc[-1])
    temperature_delta = next_temperature - latest_temperature
    estimated = 0.65 * latest_humidity + 0.35 * recent_mean - 0.45 * temperature_delta
    return float(np.clip(estimated, 5.0, 100.0))


def _estimate_future_humidity(raw_df: pd.DataFrame, step_ahead: int, next_temperature: float) -> float:
    history = raw_df["humidity"].tail(24 * 7)
    base = float(history.iloc[-1])
    hour = (pd.to_datetime(raw_df["timestamp"].iloc[-1]) + pd.Timedelta(hours=step_ahead)).hour
    hourly_shape = history.groupby(raw_df["timestamp"].tail(len(history)).dt.hour).mean()
    hour_baseline = float(hourly_shape.get(hour, history.mean()))
    temperature_delta = next_temperature - float(raw_df["temperature"].iloc[-1])
    estimated = 0.55 * base + 0.45 * hour_baseline - 0.35 * temperature_delta
    return float(np.clip(estimated, 5.0, 100.0))


def _bounded_target_value(target: str, value: float) -> float:
    if target == "aqi":
        return float(np.clip(value, 0.0, 500.0))
    if target in {"traffic_flow", "electricity_demand"}:
        return float(max(0.0, value))
    return float(value)


def _bounded_forecast_item(target: str, item: dict) -> dict:
    prediction = _bounded_target_value(target, float(item["prediction"]))
    lower = _bounded_target_value(target, float(item["lower"]))
    upper = _bounded_target_value(target, float(item["upper"]))
    if lower > upper:
        lower, upper = upper, lower
    prediction = float(np.clip(prediction, lower, upper)) if lower <= upper else prediction
    return {**item, "prediction": prediction, "lower": lower, "upper": upper}


def _uncertainty_descriptor(interval_width: float, recent_series: pd.Series) -> tuple[str, str]:
    reference = float(recent_series.tail(24).std(ddof=0))
    if reference <= 0:
        reference = max(abs(float(recent_series.tail(24).mean())) * 0.05, 1.0)
    ratio = interval_width / reference
    if ratio <= 0.75:
        return "High", "the expected range is narrow compared with recent movement"
    if ratio <= 1.35:
        return "Medium", "the expected range is moderate compared with recent movement"
    return "Low", "the expected range is wide, so the next step is less certain than usual"


def _apply_project_best_override(forecast: dict[str, dict], config=CONFIG) -> tuple[dict[str, dict], str | None, str | None]:
    model_name = select_finalized_forecast_model(config)
    if not model_name:
        return forecast, None, None

    try:
        override = latest_project_best_prediction(model_name, config)
    except Exception:
        return forecast, None, None

    adjusted = {}
    for key, values in forecast.items():
        tuned_prediction = float(override["prediction"][key])
        delta = tuned_prediction - float(values["prediction"])
        adjusted[key] = _bounded_forecast_item(
            key,
            {
                **values,
                "prediction": tuned_prediction,
                "lower": float(values["lower"]) + delta,
                "upper": float(values["upper"]) + delta,
            },
        )
    return adjusted, model_name, override["timestamp"].isoformat()


def build_past_present_future_frame(config=CONFIG, future_steps: int = 24) -> pd.DataFrame:
    raw_df = load_input_dataframe(config).copy()
    datasets = create_datasets(config, raw_df)
    prepared_df = datasets["prepared_df"]
    forecaster = load_or_train_forecaster(datasets, config)
    forecaster.fit(prepared_df)
    horizons = list(range(1, future_steps + 1))
    forecaster.fit_multi_horizon(prepared_df, horizons)
    direct_forecasts = forecaster.predict_multi_horizon(prepared_df, horizons)

    timeline = raw_df[["timestamp", "traffic_flow", "aqi", "temperature", "humidity", "electricity_demand"]].copy()
    timeline["time_segment"] = "past"
    timeline.loc[timeline.index[-1], "time_segment"] = "present"

    time_step = raw_df["timestamp"].diff().dropna().mode().iloc[0]
    future_rows: list[dict[str, Any]] = []

    for step_ahead in range(1, future_steps + 1):
        forecast = {
            key: _bounded_forecast_item(key, values)
            for key, values in direct_forecasts[step_ahead].items()
        }
        next_timestamp = raw_df["timestamp"].iloc[-1] + step_ahead * time_step
        next_temperature = float(forecast["temperature"]["prediction"])
        next_row = {
            "timestamp": next_timestamp,
            "traffic_flow": float(forecast["traffic_flow"]["prediction"]),
            "aqi": float(forecast["aqi"]["prediction"]),
            "temperature": next_temperature,
            "humidity": _estimate_future_humidity(raw_df, step_ahead, next_temperature),
            "electricity_demand": float(forecast["electricity_demand"]["prediction"]),
            "time_segment": "future",
            "step_ahead": step_ahead,
        }
        future_rows.append(next_row)

    timeline["step_ahead"] = 0
    future_df = pd.DataFrame(future_rows)
    return pd.concat([timeline, future_df], ignore_index=True)


def load_or_train_forecaster(datasets, config=CONFIG) -> ExplainableTimeSeriesForecaster:
    checkpoint_path = Path(config.checkpoint_dir) / "explainable_forecaster.pkl"
    if checkpoint_path.exists():
        try:
            forecaster = ExplainableTimeSeriesForecaster.load(checkpoint_path)
            if tuple(forecaster.config.target_columns) == tuple(config.target_columns):
                return forecaster
        except (OSError, pickle.PickleError, AttributeError, ValueError, EOFError, TypeError, ModuleNotFoundError):
            pass

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
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    models = build_models(input_dim, config)
    loaded_any = False

    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / f"{checkpoint_name_for_model(name)}.pt"
        if not checkpoint.exists():
            return None
        try:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            loaded_any = True
        except RuntimeError:
            return None

    if not loaded_any:
        return None

    metrics_df, per_target_metrics, _, _, switcher_models = evaluate_models(models, datasets, config)
    return {
        "offline_metrics": metrics_df.to_dict(orient="index"),
        "per_target_metrics": per_target_metrics,
        "streaming_model": "AdaptiveSwitcher",
        "adaptive_switcher": switcher_models,
        "source": "live_checkpoints",
    }


def _build_comparison_payload(datasets, config=CONFIG) -> Dict[str, Any]:
    live_summary = _build_live_comparison(datasets, config)
    saved_summary = _load_saved_summary(config)
    if saved_summary and len(saved_summary.get("offline_metrics", {})) > len((live_summary or {}).get("offline_metrics", {})):
        summary = saved_summary
    else:
        summary = live_summary or saved_summary
    if not summary or "offline_metrics" not in summary:
        return {
            "available": False,
            "message": "Model comparison will appear after `python3 main.py` or `python3 generate_artifacts.py` finishes with the current BiLSTM + TFT setup.",
            "models": [],
            "literature_models": [],
            "per_target_best": [],
            "streaming_metrics": [],
            "streaming_model": "",
            "analytics": {
                "pearson_top_pairs": [],
                "spearman_top_pairs": [],
                "granger_top_links": [],
            },
        }

    metrics_df = pd.DataFrame(summary["offline_metrics"]).T
    literature_rows = _load_literature_models(config)
    present_models = _ordered_model_names(metrics_df, literature_rows)
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
    literature_rows = _annotate_literature_rows(literature_rows, metrics_df)
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

    streaming_metrics = [
        {"metric": key, "value": round(float(value), 4)}
        for key, value in (summary.get("streaming_metrics") or {}).items()
    ]
    switcher_models = summary.get("adaptive_switcher", {})
    switcher_label = " | ".join(
        f"{humanize_target_name(target)}={model_name}"
        for target, model_name in switcher_models.items()
    )
    domain_df = datasets["prepared_df"][list(config.domain_columns)]
    correlation_pairs = _domain_pair_summaries(domain_df)
    return {
        "available": True,
        "message": "Comparison is built from the latest saved training artifacts." if summary.get("source") != "live_checkpoints" else "Comparison is built from the current checkpoints.",
        "models": comparison_rows,
        "literature_models": literature_rows,
        "per_target_best": best_by_target,
        "streaming_metrics": streaming_metrics,
        "streaming_model": summary.get("streaming_model", "AdaptiveSwitcher"),
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
    forecast = {key: _bounded_forecast_item(key, values) for key, values in forecast.items()}
    forecast, tuned_model_source, tuned_timestamp = _apply_project_best_override(forecast, config)
    explanations = forecaster.explain_latest_prediction(prepared_df, top_k=2)

    point_prediction = {name: values["prediction"] for name, values in forecast.items()}
    labels = classify_forecast_levels(point_prediction, recent_df)
    summary_lines = build_plain_language_summary(point_prediction, recent_df)

    time_step = recent_df["timestamp"].diff().dropna().mode().iloc[0]
    next_timestamp = pd.Timestamp(tuned_timestamp) if tuned_timestamp else prepared_df["timestamp"].iloc[-1] + time_step

    metrics = []
    for key in ("traffic_flow", "aqi", "temperature", "electricity_demand"):
        metric_forecast = forecast[key]
        current_value = float(latest[key])
        predicted_value = float(metric_forecast["prediction"])
        change_value = predicted_value - current_value
        interval_width = float(metric_forecast["upper"]) - float(metric_forecast["lower"])
        confidence_label, uncertainty_reason = _uncertainty_descriptor(interval_width, prepared_df[key])
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
                "confidence": confidence_label,
                "uncertainty_reason": uncertainty_reason,
                "explanations": explanations[key],
            }
        )
    interconnected_summary = _build_interconnected_summary(metrics)

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
        "data_freshness": _build_data_freshness(config, prepared_df),
        "point_forecast_model": tuned_model_source or "ExplainableTimeSeriesForecaster",
        "summary": summary_lines,
        "metrics": metrics,
        "interconnected_summary": interconnected_summary,
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
