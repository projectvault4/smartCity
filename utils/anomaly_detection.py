from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class AnomalyDetectionResult:
    events: list[dict[str, Any]]
    timeline: pd.DataFrame
    thresholds: dict[str, float]
    feature_columns: list[str]
    feature_frame: pd.DataFrame


def _numeric_domain_columns(df: pd.DataFrame, config) -> list[str]:
    return [col for col in config.domain_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def _robust_threshold(values: np.ndarray, percentile: float = 95.0) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust = median + 3.0 * 1.4826 * mad
    quantile = float(np.percentile(values, percentile))
    return max(robust, quantile)


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if np.isclose(float(values.max()), float(values.min())):
        return np.zeros(values.shape[0], dtype=float)
    return MinMaxScaler().fit_transform(values).ravel()


def prepare_anomaly_feature_frame(raw_df: pd.DataFrame, config) -> tuple[pd.DataFrame, list[str]]:
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    domain_columns = _numeric_domain_columns(df, config)
    df[domain_columns] = df[domain_columns].apply(pd.to_numeric, errors="coerce")
    df = df.set_index("timestamp")
    df[domain_columns] = df[domain_columns].interpolate(method="time", limit_direction="both").ffill().bfill()
    df = df.reset_index()

    ts = df["timestamp"]
    feature_df = pd.DataFrame({"timestamp": ts})
    for col in domain_columns:
        feature_df[col] = df[col]
        feature_df[f"{col}_diff_1"] = df[col].diff()
        feature_df[f"{col}_pct_change_1"] = df[col].pct_change().replace([np.inf, -np.inf], np.nan)
        for lag in (1, 2, 3, 6, 12, 24):
            feature_df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        for window in (6, 12, 24):
            rolling = df[col].rolling(window=window, min_periods=max(3, window // 2))
            mean = rolling.mean()
            std = rolling.std().replace(0, np.nan)
            feature_df[f"{col}_roll_mean_{window}"] = mean
            feature_df[f"{col}_roll_std_{window}"] = std
            feature_df[f"{col}_roll_z_{window}"] = (df[col] - mean) / std

    feature_df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    feature_df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    feature_df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    feature_df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    feature_df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    feature_columns = [col for col in feature_df.columns if col != "timestamp"]
    feature_df[feature_columns] = feature_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    feature_df[feature_columns] = feature_df[feature_columns].ffill().bfill().fillna(0.0)
    return feature_df, feature_columns


def select_anomaly_features(feature_df: pd.DataFrame, feature_columns: list[str], max_features: int = 48) -> list[str]:
    values = feature_df[feature_columns].to_numpy(dtype=float)
    keep_mask = VarianceThreshold(threshold=1e-8).fit(values).get_support()
    kept = [col for col, keep in zip(feature_columns, keep_mask) if keep]
    if len(kept) <= max_features:
        return kept

    corr = feature_df[kept].corr().abs().fillna(0.0)
    ranked = feature_df[kept].std(ddof=0).sort_values(ascending=False).index.tolist()
    selected: list[str] = []
    for candidate in ranked:
        if all(float(corr.loc[candidate, existing]) < 0.96 for existing in selected):
            selected.append(candidate)
        if len(selected) >= max_features:
            break
    return selected


def _train_autoencoder_scores(x_scaled: np.ndarray, random_seed: int) -> np.ndarray:
    n_features = x_scaled.shape[1]
    hidden = max(4, min(32, n_features // 2))
    bottleneck = max(2, min(12, n_features // 4))
    model = MLPRegressor(
        hidden_layer_sizes=(hidden, bottleneck, hidden),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=220,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=random_seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_scaled, x_scaled)
    reconstructed = model.predict(x_scaled)
    return np.mean(np.square(x_scaled - reconstructed), axis=1)


def _classify_severity(score: float) -> str:
    if score >= 0.9:
        return "Critical"
    if score >= 0.7:
        return "High"
    if score >= 0.5:
        return "Medium"
    if score >= 0.3:
        return "Low"
    return "Normal"


def _event_type(row: pd.Series, dominant_feature: str) -> str:
    traffic_z = abs(float(row.get("traffic_flow_roll_z_24", 0.0)))
    aqi_z = abs(float(row.get("aqi_roll_z_24", 0.0)))
    energy_z = abs(float(row.get("electricity_demand_roll_z_24", 0.0)))
    temperature_z = abs(float(row.get("temperature_roll_z_24", 0.0)))
    humidity_z = abs(float(row.get("humidity_roll_z_24", 0.0)))

    if "traffic_flow" in dominant_feature and traffic_z >= 2.5 and aqi_z >= 1.2:
        return "Possible traffic disruption"
    if "traffic_flow" in dominant_feature:
        return "Possible congestion pattern"
    if "aqi" in dominant_feature:
        return "Possible air-quality deterioration"
    if "electricity_demand" in dominant_feature:
        return "Possible energy demand surge"
    if temperature_z >= 2.0 or humidity_z >= 2.0:
        return "Possible weather impact"
    if dominant_feature.endswith("_diff_1") or dominant_feature.endswith("_pct_change_1"):
        return "Possible abrupt signal change"
    return "Possible public gathering pattern"


def _event_narrative(event_type: str, drivers: list[dict[str, Any]]) -> str:
    driver_text = ", ".join(driver["feature"].replace("_", " ") for driver in drivers[:3])
    return f"{event_type} inferred from unusual historical movement in {driver_text}. This is a model-generated explanation from uploaded historical data, not a confirmed real-world event."


def _build_event_rows(
    feature_df: pd.DataFrame,
    selected_features: list[str],
    combined_score: np.ndarray,
    is_anomaly: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    feature_values = feature_df[selected_features]
    z_values = ((feature_values - feature_values.mean()) / feature_values.std(ddof=0).replace(0, np.nan)).fillna(0.0)
    anomaly_indices = np.flatnonzero(is_anomaly)
    events = []
    for idx in anomaly_indices:
        row = feature_df.iloc[idx]
        contributions = z_values.iloc[idx].abs().sort_values(ascending=False)
        drivers = [
            {
                "feature": feature.replace("_", " ").title(),
                "contribution": round(float(value), 3),
            }
            for feature, value in contributions.head(4).items()
        ]
        dominant = contributions.index[0] if len(contributions) else selected_features[0]
        event_type = _event_type(row, dominant)
        score = float(combined_score[idx])
        events.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "event_type": event_type,
                "severity": _classify_severity(score),
                "anomaly_score": round(score, 4),
                "threshold": round(float(threshold), 4),
                "drivers": drivers,
                "description": _event_narrative(event_type, drivers),
                "latitude": 12.9716 + 0.035 * np.sin(idx / 7.0),
                "longitude": 77.5946 + 0.035 * np.cos(idx / 9.0),
            }
        )
    return sorted(events, key=lambda item: item["anomaly_score"], reverse=True)


def detect_urban_anomalies(raw_df: pd.DataFrame, config, max_events: int = 24) -> AnomalyDetectionResult:
    feature_df, feature_columns = prepare_anomaly_feature_frame(raw_df, config)
    selected_features = select_anomaly_features(feature_df, feature_columns)
    x = feature_df[selected_features].to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x)

    isolation = IsolationForest(
        n_estimators=240,
        contamination="auto",
        max_samples=min(256, len(feature_df)),
        random_state=config.random_seed,
    )
    isolation.fit(x_scaled)
    isolation_score = _normalize_scores(-isolation.decision_function(x_scaled))

    reconstruction_error = _train_autoencoder_scores(x_scaled, config.random_seed)
    autoencoder_score = _normalize_scores(reconstruction_error)

    combined_score = 0.55 * isolation_score + 0.45 * autoencoder_score
    threshold = _robust_threshold(combined_score, percentile=94.0)
    is_anomaly = combined_score >= threshold

    timeline = feature_df[["timestamp", *[col for col in config.domain_columns if col in feature_df.columns]]].copy()
    timeline["isolation_score"] = isolation_score
    timeline["autoencoder_score"] = autoencoder_score
    timeline["anomaly_score"] = combined_score
    timeline["is_anomaly"] = is_anomaly

    events = _build_event_rows(feature_df, selected_features, combined_score, is_anomaly, threshold)[:max_events]
    return AnomalyDetectionResult(
        events=events,
        timeline=timeline,
        thresholds={
            "combined": float(threshold),
            "autoencoder_reconstruction_error": _robust_threshold(reconstruction_error, percentile=95.0),
            "isolation_score": _robust_threshold(isolation_score, percentile=95.0),
        },
        feature_columns=selected_features,
        feature_frame=feature_df,
    )


def _domain_for_feature(feature: str) -> str:
    normalized = feature.lower().replace(" ", "_")
    if "traffic" in normalized:
        return "Traffic"
    if "aqi" in normalized or "air" in normalized:
        return "AQI"
    if "electricity" in normalized or "energy" in normalized:
        return "Electricity"
    if "temperature" in normalized:
        return "Temperature"
    if "humidity" in normalized:
        return "Humidity"
    return "Urban"


def _severity_counts(scores: pd.Series) -> dict[str, int]:
    counts = Counter(_classify_severity(float(score)) for score in scores)
    return {label: int(counts.get(label, 0)) for label in ["Critical", "High", "Medium", "Low", "Normal"]}


def _score_histogram(scores: pd.Series, bins: int = 10) -> list[dict[str, Any]]:
    counts, edges = np.histogram(scores.to_numpy(dtype=float), bins=bins, range=(0.0, 1.0))
    return [
        {
            "range": f"{edges[idx]:.1f}-{edges[idx + 1]:.1f}",
            "start": round(float(edges[idx]), 3),
            "end": round(float(edges[idx + 1]), 3),
            "count": int(count),
        }
        for idx, count in enumerate(counts)
    ]


def _monthly_stats(timeline: pd.DataFrame) -> list[dict[str, Any]]:
    frame = timeline.copy()
    frame["month"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m")
    grouped = frame.groupby("month", as_index=False).agg(
        records=("anomaly_score", "size"),
        anomalies=("is_anomaly", "sum"),
        avg_score=("anomaly_score", "mean"),
        max_score=("anomaly_score", "max"),
    )
    return [
        {
            "month": row["month"],
            "records": int(row["records"]),
            "anomalies": int(row["anomalies"]),
            "avg_score": round(float(row["avg_score"]), 4),
            "max_score": round(float(row["max_score"]), 4),
            "severity": _classify_severity(float(row["max_score"])),
        }
        for _, row in grouped.iterrows()
    ]


def _calendar_heatmap(timeline: pd.DataFrame) -> list[dict[str, Any]]:
    frame = timeline.copy()
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.date.astype(str)
    grouped = frame.groupby("date", as_index=False).agg(
        max_score=("anomaly_score", "max"),
        avg_score=("anomaly_score", "mean"),
        anomaly_count=("is_anomaly", "sum"),
    )
    return [
        {
            "date": row["date"],
            "score": round(float(row["max_score"]), 4),
            "avg_score": round(float(row["avg_score"]), 4),
            "anomaly_count": int(row["anomaly_count"]),
            "severity": _classify_severity(float(row["max_score"])),
        }
        for _, row in grouped.iterrows()
    ]


def _top_anomalous_days(timeline: pd.DataFrame, feature_df: pd.DataFrame, selected_features: list[str], limit: int = 12) -> list[dict[str, Any]]:
    frame = timeline.copy()
    feature_copy = feature_df.copy()
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.date.astype(str)
    feature_copy["date"] = pd.to_datetime(feature_copy["timestamp"]).dt.date.astype(str)
    z_values = ((feature_copy[selected_features] - feature_copy[selected_features].mean()) / feature_copy[selected_features].std(ddof=0).replace(0, np.nan)).fillna(0.0)
    z_values["date"] = feature_copy["date"]
    daily_driver: dict[str, str] = {}
    for date, group in z_values.groupby("date"):
        daily_driver[date] = group.drop(columns=["date"]).abs().sum().idxmax()
    grouped = frame.groupby("date", as_index=False).agg(
        score=("anomaly_score", "max"),
        anomalies=("is_anomaly", "sum"),
        traffic=("traffic_flow", "mean"),
        aqi=("aqi", "mean"),
        electricity=("electricity_demand", "mean"),
    )
    rows = grouped.sort_values("score", ascending=False).head(limit)
    result = []
    for _, row in rows.iterrows():
        driver = daily_driver.get(row["date"], "urban_signal")
        result.append(
            {
                "date": row["date"],
                "score": round(float(row["score"]), 4),
                "severity": _classify_severity(float(row["score"])),
                "category": _domain_for_feature(driver),
                "anomaly_count": int(row["anomalies"]),
                "traffic": round(float(row["traffic"]), 2),
                "aqi": round(float(row["aqi"]), 2),
                "electricity_demand": round(float(row["electricity"]), 2),
            }
        )
    return result


def _scatter_points(timeline: pd.DataFrame, limit: int = 700) -> list[dict[str, Any]]:
    frame = timeline.copy()
    if len(frame) > limit:
        frame = frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=int)]
    return [
        {
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "traffic_flow": round(float(row.get("traffic_flow", 0.0)), 3),
            "aqi": round(float(row.get("aqi", 0.0)), 3),
            "score": round(float(row["anomaly_score"]), 4),
            "severity": _classify_severity(float(row["anomaly_score"])),
            "is_anomaly": bool(row["is_anomaly"]),
        }
        for _, row in frame.iterrows()
    ]


def _score_series(timeline: pd.DataFrame, limit: int = 500) -> list[dict[str, Any]]:
    frame = timeline.copy()
    if len(frame) > limit:
        frame = frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=int)]
    return [
        {
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "autoencoder_score": round(float(row["autoencoder_score"]), 4),
            "isolation_score": round(float(row["isolation_score"]), 4),
            "hybrid_score": round(float(row["anomaly_score"]), 4),
        }
        for _, row in frame.iterrows()
    ]


def _dependency_links(timeline: pd.DataFrame, config) -> list[dict[str, Any]]:
    cols = [col for col in config.domain_columns if col in timeline.columns]
    corr = timeline[cols].corr().fillna(0.0)
    links = []
    for idx, left in enumerate(cols):
        for right in cols[idx + 1 :]:
            value = float(corr.loc[left, right])
            links.append(
                {
                    "source": left.replace("_", " ").title(),
                    "target": right.replace("_", " ").title(),
                    "strength": round(abs(value), 4),
                    "correlation": round(value, 4),
                    "direction": "positive" if value >= 0 else "negative",
                }
            )
    return sorted(links, key=lambda item: item["strength"], reverse=True)


def _summary_cards(timeline: pd.DataFrame, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anomaly_rows = timeline[timeline["is_anomaly"]]
    affected = "Urban"
    if events and events[0].get("drivers"):
        affected = _domain_for_feature(events[0]["drivers"][0]["feature"])
    return [
        {"label": "Records Analyzed", "value": f"{len(timeline):,}", "detail": "Historical 2025 records"},
        {"label": "Anomalies Detected", "value": f"{len(anomaly_rows):,}", "detail": "Hybrid score above threshold"},
        {"label": "Highest Anomaly Score", "value": f"{timeline['anomaly_score'].max():.3f}", "detail": _classify_severity(float(timeline["anomaly_score"].max()))},
        {"label": "Affected Domain", "value": affected, "detail": "Top SHAP-style contribution"},
    ]


def _recommendations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return [
            {
                "title": "Continue historical monitoring",
                "priority": "Normal",
                "reason": "No high-scoring anomaly clusters were detected in the uploaded historical dataset.",
                "actions": ["Maintain baseline model review", "Re-run after adding verified historical CSV data"],
            }
        ]

    domain_counts: Counter[str] = Counter()
    for event in events:
        for driver in event["drivers"][:2]:
            domain_counts[_domain_for_feature(driver["feature"])] += 1

    templates = {
        "Traffic": ("Review peak-hour traffic control plan", "Traffic features repeatedly appear as high-contribution anomaly drivers.", ["Evaluate bus-priority support", "Simulate flexible office timings", "Review heavy-vehicle restrictions historically"]),
        "AQI": ("Assess air-quality mitigation strategy", "AQI deviations are visible in anomaly drivers.", ["Compare AQI spikes with traffic-heavy hours", "Evaluate temporary heavy-vehicle restrictions", "Prepare advisory templates"]),
        "Electricity": ("Analyze demand-response readiness", "Electricity demand contributes strongly to abnormal historical patterns.", ["Review peak demand windows", "Test load shifting policies", "Coordinate non-critical load scheduling"]),
        "Temperature": ("Plan weather-sensitive response", "Temperature movements contribute to unusual historical patterns.", ["Compare heat periods with energy changes", "Evaluate cooling-center readiness", "Simulate demand-side energy response"]),
        "Humidity": ("Review humidity-linked weather impact", "Humidity contributes to anomalous weather behavior.", ["Compare humidity with AQI and travel delay", "Evaluate weather advisory thresholds", "Flag periods for manual review"]),
    }
    recommendations = []
    for domain, _ in domain_counts.most_common(4):
        if domain in templates:
            title, reason, actions = templates[domain]
            recommendations.append({"title": title, "priority": "High" if domain in {"Traffic", "AQI", "Electricity"} else "Medium", "reason": reason, "actions": actions})
    recommendations.append(
        {
            "title": "Use scenario simulation before action",
            "priority": "Medium",
            "reason": "The dashboard uses uploaded historical data only and should support planning, not automatic live control.",
            "actions": ["Compare policy alternatives", "Document assumptions", "Validate with domain experts before field use"],
        }
    )
    return recommendations


def _dynamic_insights(events: list[dict[str, Any]], timeline: pd.DataFrame) -> list[str]:
    if not events:
        return ["The uploaded historical dataset does not show anomaly clusters above the adaptive threshold."]
    top = events[0]
    high_count = sum(1 for event in events if event["severity"] in {"High", "Critical"})
    avg_score = float(timeline["anomaly_score"].mean())
    return [
        f"The highest hybrid anomaly score is {top['anomaly_score']:.3f} on {top['timestamp'][:10]}, classified as {top['severity']}.",
        f"{high_count} high-priority historical events were identified by combining Isolation Forest and autoencoder reconstruction scores.",
        f"The average anomaly score across all analyzed records is {avg_score:.3f}, so the dashboard emphasizes deviations instead of ordinary seasonal movement.",
        "Interpretations are possible explanations from uploaded 2025 historical data and do not claim confirmed live incidents.",
    ]


def build_anomaly_payload(config) -> dict[str, Any]:
    from utils.data_utils import load_input_dataframe

    raw_df = load_input_dataframe(config)
    result = detect_urban_anomalies(raw_df, config, max_events=80)
    latest_events = sorted(result.events, key=lambda item: item["timestamp"], reverse=True)[:8]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = Path(config.output_dir) / "urban_anomaly_timeline.csv"
    events_path = Path(config.output_dir) / "urban_events.json"
    result.timeline.to_csv(timeline_path, index=False)
    events_path.write_text(json.dumps(result.events, indent=2), encoding="utf-8")

    top_days = _top_anomalous_days(result.timeline, result.feature_frame, result.feature_columns)
    high_events = [event for event in result.events if event["severity"] in {"Critical", "High"}]
    return {
        "objective": "Intelligent Anomaly Detection for Urban Event Discovery",
        "city": getattr(config, "city", "default"),
        "data_mode": "Uploaded historical CSV data only; no live sensors or real-time streaming.",
        "last_updated": raw_df["timestamp"].max().isoformat(),
        "thresholds": {key: round(value, 4) for key, value in result.thresholds.items()},
        "feature_count": len(result.feature_columns),
        "event_count": len(result.events),
        "severity_counts": _severity_counts(result.timeline["anomaly_score"]),
        "summary_cards": _summary_cards(result.timeline, result.events),
        "events": latest_events,
        "all_events": result.events,
        "top_anomalous_days": top_days,
        "monthly_stats": _monthly_stats(result.timeline),
        "heatmap": _calendar_heatmap(result.timeline),
        "histogram": _score_histogram(result.timeline["anomaly_score"]),
        "scatter": _scatter_points(result.timeline),
        "score_series": _score_series(result.timeline),
        "dependency_links": _dependency_links(result.timeline, config),
        "dynamic_insights": _dynamic_insights(result.events, result.timeline),
        "recommendations": _recommendations(result.events),
        "severity_legend": [
            {"label": "Normal", "min": 0.0, "max": 0.3},
            {"label": "Low", "min": 0.3, "max": 0.5},
            {"label": "Medium", "min": 0.5, "max": 0.7},
            {"label": "High", "min": 0.7, "max": 0.9},
            {"label": "Critical", "min": 0.9, "max": 1.0},
        ],
        "artifacts": {
            "timeline": str(timeline_path),
            "events": str(events_path),
        },
        "exports": {
            "csv": "/api/anomalies/export.csv",
            "pdf": "/api/anomalies/export.pdf",
        },
        "model_integration": [
            "Isolation Forest captures sparse multivariate outliers in the engineered historical feature space.",
            "The autoencoder reconstructs normal historical behavior and exposes high-error records.",
            "The normalized hybrid anomaly score combines both signals into a 0-1 severity scale for decision support.",
            "SHAP-style contributions are computed from standardized feature deviations for transparent auditability.",
        ],
        "dashboard": {
            "cards": "Records analyzed, anomalies detected, highest score, and affected domain.",
            "visuals": "Calendar heatmap, histogram, scatter plot, reconstruction score series, and dependency links.",
            "filters": "Date, domain, and severity filters are applied client-side on the historical result set.",
        },
        "audit_note": (
            f"{len(high_events)} high-priority records were detected from historical data. "
            "Possible explanations should be treated as planning hypotheses, not confirmed incidents."
        ),
    }


def build_anomaly_csv_export(config) -> bytes:
    payload = build_anomaly_payload(config)
    rows = payload["all_events"] or payload["events"]
    buffer = StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_anomaly_pdf_export(config) -> bytes:
    payload = build_anomaly_payload(config)
    buffer = BytesIO()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("ForeSightX Hybrid Anomaly Detection Report", fontsize=16, fontweight="bold")

    hist = payload["histogram"]
    axes[0, 0].bar([item["range"] for item in hist], [item["count"] for item in hist], color="#50e3c2")
    axes[0, 0].set_title("Hybrid Anomaly Score Histogram")
    axes[0, 0].tick_params(axis="x", rotation=45)

    months = payload["monthly_stats"]
    axes[0, 1].plot([item["month"] for item in months], [item["max_score"] for item in months], marker="o", color="#8ab4ff")
    axes[0, 1].set_title("Monthly Maximum Score")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].set_ylim(0, 1)

    scatter = payload["scatter"]
    colors = ["#ff5f7e" if item["is_anomaly"] else "#8b95a7" for item in scatter]
    axes[1, 0].scatter([item["traffic_flow"] for item in scatter], [item["aqi"] for item in scatter], c=colors, s=12, alpha=0.75)
    axes[1, 0].set_title("Traffic vs AQI: Normal vs Anomalous")
    axes[1, 0].set_xlabel("Traffic Flow")
    axes[1, 0].set_ylabel("AQI")

    series = payload["score_series"]
    axes[1, 1].plot([item["hybrid_score"] for item in series], label="Hybrid", color="#f8d66d")
    axes[1, 1].plot([item["autoencoder_score"] for item in series], label="Autoencoder", color="#50e3c2", alpha=0.75)
    axes[1, 1].set_title("Score Timeline Sample")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    fig.text(0.05, 0.02, payload["data_mode"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def _ensure_project_root_on_path() -> Path:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def main() -> None:
    project_root = _ensure_project_root_on_path()
    os.chdir(project_root)
    from utils.config import CONFIG, apply_city_config

    config = apply_city_config(CONFIG, "bangalore")
    payload = build_anomaly_payload(config)
    print(payload["objective"])
    print(f"City: {payload['city']}")
    print(f"Detected events: {payload['event_count']}")
    print(f"Severity counts: {payload['severity_counts']}")
    print(f"Timeline artifact: {payload['artifacts']['timeline']}")
    print(f"Events artifact: {payload['artifacts']['events']}")
    if payload["events"]:
        top_event = payload["events"][0]
        print(f"Latest event: {top_event['severity']} - {top_event['event_type']} at {top_event['timestamp']}")


if __name__ == "__main__":
    main()
