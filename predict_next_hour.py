from __future__ import annotations

from pathlib import Path

from utils.config import CONFIG
from utils.explainable_forecasting import classify_forecast_levels
from utils.forecast_service import build_forecast_payload, build_past_present_future_frame


def _aqi_health_note(aqi_value: float, status: str) -> str:
    normalized = status.lower()
    if "good" in normalized:
        return "air quality is generally safe for most people"
    if "moderate" in normalized:
        return "unusually sensitive people may feel mild breathing discomfort"
    if "sensitive" in normalized:
        return (
            "more likely to affect children, older adults, and people with asthma, lung disease, or heart disease"
        )
    return "more likely to affect everyone, especially children, older adults, and people with breathing or heart conditions"


def _format_metric_line(metric: dict) -> str:
    line = f"{metric['label']}: {metric['prediction']:.0f} ({metric['status']}, {metric['confidence']} confidence)"
    if metric["key"] == "aqi":
        line += f" - {_aqi_health_note(metric['prediction'], metric['status'])}"
    return line


def _simplify_reason(reason: str) -> str:
    simple = reason
    replacements = {
        "12-Hour Average": "recent 12-hour average",
        "6-Hour Average": "recent 6-hour average",
        "6-Hour Variation": "recent 6-hour changes",
        "6-Hour Low": "recent 6-hour low",
        "pushed upward the": "helped increase the",
        "pulled downward the": "helped lower the",
        "forecast.": "prediction.",
    }
    for old, new in replacements.items():
        simple = simple.replace(old, new)
    return simple


def _simple_change_line(metric: dict) -> str:
    change = float(metric["change"])
    label = metric["label"]
    current = float(metric["current"])
    prediction = float(metric["prediction"])
    if abs(change) < 1:
        return f"{label} should stay close to the current value, around {prediction:.0f}."
    if change > 0:
        return f"{label} may go up from {current:.0f} to about {prediction:.0f}."
    return f"{label} may go down from {current:.0f} to about {prediction:.0f}."


def _plain_reason_lines(metrics: list[dict]) -> list[str]:
    metric_map = {metric["key"]: metric for metric in metrics}
    traffic = metric_map["traffic_flow"]
    aqi = metric_map["aqi"]
    temperature = metric_map["temperature"]
    demand = metric_map["electricity_demand"]

    lines = []

    if traffic["change"] > 1:
        lines.append(
            f"Traffic may increase because recent city activity still looks strong, so the model expects more vehicles on the road."
        )
    elif traffic["change"] < -1:
        lines.append(
            f"Traffic may decrease because the recent traffic pattern has been cooling down, so the model expects fewer vehicles next hour."
        )
    else:
        lines.append("Traffic may stay close to the current level because the recent traffic pattern looks steady.")

    if aqi["change"] > 1:
        lines.append(
            f"AQI may increase mainly because traffic and urban activity are expected to stay stronger, which can add more pollution to the air."
        )
    elif aqi["change"] < -1:
        lines.append(
            f"AQI may decrease because traffic pressure is expected to ease, which usually reduces pollution buildup."
        )
    else:
        lines.append("AQI may stay near the current level because pollution conditions do not look very different from the recent pattern.")

    if temperature["change"] > 1:
        lines.append(
            f"Temperature may increase because the recent weather trend is slightly warmer, so the model expects the next hour to stay warmer too."
        )
    elif temperature["change"] < -1:
        lines.append(
            f"Temperature may decrease because the recent weather trend is softer, so the next hour is expected to be a little cooler."
        )
    else:
        lines.append("Temperature may stay almost the same because recent weather conditions look stable.")

    if demand["change"] > 1:
        lines.append(
            f"Electricity demand may increase because warmer conditions and continued city activity usually make people use more power."
        )
    elif demand["change"] < -1:
        lines.append(
            f"Electricity demand may decrease because traffic and temperature are expected to stay lower, so power usage may also reduce."
        )
    else:
        lines.append("Electricity demand may stay near the current level because overall activity and temperature do not change much.")

    return lines


def _uncertainty_lines(metrics: list[dict]) -> list[str]:
    lines = []
    for metric in metrics:
        width = float(metric["upper"]) - float(metric["lower"])
        lines.append(
            f"{metric['label']}: {metric['confidence']} confidence because {metric['uncertainty_reason']}. "
            f"The expected range is about {metric['lower']:.0f} to {metric['upper']:.0f}."
        )
    return lines


def _format_future_chain_line(row: dict, history_df) -> str:
    labels = classify_forecast_levels(
        {
            "traffic_flow": float(row["traffic_flow"]),
            "aqi": float(row["aqi"]),
            "temperature": float(row["temperature"]),
            "electricity_demand": float(row["electricity_demand"]),
        },
        history_df,
    )
    return (
        f"- {row['timestamp']}: "
        f"Traffic={row['traffic_flow']:.0f} ({labels['traffic_flow']}) -> "
        f"AQI={row['aqi']:.0f} ({labels['aqi']}) -> "
        f"Temp={row['temperature']:.0f} ({labels['temperature']}) -> "
        f"Demand={row['electricity_demand']:.0f} ({labels['electricity_demand']})"
    )


def _build_alerts(metrics: list[dict]) -> list[str]:
    alerts = []
    for metric in metrics:
        status = metric["status"].lower()
        if metric["key"] == "traffic_flow" and "busy" in status:
            alerts.append("Traffic congestion expected")
        if metric["key"] == "aqi" and ("unhealthy" in status or "moderate" in status):
            alerts.append(f"Air quality may worsen next hour: {_aqi_health_note(metric['prediction'], metric['status'])}.")
        if metric["key"] == "temperature" and "warm" in status:
            alerts.append("Temperature may stay on the warmer side")
        if metric["key"] == "electricity_demand" and "high" in status:
            alerts.append("Higher electricity demand expected")
    return alerts


def main():
    payload = build_forecast_payload(CONFIG)
    timeline_df = build_past_present_future_frame(CONFIG, future_steps=CONFIG.direct_forecast_steps)
    alerts = _build_alerts(payload["metrics"])
    output_path = Path(CONFIG.output_dir) / "past_present_future_forecast.csv"
    timeline_df.to_csv(output_path, index=False)

    print(f"Prediction Time: {payload['forecast_for'].replace('T', ' ')}")
    print("\nNext Hour Prediction:\n")

    for metric in payload["metrics"]:
        print(_format_metric_line(metric))

    print("\nExpected Ranges:")
    for metric in payload["metrics"]:
        print(f"{metric['label']}: {metric['lower']:.0f} - {metric['upper']:.0f}")

    print("\nAlerts:")
    if alerts:
        for alert in alerts:
            print(f"- {alert}")
    else:
        print("- No major alerts expected")

    print("\nWhy The Model Thinks This:")
    for line in _plain_reason_lines(payload["metrics"]):
        print(f"- {line}")

    print("\nConfidence And Uncertainty:")
    for line in _uncertainty_lines(payload["metrics"]):
        print(f"- {line}")

    print("\nSimple Summary:")
    for metric in payload["metrics"]:
        print(f"- {_simple_change_line(metric)}")

    print("\nInterconnected Output:")
    for line in payload["interconnected_summary"]:
        print(f"- {line}")

    print("\nPast Present Future File:")
    print(f"- Saved timestamped history + {CONFIG.direct_forecast_steps} direct future hours to {output_path}")

    print("\nFuture Preview:")
    preview = timeline_df[timeline_df["time_segment"] == "future"].head(5)
    history_df = timeline_df[timeline_df["time_segment"].isin(["past", "present"])]
    for _, row in preview.iterrows():
        print(_format_future_chain_line(row, history_df))


if __name__ == "__main__":
    main()
