from __future__ import annotations

from utils.config import CONFIG
from utils.forecast_service import build_forecast_payload


def _build_alerts(metrics: list[dict]) -> list[str]:
    alerts = []
    for metric in metrics:
        status = metric["status"].lower()
        if metric["key"] == "traffic_flow" and "busy" in status:
            alerts.append("Traffic congestion expected")
        if metric["key"] == "aqi" and ("unhealthy" in status or "moderate" in status):
            alerts.append("Air quality may worsen next hour")
        if metric["key"] == "temperature" and "warm" in status:
            alerts.append("Temperature may stay on the warmer side")
        if metric["key"] == "electricity_demand" and "high" in status:
            alerts.append("Higher electricity demand expected")
    return alerts


def main():
    payload = build_forecast_payload(CONFIG)
    alerts = _build_alerts(payload["metrics"])

    print(f"Prediction Time: {payload['forecast_for'].replace('T', ' ')}")
    print("\nNext Hour Prediction:\n")

    for metric in payload["metrics"]:
        print(f"{metric['label']}: {metric['prediction']:.0f} ({metric['status']})")

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
    for metric in payload["metrics"]:
        print(f"- {metric['label']}:")
        for reason in metric["explanations"]:
            print(f"  - {reason}")

    print("\nSimple Summary:")
    for line in payload["summary"]:
        print(f"- {line}")


if __name__ == "__main__":
    main()
