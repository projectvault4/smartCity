"""Generate 2026 anomaly artifacts from the 2026 forecast series.

Runs the same Isolation Forest + autoencoder hybrid used for the training
artifacts, but scores the full-year 2026 forecast (outputs/forecast_2026.csv)
so the anomaly dashboard describes the forecasted year instead of the 2022-2025
training history.

Writes:
  outputs/urban_events_2026.json
  outputs/urban_anomaly_timeline_2026.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from utils.anomaly_detection import detect_urban_anomalies, ANOMALY_WARMUP_ROWS
from utils.config import CONFIG


def main() -> None:
    forecast_path = PROJECT_ROOT / "outputs" / "forecast_2026.csv"
    if not forecast_path.exists():
        raise SystemExit(f"Missing 2026 forecast: {forecast_path}")

    df = pd.read_csv(forecast_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    # The warm-up window is computed from lag/rolling features; for a full-year
    # forecast starting at 00:00 the first ~24 rows have no history, same as
    # the training pipeline. Reuse the same constant so results stay consistent.
    result = detect_urban_anomalies(df, CONFIG, max_events=80)

    events_out = PROJECT_ROOT / "outputs" / "urban_events_2026.json"
    timeline_out = PROJECT_ROOT / "outputs" / "urban_anomaly_timeline_2026.csv"
    result.timeline.to_csv(timeline_out, index=False)
    events_out.write_text(json.dumps(result.events, indent=2), encoding="utf-8")

    print(f"Rows scored: {len(result.timeline)}")
    print(f"Events written: {len(result.events)}")
    print(f"Timeline: {timeline_out}")
    print(f"Events: {events_out}")
    if result.events:
        top = result.events[0]
        print(f"Top event: {top['severity']} - {top['event_type']} at {top['timestamp']}")


if __name__ == "__main__":
    main()
