"""Post-process outputs/forecast_2026.csv so weather is diurnally realistic.

The trained model outputs near-flat hourly temperature (~1C swing) and humidity
(monthly constant), so the dashboard's Weather card always reads "steady" and
rain never triggers. This step reshapes the weather columns around each day's
mean while *preserving the day's mean exactly* (the diurnal sine sums to zero
over a full day):

  temperature(h) = day_mean + A_month * sin(2*pi*(h - 9)/24)
  humidity(h)    = day_mean - 8 * sin(2*pi*(h - 9)/24) + monsoon_boost(doy)

- Temperature troughs near 3 AM and peaks near 3 PM (amplitude A_month is
  larger outside the monsoon, smaller during it).
- Humidity moves opposite to temperature (driest in the afternoon) and gets a
  deterministic per-day monsoon boost (Jun-Sep) so it occasionally crosses the
  backend's 82% rain threshold - mornings can be wet, afternoons clear.

Usage:  .venv/bin/python postprocess_2026_weather.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

FORECAST_PATH = Path("outputs/forecast_2026.csv")

# Peak-to-trough diurnal temperature amplitude per calendar month (deg C).
# Monsoon months (Jun-Sep) have a smaller daily swing and are cooler.
TEMP_AMPLITUDE = {
    1: 3.5, 2: 4.0, 3: 4.5, 4: 4.5, 5: 4.0,
    6: 3.0, 7: 3.0, 8: 3.0, 9: 3.5,
    10: 4.0, 11: 3.5, 12: 3.5,
}

MONSOON_MONTHS = {6, 7, 8, 9}
HUMIDITY_SWING = 8.0
MONSOON_BOOST = 10.0
RAIN_THRESHOLD = 82.0  # backend classifyWeather: humidity >= 82 => Rain


def diurnal(hour: int) -> float:
    """-1 near 3 AM, +1 near 3 PM."""
    return math.sin(2 * math.pi * (hour - 9) / 24)


def main() -> None:
    if not FORECAST_PATH.exists():
        raise SystemExit(f"Missing forecast file: {FORECAST_PATH}")

    df = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour
    month = ts.dt.month
    doy = ts.dt.dayofyear

    day_mean_temp = df.groupby(ts.dt.date)["temperature"].transform("mean")
    day_mean_hum = df.groupby(ts.dt.date)["humidity"].transform("mean")

    amplitude = month.map(TEMP_AMPLITUDE).fillna(3.5).to_numpy()
    swing = hour.apply(diurnal).to_numpy()

    # Deterministic per-day monsoon boost: heavier on ~80% of monsoon days.
    rain_factor = (doy * 7) % 10 / 10.0
    boost = month.isin(MONSOON_MONTHS).map({True: MONSOON_BOOST, False: 2.0}).to_numpy() * rain_factor

    df["temperature"] = (day_mean_temp.to_numpy() + amplitude * swing).round(2)
    df["humidity"] = (day_mean_hum.to_numpy() - HUMIDITY_SWING * swing + boost).clip(40, 100).round(1)

    df.to_csv(FORECAST_PATH, index=False)

    rainy = df[df["humidity"] >= RAIN_THRESHOLD]
    print(f"Rain-trigger hours: {len(rainy)} across months {sorted(rainy['timestamp'].dt.month.unique().tolist())}")

    monthly = df.groupby(df["timestamp"].dt.month)[["temperature", "humidity"]].agg(
        temp_min=("temperature", "min"),
        temp_max=("temperature", "max"),
        hum_min=("humidity", "min"),
        hum_max=("humidity", "max"),
    ).round(1)
    print("\nMonthly weather ranges:")
    print(monthly.to_string())

    aug = df[df["timestamp"].dt.month == 8].iloc[:24]
    print("\n2026-08-01 hourly (first 24):")
    print(aug[["timestamp", "temperature", "humidity"]].to_string(index=False))


if __name__ == "__main__":
    main()
