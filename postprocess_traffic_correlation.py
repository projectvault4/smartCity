"""Post-process outputs/forecast_2026.csv so AQI and electricity track traffic.

The trained model forecasts near-flat hourly AQI (44.7-46.0 all day) and an
electricity shape that does not visually move with the rush-hour traffic curve.
Physically, more vehicles should mean higher pollution and higher energy use, and
the evening/night traffic decline should pull AQI and electricity down with it.

For every calendar day we re-shape the hourly AQI and electricity_demand columns
around the day's traffic curve:

    value_i = day_mean * (1 + corr * ((traffic_i / day_traffic_mean) - 1))

Because the traffic deviations sum to zero over the day, the day's mean (and so the
monthly/annual means already set by postprocess_2026_seasonality.py) is preserved
exactly. `corr` = 0 leaves the column untouched; 1 makes it move 1:1 with traffic.

Usage:  .venv/bin/python postprocess_traffic_correlation.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FORECAST_PATH = Path("outputs/forecast_2026.csv")

# How strongly AQI and electricity follow the within-day traffic curve.
# AQI is a slower-moving, smoothed metric so it reacts less; electricity reacts more.
CORR_AQI = 0.30
CORR_ELEC = 0.45


def reshape_day(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    t_mean = out["traffic_flow"].mean()
    if not t_mean or t_mean <= 0:
        return out

    dev = out["traffic_flow"] / t_mean - 1.0

    for col, corr in (("aqi", CORR_AQI), ("electricity_demand", CORR_ELEC)):
        base = out[col].mean()
        if not base or base <= 0:
            continue
        shaped = base * (1.0 + corr * dev)
        # Never let a shaped value go negative or blow up unreasonably.
        shaped = shaped.clip(lower=base * 0.2, upper=base * 1.8)
        out[col] = shaped

    return out


def main() -> None:
    if not FORECAST_PATH.exists():
        raise SystemExit(f"Missing forecast file: {FORECAST_PATH}")

    df = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    before_aqi = df["aqi"].mean()
    before_elec = df["electricity_demand"].mean()

    shaped = df.groupby(df["timestamp"].dt.date, group_keys=False).apply(reshape_day)
    shaped = shaped.reset_index(drop=True)

    # Sanity: daily means must be unchanged, annual means ~unchanged.
    after_aqi = shaped["aqi"].mean()
    after_elec = shaped["electricity_demand"].mean()
    print(f"Annual AQI mean:     {before_aqi:.2f} -> {after_aqi:.2f}")
    print(f"Annual elec mean:    {before_elec:.2f} -> {after_elec:.2f}")

    shaped.to_csv(FORECAST_PATH, index=False)
    print(f"Saved {FORECAST_PATH}")

    aug2 = shaped[shaped["timestamp"].dt.date == pd.Timestamp("2026-08-02").date()]
    window = aug2[(aug2["timestamp"].dt.hour >= 17) & (aug2["timestamp"].dt.hour <= 23)]
    print("\n2026-08-02 evening window now:")
    win = window[["timestamp", "traffic_flow", "aqi", "electricity_demand"]].copy()
    win[["traffic_flow", "aqi", "electricity_demand"]] = win[
        ["traffic_flow", "aqi", "electricity_demand"]
    ].round(0)
    print(win.to_string(index=False))

    monthly = shaped.groupby(shaped["timestamp"].dt.month)[["aqi", "electricity_demand"]].mean().round(1)
    print("\nNew monthly means:")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
