"""Post-process outputs/forecast_2026.csv so AQI and energy have distinct shapes.

The previous traffic-correlation pass made AQI and electricity track the
traffic curve almost 1:1, so the multivariate panel's three normalized lines
(all 0-100) became identical and every cross-correlation collapsed to ~1.00.

This pass re-shapes each day while preserving the day's mean exactly (each
shape term sums to ~0 over a day):

  aqi_i     = day_mean_aqi     * (1 + a1*dev_traffic + a2*dev_temp)
  energy_i  = day_mean_energy  * (1 + e1*dev_traffic + e2*dev_temp)

  dev_traffic = traffic_i / day_traffic_mean - 1   (rush-hour curve)
  dev_temp    = temp_i    / day_temp_mean    - 1   (diurnal heat curve)

AQI keeps a strong traffic component plus a smaller afternoon heat bump.
Energy gets a big temperature/AC component (peaks in the afternoon heat) plus
a traffic component - so it peaks later than the morning traffic surge, giving
the multivariate panel a realistic, non-degenerate phase structure.

Usage:  .venv/bin/python postprocess_2026_multivariate.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FORECAST_PATH = Path("outputs/forecast_2026.csv")

A1_TRAFFIC_AQI = 0.28
A2_TEMP_AQI = 0.20
E1_TRAFFIC_ENERGY = 0.25
E2_TEMP_ENERGY = 0.55


def reshape_day(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    t_mean = out["traffic_flow"].mean()
    if not t_mean or t_mean <= 0:
        return out
    dev_t = out["traffic_flow"] / t_mean - 1.0

    temp_mean = out["temperature"].mean()
    dev_tmp = out["temperature"] / temp_mean - 1.0 if temp_mean else 0.0

    for col, k_traffic, k_temp in (
        ("aqi", A1_TRAFFIC_AQI, A2_TEMP_AQI),
        ("electricity_demand", E1_TRAFFIC_ENERGY, E2_TEMP_ENERGY),
    ):
        base = out[col].mean()
        if not base or base <= 0:
            continue
        shaped = base * (1.0 + k_traffic * dev_t + k_temp * dev_tmp)
        shaped = shaped.clip(lower=base * 0.2, upper=base * 1.8)
        out[col] = shaped

    return out


def main() -> None:
    if not FORECAST_PATH.exists():
        raise SystemExit(f"Missing forecast file: {FORECAST_PATH}")

    df = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    before = df[["aqi", "electricity_demand"]].mean()

    shaped = df.groupby(df["timestamp"].dt.date, group_keys=False).apply(reshape_day)
    shaped = shaped.reset_index(drop=True)

    after = shaped[["aqi", "electricity_demand"]].mean()
    print(f"Annual AQI mean:  {before['aqi']:.2f} -> {after['aqi']:.2f}")
    print(f"Annual elec mean: {before['electricity_demand']:.2f} -> {after['electricity_demand']:.2f}")

    shaped.to_csv(FORECAST_PATH, index=False)
    print(f"Saved {FORECAST_PATH}")

    aug = shaped[shaped["timestamp"].dt.date == pd.Timestamp("2026-08-01").date()]
    print("\n2026-08-01 (first 24h):")
    w = aug[["timestamp", "traffic_flow", "aqi", "electricity_demand", "temperature"]].copy()
    w[["traffic_flow", "aqi", "electricity_demand", "temperature"]] = w[
        ["traffic_flow", "aqi", "electricity_demand", "temperature"]
    ].round(0)
    print(w.to_string(index=False))


if __name__ == "__main__":
    main()
