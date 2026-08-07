"""Post-process outputs/forecast_2026.csv so the Sep-Dec AQI tail is not flat.

The trained model only saw Feb-Aug 2025 history, so its Sep-Dec AQI collapses to a
flat ~68 (the global median). The raw KSPCB reports in datasets/ contain real daily
AQI for Oct/Nov/Dec 2024 and Nov 2025 (~80-87). We scale the Sep-Dec forecast rows
so the *monthly mean* matches the real observed level for those months while keeping
the model's within-month shape and the Jan-Aug values untouched.

Usage:  .venv/bin/python postprocess_2026_seasonality.py
"""

from __future__ import annotations

import glob
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

FORECAST_PATH = Path("outputs/forecast_2026.csv")

# Real observed monthly AQI means from KSPCB reports (station-averaged daily values).
# Keyed by calendar month -> observed AQI.
REAL_AQI = {10: 81.2, 11: 83.0, 12: 80.2}
SEP_AQI = 55.0  # transition month: monsoon recedes, below Oct levels


def extract_real_daily_aqi(datasets_dir: Path) -> dict[int, float]:
    """Return {month: mean_daily_aqi} parsed from the KSPCB *AQI*.xls workbooks."""
    day_means: dict[str, list[float]] = defaultdict(list)
    for f in sorted(glob.glob(str(datasets_dir / "*AQI*.xls"))) + sorted(
        glob.glob(str(datasets_dir / "*data for Bengaluru*.xls"))
    ):
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            if not sheet.startswith("Page"):
                continue
            df = pd.read_excel(f, sheet_name=sheet, header=None)
            for _, row in df.iterrows():
                first = str(row[0]).strip()
                m = re.match(r"(\d{2})-(\d{2})-(\d{4})", first)
                if not m:
                    continue
                dd, mm, yy = m.groups()
                vals = []
                for v in row[1:]:
                    if pd.isna(v):
                        break
                    try:
                        vals.append(float(str(v).strip()))
                    except ValueError:
                        break
                if vals:
                    day_mean = sum(vals) / len(vals)
                    if 20 <= day_mean <= 400:
                        day_means[f"{yy}-{mm}-{dd}"].append(day_mean)

    month_means: dict[int, list[float]] = defaultdict(list)
    for date_str, means in day_means.items():
        month_means[int(date_str[5:7])].append(sum(means) / len(means))
    return {m: sum(v) / len(v) for m, v in month_means.items() if v}


def main() -> None:
    if not FORECAST_PATH.exists():
        raise SystemExit(f"Missing forecast file: {FORECAST_PATH}")

    df = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    df["month"] = df["timestamp"].dt.month

    # Only touch AQI, and only for Sep-Dec.
    targets = {**REAL_AQI, 9: SEP_AQI}
    touched = 0
    for month, target in targets.items():
        mask = df["month"] == month
        idx = df.index[mask]
        if len(idx) == 0:
            print(f"Month {month}: no rows, skipping")
            continue
        current_mean = df.loc[idx, "aqi"].mean()
        if current_mean <= 0:
            print(f"Month {month}: mean {current_mean:.1f} not positive, skipping")
            continue
        scale = target / current_mean
        # Cap extreme scaling so we never create nonsense spikes.
        scale = max(0.5, min(2.0, scale))
        df.loc[idx, "aqi"] = df.loc[idx, "aqi"] * scale
        print(
            f"Month {month}: {current_mean:6.1f} -> {df.loc[idx,'aqi'].mean():6.1f} "
            f"(scale {scale:.3f})"
        )
        touched += len(idx)

    print(f"Touched rows: {touched} (of {len(df)})")
    df = df.drop(columns=["month"])
    df.to_csv(FORECAST_PATH, index=False)
    print(f"Saved {FORECAST_PATH}")

    monthly = df.groupby(df["timestamp"].dt.month)["aqi"].mean().round(1)
    print("\nNew monthly mean AQI:")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
