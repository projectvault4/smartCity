from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import CONFIG, apply_city_config
from utils.data_utils import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
    create_datasets,
    load_input_dataframe,
    set_seed,
)
from utils.explainable_forecasting import ExplainableTimeSeriesForecaster
from utils.forecast_service import load_or_train_forecaster

FORECAST_YEAR = 2026
MAX_WINDOW = 24
# Keep a full week of history so the largest lag (24h) and rolling windows are valid.
HISTORY_WINDOW = 24 * 7
# Blend weight for the learned model prediction vs. the historical seasonal profile.
# Damped recursion keeps the year-ahead forecast stable (lag feedback can diverge).
MODEL_BLEND = 0.55
SEASONAL_BLEND = 1.0 - MODEL_BLEND


def _hierarchical_median_lookup(
    df: pd.DataFrame,
    column: str,
    ts: pd.Timestamp,
) -> float:
    """Hierarchical seasonal lookup: (month, dow, hour) -> (month, hour) -> (dow, hour) -> (hour)."""
    ts_index = pd.to_datetime(df["timestamp"])
    hour_series = df[column]

    groups = [
        (ts.month, ts.dayofweek, ts.hour),
        (ts.month, ts.hour),
        (ts.dayofweek, ts.hour),
        (ts.hour,),
    ]
    masks = [
        (ts_index.dt.month == ts.month) & (ts_index.dt.dayofweek == ts.dayofweek) & (ts_index.dt.hour == ts.hour),
        (ts_index.dt.month == ts.month) & (ts_index.dt.hour == ts.hour),
        (ts_index.dt.dayofweek == ts.dayofweek) & (ts_index.dt.hour == ts.hour),
        (ts_index.dt.hour == ts.hour),
    ]
    for mask in masks:
        values = hour_series[mask].dropna()
        if len(values) >= 5:
            return float(values.median())
    return float(hour_series.median())


def _seasonal_profiles(prepared_df: pd.DataFrame, columns: tuple) -> dict[str, pd.DataFrame]:
    profile_df = prepared_df[["timestamp", *list(columns)]].copy()
    ts = pd.to_datetime(profile_df["timestamp"])
    profile_df["month"] = ts.dt.month
    profile_df["day_of_week"] = ts.dt.dayofweek
    profile_df["hour"] = ts.dt.hour
    return {col: profile_df for col in columns}


def _feature_frame(history: pd.DataFrame, config, target_cols: tuple) -> pd.DataFrame:
    """Build the engineered feature row for the newest entry in `history`."""
    frame = add_time_features(history.copy())
    frame = add_lag_features(frame, list(config.domain_columns), config.lag_steps)
    frame = add_rolling_features(frame, list(config.domain_columns), config.rolling_windows)
    return frame


def build_2026_forecast_frame(config=CONFIG) -> pd.DataFrame:
    set_seed(config.random_seed)
    raw_df = load_input_dataframe(config).copy()
    datasets = create_datasets(config, raw_df)
    prepared_df = datasets["prepared_df"]
    forecaster = load_or_train_forecaster(datasets, config)
    if not forecaster.models_by_horizon:
        raise RuntimeError("Trained forecaster has no fitted models; run main.py first.")

    horizon = int(config.forecast_horizon)
    models = forecaster.models_by_horizon[horizon]
    scaler = forecaster.scalers_by_horizon[horizon]
    feature_columns = forecaster.feature_columns_by_horizon[horizon]
    target_columns = tuple(config.target_columns)
    profile_df = _seasonal_profiles(prepared_df, tuple(config.domain_columns))[config.domain_columns[0]]

    last_ts = prepared_df["timestamp"].iloc[-1]
    future_start = last_ts + pd.Timedelta(hours=1)
    future_end = pd.Timestamp(f"{FORECAST_YEAR}-12-31 23:00:00")
    future_range = pd.date_range(future_start, future_end, freq="h")

    domain_cols = ["timestamp", *list(config.domain_columns)]
    history = prepared_df[domain_cols].tail(HISTORY_WINDOW).reset_index(drop=True)

    time_feature_cols = [
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "season_sin", "season_cos",
    ]

    def _bounded(target: str, value: float, ts: pd.Timestamp) -> float:
        value = float(value)
        if not np.isfinite(value):
            value = _hierarchical_median_lookup(profile_df, target, ts)
        if target == "aqi":
            return float(np.clip(value, 0.0, 500.0))
        if target in {"traffic_flow", "electricity_demand"}:
            return float(max(0.0, value))
        return value

    rows: list[dict] = []
    for step, ts in enumerate(future_range, start=1):
        frame = _feature_frame(history.tail(HISTORY_WINDOW).reset_index(drop=True), config, target_columns)
        last_row = frame.loc[len(frame) - 1, feature_columns].to_frame().T.reset_index(drop=True)
        time_frame = add_time_features(pd.DataFrame([{"timestamp": ts}])).iloc[0]
        for col in time_feature_cols:
            if col in last_row.columns:
                last_row.iloc[0, last_row.columns.get_loc(col)] = time_frame[col]
        scaled = scaler.transform(last_row[feature_columns])
        predictions = {target: float(model.predict(scaled)[0]) for target, model in models.items()}

        next_row = {"timestamp": ts, "time_segment": "future", "step_ahead": step}
        for col in config.domain_columns:
            if col in target_columns:
                seasonal = _hierarchical_median_lookup(profile_df, col, ts)
                blended = MODEL_BLEND * predictions[col] + SEASONAL_BLEND * seasonal
                next_row[col] = _bounded(col, blended, ts)
            else:
                next_row[col] = _hierarchical_median_lookup(profile_df, col, ts)

        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        history = history.tail(HISTORY_WINDOW).reset_index(drop=True)
        rows.append(next_row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate the full-year 2026 forecast with the trained hybrid model.")
    parser.add_argument("--city", default=None, help="Use city-specific data and outputs, e.g. delhi.")
    parser.add_argument("--output", default=None, help="Output CSV path (defaults to outputs/forecast_2026.csv).")
    args = parser.parse_args()

    config = apply_city_config(CONFIG, args.city)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_df = build_2026_forecast_frame(config)
    forecast_df = forecast_df[pd.to_datetime(forecast_df["timestamp"]).dt.year == FORECAST_YEAR].reset_index(drop=True)
    output_path = Path(args.output) if args.output else Path(config.output_dir) / "forecast_2026.csv"
    forecast_df.to_csv(output_path, index=False)

    print(f"Saved {FORECAST_YEAR} forecast: {output_path}")
    print(f"Rows: {len(forecast_df)} | from {forecast_df['timestamp'].min()} to {forecast_df['timestamp'].max()}")
    monthly = forecast_df.groupby(forecast_df["timestamp"].dt.month)[
        ["traffic_flow", "aqi", "temperature", "humidity", "electricity_demand"]
    ].mean().round(2)
    monthly.index = [f"2026-{m:02d}" for m in monthly.index]
    print("\nMonthly means:")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
