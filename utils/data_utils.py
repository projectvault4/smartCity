from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_multidomain_data(num_steps: int, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    idx = pd.date_range("2023-01-01", periods=num_steps, freq="h")
    t = np.arange(num_steps)

    daily = np.sin(2 * np.pi * t / 24)
    weekly = np.sin(2 * np.pi * t / (24 * 7))
    annual = np.sin(2 * np.pi * t / (24 * 365))

    temperature = 24 + 8 * daily + 3 * annual + rng.normal(0, 1.2, num_steps)
    humidity = 55 - 12 * daily + 6 * weekly + rng.normal(0, 2.2, num_steps)
    wind_speed = 8 + 1.5 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 0.7, num_steps)

    commute_profile = np.maximum(0, np.sin(2 * np.pi * (t - 6) / 24)) + np.maximum(0, np.sin(2 * np.pi * (t - 16) / 24))
    traffic = 120 + 42 * commute_profile + 18 * weekly + 0.35 * humidity + rng.normal(0, 4.5, num_steps)

    electricity = (
        210
        + 2.8 * temperature
        + 0.42 * traffic
        + 20 * np.maximum(0, daily)
        + 10 * weekly
        + rng.normal(0, 5.5, num_steps)
    )

    pollution = (
        40
        + 0.32 * traffic
        + 0.09 * electricity
        - 1.7 * wind_speed
        + 0.18 * humidity
        + 4.5 * np.maximum(0, daily)
        + rng.normal(0, 3.0, num_steps)
    )

    drift_start = int(num_steps * 0.72)
    pollution[drift_start:] += 0.13 * electricity[drift_start:] + 0.12 * temperature[drift_start:]
    traffic[drift_start:] += 14 * np.sin(2 * np.pi * t[drift_start:] / 12)
    electricity[drift_start:] += 18 * np.maximum(0, np.sin(2 * np.pi * (t[drift_start:] - 4) / 24))

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "traffic_flow": traffic,
            "aqi": pollution,
            "electricity_demand": electricity,
            "temperature": temperature,
            "humidity": humidity,
        }
    )
    return df


def load_input_dataframe(config) -> pd.DataFrame:
    data_path = Path(config.data_file)
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_synthetic_multidomain_data(config.num_steps, config.random_seed)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    expected_columns = {"timestamp", *config.domain_columns}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing_columns)}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return clean_fuzzy_data(df, config)


def clean_fuzzy_data(df: pd.DataFrame, config) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_columns = [col for col in config.domain_columns if col in cleaned.columns]

    # Remove duplicate timestamps and preserve the latest reading.
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # Convert bad strings or corrupt values into missing values first.
    for col in numeric_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Time-aware interpolation repairs small gaps without leaking future labels across the full dataset.
    cleaned[numeric_columns] = cleaned[numeric_columns].interpolate(method="linear", limit_direction="both")

    # Clip extreme outliers with an IQR fence to make the training signal more stable.
    for col in numeric_columns:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)

    cleaned[numeric_columns] = cleaned[numeric_columns].ffill().bfill()
    return cleaned


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    hour = ts.dt.hour
    day_of_week = ts.dt.dayofweek
    day_of_year = ts.dt.dayofyear

    out["hour"] = hour
    out["day_of_week"] = day_of_week
    out["month"] = ts.dt.month
    out["is_weekend"] = (day_of_week >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    out["season_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    out["season_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    return out


def add_lag_features(df: pd.DataFrame, columns: List[str], lag_steps: Tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for lag in lag_steps:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, columns: List[str], windows: Tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for window in windows:
            roll = out[col].rolling(window=window)
            out[f"{col}_roll_mean_{window}"] = roll.mean()
            out[f"{col}_roll_std_{window}"] = roll.std()
            out[f"{col}_roll_min_{window}"] = roll.min()
            out[f"{col}_roll_max_{window}"] = roll.max()
    return out


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler() if config.scaler_name == "standard" else MinMaxScaler()
        self.target_scaler = StandardScaler() if config.scaler_name == "standard" else MinMaxScaler()
        self.feature_columns: List[str] = []
        self.target_columns = list(config.target_columns)

    def prepare_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = add_time_features(raw_df)
        df = add_lag_features(df, list(self.config.domain_columns), self.config.lag_steps)
        df = add_rolling_features(df, list(self.config.domain_columns), self.config.rolling_windows)
        df = df.dropna().reset_index(drop=True)
        return df

    def split_dataframe(self, df: pd.DataFrame):
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

    def fit_transform(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        ignore = {"timestamp"}
        self.feature_columns = [col for col in train_df.columns if col not in ignore]

        x_train = self.scaler.fit_transform(train_df[self.feature_columns])
        x_val = self.scaler.transform(val_df[self.feature_columns])
        x_test = self.scaler.transform(test_df[self.feature_columns])

        y_train = self.target_scaler.fit_transform(train_df[self.target_columns])
        y_val = self.target_scaler.transform(val_df[self.target_columns])
        y_test = self.target_scaler.transform(test_df[self.target_columns])
        return x_train, y_train, x_val, y_val, x_test, y_test

    def transform_dataframe(self, df: pd.DataFrame):
        x = self.scaler.transform(df[self.feature_columns])
        y = self.target_scaler.transform(df[self.target_columns])
        return x, y

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(1, -1)
        return self.target_scaler.inverse_transform(y_scaled)


def build_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int, horizon: int):
    xs, ys = [], []
    for end_idx in range(seq_len, len(features) - horizon + 1):
        start_idx = end_idx - seq_len
        xs.append(features[start_idx:end_idx])
        ys.append(targets[end_idx + horizon - 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def create_datasets(config, raw_df: pd.DataFrame):
    processor = DataProcessor(config)
    prepared_df = processor.prepare_dataframe(raw_df)
    train_df, val_df, test_df = processor.split_dataframe(prepared_df)
    x_train, y_train, x_val, y_val, x_test, y_test = processor.fit_transform(train_df, val_df, test_df)

    train_seq = build_sequences(x_train, y_train, config.seq_len, config.forecast_horizon)
    val_seq = build_sequences(x_val, y_val, config.seq_len, config.forecast_horizon)
    test_seq = build_sequences(x_test, y_test, config.seq_len, config.forecast_horizon)

    return {
        "processor": processor,
        "prepared_df": prepared_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_seq": train_seq,
        "val_seq": val_seq,
        "test_seq": test_seq,
    }


def rolling_mean(values: List[float], window: int) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()
