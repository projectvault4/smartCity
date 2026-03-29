from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from utils.metrics import compute_all_metrics, compute_metrics_by_target


def _future_target_name(target: str, horizon: int) -> str:
    return f"{target}_t_plus_{horizon}"


def humanize_target_name(target: str) -> str:
    labels = {
        "traffic_flow": "Traffic Flow",
        "aqi": "AQI",
        "temperature": "Temperature",
        "electricity_demand": "Electricity Demand",
    }
    return labels.get(target, target.replace("_", " ").title())


def _readable_feature_name(feature_name: str) -> str:
    if feature_name.endswith("_lag_1"):
        return f"{feature_name[:-6].replace('_', ' ')} in the last hour"
    if "_lag_" in feature_name:
        base, lag = feature_name.rsplit("_lag_", 1)
        return f"{base.replace('_', ' ')} {lag} hours ago"
    if "_roll_mean_" in feature_name:
        base, window = feature_name.rsplit("_roll_mean_", 1)
        return f"{base.replace('_', ' ')} {window}-hour average"
    if "_roll_std_" in feature_name:
        base, window = feature_name.rsplit("_roll_std_", 1)
        return f"{base.replace('_', ' ')} {window}-hour variation"
    if "_roll_min_" in feature_name:
        base, window = feature_name.rsplit("_roll_min_", 1)
        return f"{base.replace('_', ' ')} {window}-hour low"
    if "_roll_max_" in feature_name:
        base, window = feature_name.rsplit("_roll_max_", 1)
        return f"{base.replace('_', ' ')} {window}-hour high"

    replacements = {
        "hour_sin": "time of day pattern",
        "hour_cos": "time of day pattern",
        "dow_sin": "day of week pattern",
        "dow_cos": "day of week pattern",
        "season_sin": "season pattern",
        "season_cos": "season pattern",
        "is_weekend": "weekend effect",
    }
    return replacements.get(feature_name, feature_name.replace("_", " "))


def _direction_text(value: float) -> str:
    return "pushed upward" if value >= 0 else "pulled downward"


def _build_supervised_frame(prepared_df: pd.DataFrame, target_columns: Tuple[str, ...], horizon: int) -> pd.DataFrame:
    df = prepared_df.copy()
    for target in target_columns:
        df[_future_target_name(target, horizon)] = df[target].shift(-horizon)
    return df.dropna().reset_index(drop=True)


@dataclass
class ExplainableForecastArtifacts:
    metrics: pd.DataFrame
    per_target_metrics: Dict[str, Dict[str, Dict[str, float]]]
    test_predictions: pd.DataFrame


class ExplainableTimeSeriesForecaster:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.future_target_columns: List[str] = []
        self.models: Dict[str, RidgeCV] = {}
        self.interval_bounds: Dict[str, Tuple[float, float]] = {}
        self.validation_metrics: Dict[str, Dict[str, float]] = {}

    def _split_supervised_frame(self, supervised_df: pd.DataFrame):
        n = len(supervised_df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        return (
            supervised_df.iloc[:train_end].copy(),
            supervised_df.iloc[train_end:val_end].copy(),
            supervised_df.iloc[val_end:].copy(),
        )

    def fit(self, prepared_df: pd.DataFrame) -> ExplainableForecastArtifacts:
        supervised_df = _build_supervised_frame(
            prepared_df=prepared_df,
            target_columns=self.config.target_columns,
            horizon=self.config.forecast_horizon,
        )
        self.future_target_columns = [
            _future_target_name(target, self.config.forecast_horizon) for target in self.config.target_columns
        ]
        self.feature_columns = [
            col for col in supervised_df.columns if col not in {"timestamp", *self.future_target_columns}
        ]

        train_df, val_df, test_df = self._split_supervised_frame(supervised_df)
        x_train = self.scaler.fit_transform(train_df[self.feature_columns])
        x_val = self.scaler.transform(val_df[self.feature_columns])
        x_test = self.scaler.transform(test_df[self.feature_columns])

        metrics = {}
        per_target = {}
        test_predictions = {}
        for target in self.config.target_columns:
            y_train = train_df[_future_target_name(target, self.config.forecast_horizon)].to_numpy()
            y_val = val_df[_future_target_name(target, self.config.forecast_horizon)].to_numpy()
            y_test = test_df[_future_target_name(target, self.config.forecast_horizon)].to_numpy()

            model = RidgeCV(alphas=np.logspace(-2, 3, 40))
            model.fit(x_train, y_train)
            self.models[target] = model

            val_pred = model.predict(x_val)
            test_pred = model.predict(x_test)
            residuals = y_val - val_pred

            self.validation_metrics[target] = compute_all_metrics(y_val, val_pred)
            self.interval_bounds[target] = (
                float(np.quantile(residuals, 0.1)),
                float(np.quantile(residuals, 0.9)),
            )

            metrics[target] = compute_all_metrics(y_test, test_pred)
            per_target[target] = {target: metrics[target]}
            test_predictions[target] = test_pred

        metrics_df = pd.DataFrame(metrics).T
        prediction_frame = pd.DataFrame(test_predictions, index=test_df["timestamp"])
        return ExplainableForecastArtifacts(
            metrics=metrics_df,
            per_target_metrics=per_target,
            test_predictions=prediction_frame,
        )

    def predict_from_prepared(self, prepared_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        latest_features = prepared_df[self.feature_columns].tail(1)
        x_latest = self.scaler.transform(latest_features)
        forecast: Dict[str, Dict[str, float]] = {}

        for target, model in self.models.items():
            point_estimate = float(model.predict(x_latest)[0])
            low_residual, high_residual = self.interval_bounds[target]
            forecast[target] = {
                "prediction": point_estimate,
                "lower": point_estimate + low_residual,
                "upper": point_estimate + high_residual,
            }
        return forecast

    def explain_latest_prediction(self, prepared_df: pd.DataFrame, top_k: int = 3) -> Dict[str, List[str]]:
        latest_features = prepared_df[self.feature_columns].tail(1)
        scaled_latest = self.scaler.transform(latest_features)[0]
        explanations: Dict[str, List[str]] = {}

        for target, model in self.models.items():
            contributions = model.coef_ * scaled_latest
            ranked_idx = np.argsort(np.abs(contributions))[::-1]
            lines = []
            seen = set()
            for idx in ranked_idx:
                feature_name = _readable_feature_name(self.feature_columns[idx])
                if feature_name in seen:
                    continue
                seen.add(feature_name)
                lines.append(
                    f"{feature_name.title()} {_direction_text(contributions[idx])} the {humanize_target_name(target)} forecast."
                )
                if len(lines) == top_k:
                    break
            explanations[target] = lines or ["Recent conditions are stable, so the forecast stays close to the recent pattern."]
        return explanations

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ExplainableTimeSeriesForecaster":
        with open(path, "rb") as f:
            return pickle.load(f)


def classify_forecast_levels(prediction: Dict[str, float], recent_df: pd.DataFrame) -> Dict[str, str]:
    traffic_recent = recent_df["traffic_flow"]
    demand_recent = recent_df["electricity_demand"]

    if prediction["traffic_flow"] >= float(traffic_recent.quantile(0.8)):
        traffic_label = "Busy"
    elif prediction["traffic_flow"] <= float(traffic_recent.quantile(0.2)):
        traffic_label = "Light"
    else:
        traffic_label = "Normal"

    aqi_value = prediction["aqi"]
    if aqi_value <= 50:
        aqi_label = "Good"
    elif aqi_value <= 100:
        aqi_label = "Moderate"
    elif aqi_value <= 150:
        aqi_label = "Unhealthy for sensitive groups"
    else:
        aqi_label = "Unhealthy"

    temperature_recent = recent_df["temperature"]
    if prediction["temperature"] >= float(temperature_recent.quantile(0.8)):
        weather_label = "Warm"
    elif prediction["temperature"] <= float(temperature_recent.quantile(0.2)):
        weather_label = "Cool"
    else:
        weather_label = "Stable"

    if prediction["electricity_demand"] >= float(demand_recent.quantile(0.8)):
        electricity_label = "High"
    elif prediction["electricity_demand"] <= float(demand_recent.quantile(0.2)):
        electricity_label = "Low"
    else:
        electricity_label = "Normal"

    return {
        "traffic_flow": traffic_label,
        "aqi": aqi_label,
        "temperature": weather_label,
        "electricity_demand": electricity_label,
    }


def build_plain_language_summary(prediction: Dict[str, float], recent_df: pd.DataFrame) -> List[str]:
    latest = recent_df.iloc[-1]
    summary = []
    for target in ("traffic_flow", "aqi", "temperature", "electricity_demand"):
        current = float(latest[target])
        future = float(prediction[target])
        delta = future - current
        if abs(delta) < 1:
            trend = "should stay about the same"
        elif delta > 0:
            trend = f"is likely to rise by about {delta:.0f}"
        else:
            trend = f"is likely to fall by about {abs(delta):.0f}"
        summary.append(f"{humanize_target_name(target)} {trend} compared with the latest reading.")
    return summary
