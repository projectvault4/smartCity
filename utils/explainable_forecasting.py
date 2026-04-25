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
        self.models_by_horizon: Dict[int, Dict[str, RidgeCV]] = {}
        self.scalers_by_horizon: Dict[int, StandardScaler] = {}
        self.feature_columns_by_horizon: Dict[int, List[str]] = {}
        self.future_target_columns_by_horizon: Dict[int, List[str]] = {}
        self.interval_bounds_by_horizon: Dict[int, Dict[str, Tuple[float, float]]] = {}
        self.validation_metrics_by_horizon: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.primary_horizon: int = int(config.forecast_horizon)

    def _split_supervised_frame(self, supervised_df: pd.DataFrame):
        n = len(supervised_df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        return (
            supervised_df.iloc[:train_end].copy(),
            supervised_df.iloc[train_end:val_end].copy(),
            supervised_df.iloc[val_end:].copy(),
        )

    def _sync_primary_horizon(self, horizon: int) -> None:
        self.primary_horizon = int(horizon)
        self.models = self.models_by_horizon[horizon]
        self.scaler = self.scalers_by_horizon[horizon]
        self.feature_columns = self.feature_columns_by_horizon[horizon]
        self.future_target_columns = self.future_target_columns_by_horizon[horizon]
        self.interval_bounds = self.interval_bounds_by_horizon[horizon]
        self.validation_metrics = self.validation_metrics_by_horizon[horizon]

    def _fit_horizon(self, prepared_df: pd.DataFrame, horizon: int) -> ExplainableForecastArtifacts:
        supervised_df = _build_supervised_frame(
            prepared_df=prepared_df,
            target_columns=self.config.target_columns,
            horizon=horizon,
        )
        future_target_columns = [_future_target_name(target, horizon) for target in self.config.target_columns]
        feature_columns = [col for col in supervised_df.columns if col not in {"timestamp", *future_target_columns}]

        train_df, val_df, test_df = self._split_supervised_frame(supervised_df)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[feature_columns])
        x_val = scaler.transform(val_df[feature_columns])
        x_test = scaler.transform(test_df[feature_columns])

        models: Dict[str, RidgeCV] = {}
        bounds: Dict[str, Tuple[float, float]] = {}
        validation_metrics: Dict[str, Dict[str, float]] = {}
        metrics = {}
        per_target = {}
        test_predictions = {}
        for target in self.config.target_columns:
            y_train = train_df[_future_target_name(target, horizon)].to_numpy()
            y_val = val_df[_future_target_name(target, horizon)].to_numpy()
            y_test = test_df[_future_target_name(target, horizon)].to_numpy()

            model = RidgeCV(alphas=np.logspace(-2, 3, 40))
            model.fit(x_train, y_train)
            models[target] = model

            val_pred = model.predict(x_val)
            test_pred = model.predict(x_test)
            residuals = y_val - val_pred

            validation_metrics[target] = compute_all_metrics(y_val, val_pred)
            low_q, high_q = self.config.uncertainty_quantiles
            bounds[target] = (
                float(np.quantile(residuals, low_q)),
                float(np.quantile(residuals, high_q)),
            )

            metrics[target] = compute_all_metrics(y_test, test_pred)
            per_target[target] = {target: metrics[target]}
            test_predictions[target] = test_pred

        self.models_by_horizon[horizon] = models
        self.scalers_by_horizon[horizon] = scaler
        self.feature_columns_by_horizon[horizon] = feature_columns
        self.future_target_columns_by_horizon[horizon] = future_target_columns
        self.interval_bounds_by_horizon[horizon] = bounds
        self.validation_metrics_by_horizon[horizon] = validation_metrics

        metrics_df = pd.DataFrame(metrics).T
        prediction_frame = pd.DataFrame(test_predictions, index=test_df["timestamp"])
        return ExplainableForecastArtifacts(
            metrics=metrics_df,
            per_target_metrics=per_target,
            test_predictions=prediction_frame,
        )

    def fit(self, prepared_df: pd.DataFrame) -> ExplainableForecastArtifacts:
        artifacts = self._fit_horizon(prepared_df, self.config.forecast_horizon)
        self._sync_primary_horizon(self.config.forecast_horizon)
        return artifacts

    def fit_multi_horizon(self, prepared_df: pd.DataFrame, horizons: List[int]) -> Dict[int, ExplainableForecastArtifacts]:
        artifacts = {}
        for horizon in sorted({int(h) for h in horizons if int(h) > 0}):
            artifacts[horizon] = self._fit_horizon(prepared_df, horizon)
        if self.config.forecast_horizon in artifacts:
            self._sync_primary_horizon(self.config.forecast_horizon)
        return artifacts

    def predict_from_prepared(self, prepared_df: pd.DataFrame, horizon: int | None = None) -> Dict[str, Dict[str, float]]:
        horizon = self.primary_horizon if horizon is None else int(horizon)
        if horizon not in self.models_by_horizon:
            raise ValueError(f"Horizon {horizon} has not been fitted yet")
        feature_columns = self.feature_columns_by_horizon[horizon]
        scaler = self.scalers_by_horizon[horizon]
        latest_features = prepared_df[feature_columns].tail(1)
        x_latest = scaler.transform(latest_features)
        forecast: Dict[str, Dict[str, float]] = {}

        for target, model in self.models_by_horizon[horizon].items():
            point_estimate = float(model.predict(x_latest)[0])
            low_residual, high_residual = self.interval_bounds_by_horizon[horizon][target]
            forecast[target] = {
                "prediction": point_estimate,
                "lower": point_estimate + low_residual,
                "upper": point_estimate + high_residual,
            }
        return forecast

    def predict_multi_horizon(self, prepared_df: pd.DataFrame, horizons: List[int]) -> Dict[int, Dict[str, Dict[str, float]]]:
        return {int(horizon): self.predict_from_prepared(prepared_df, horizon=int(horizon)) for horizon in horizons}

    def explain_latest_prediction(self, prepared_df: pd.DataFrame, top_k: int = 3, horizon: int | None = None) -> Dict[str, List[str]]:
        horizon = self.primary_horizon if horizon is None else int(horizon)
        feature_columns = self.feature_columns_by_horizon[horizon]
        scaler = self.scalers_by_horizon[horizon]
        latest_features = prepared_df[feature_columns].tail(1)
        scaled_latest = scaler.transform(latest_features)[0]
        explanations: Dict[str, List[str]] = {}

        for target, model in self.models_by_horizon[horizon].items():
            contributions = model.coef_ * scaled_latest
            ranked_idx = np.argsort(np.abs(contributions))[::-1]
            lines = []
            seen = set()
            for idx in ranked_idx:
                feature_name = _readable_feature_name(feature_columns[idx])
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
            forecaster = pickle.load(f)
        if not hasattr(forecaster, "models_by_horizon"):
            forecaster.models_by_horizon = {int(forecaster.config.forecast_horizon): forecaster.models}
            forecaster.scalers_by_horizon = {int(forecaster.config.forecast_horizon): forecaster.scaler}
            forecaster.feature_columns_by_horizon = {int(forecaster.config.forecast_horizon): forecaster.feature_columns}
            forecaster.future_target_columns_by_horizon = {
                int(forecaster.config.forecast_horizon): forecaster.future_target_columns
            }
            forecaster.interval_bounds_by_horizon = {int(forecaster.config.forecast_horizon): forecaster.interval_bounds}
            forecaster.validation_metrics_by_horizon = {
                int(forecaster.config.forecast_horizon): forecaster.validation_metrics
            }
            forecaster.primary_horizon = int(forecaster.config.forecast_horizon)
        return forecaster


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
