"""Leakage-free anomaly scoring with a measured evaluation.

Differences from the dashboard's exploratory detector (utils/anomaly_detection.py):
  * scaler, Isolation Forest and autoencoder are fit on the TRAINING split only;
  * scores are mapped through the training score distribution (empirical CDF),
    so a "0.99" means "rarer than 99% of training hours";
  * the alert threshold is the 98th percentile of the VALIDATION scores, so the
    false-alarm rate is set on data the detector has not been fit on and then
    *measured* on the held-out test window;
  * it is compared with a simple rolling z-score baseline;
  * detection power is measured by injecting synthetic incidents into a copy
    of the test window (traffic surge, AQI jump, demand spike, sensor freeze)
    and counting how many are caught.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from utils.anomaly_detection import prepare_anomaly_feature_frame, select_anomaly_features

INJECTIONS = [
    {"id": "traffic_surge", "label": "Traffic surge (+60% for 2 h)", "column": "traffic_flow", "kind": "scale", "value": 0.6, "hours": 2},
    {"id": "aqi_jump", "label": "AQI jump (+30 for 6 h)", "column": "aqi", "kind": "shift", "value": 30.0, "hours": 6},
    {"id": "demand_spike", "label": "Demand spike (+30% for 2 h)", "column": "electricity_demand", "kind": "scale", "value": 0.3, "hours": 2},
    {"id": "sensor_freeze", "label": "Traffic sensor frozen (flat for 6 h)", "column": "traffic_flow", "kind": "freeze", "value": 0.0, "hours": 6},
]


class LeakFreeDetector:
    def __init__(self, raw_df: pd.DataFrame, config, train_end: pd.Timestamp, val_end: pd.Timestamp, seed: int = 42):
        self.config = config
        self.train_end = train_end
        self.val_end = val_end
        feature_df, feature_columns = prepare_anomaly_feature_frame(raw_df, config)
        ts = pd.to_datetime(feature_df["timestamp"])
        train_mask = (ts <= train_end).to_numpy().copy()
        train_mask[:24] = False  # warm-up rows have zero-filled lags
        self.features = select_anomaly_features(feature_df[train_mask].reset_index(drop=True), feature_columns)
        x_train = feature_df.loc[train_mask, self.features].to_numpy(dtype=float)
        self.scaler = StandardScaler().fit(x_train)
        xs = self.scaler.transform(x_train)
        self.iso = IsolationForest(n_estimators=300, max_samples=min(256, len(xs)), random_state=seed).fit(xs)
        n_feat = xs.shape[1]
        self.ae = MLPRegressor(
            hidden_layer_sizes=(max(4, min(32, n_feat // 2)), max(2, min(12, n_feat // 4)), max(4, min(32, n_feat // 2))),
            max_iter=300, early_stopping=True, n_iter_no_change=12, random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.ae.fit(xs, xs)
        self._train_iso = np.sort(-self.iso.decision_function(xs))
        self._train_ae = np.sort(np.mean((xs - self.ae.predict(xs)) ** 2, axis=1))
        self._train_z = None
        val_scores = self.score(raw_df)
        vts = pd.to_datetime(val_scores["timestamp"])
        val_mask = (vts > train_end) & (vts <= val_end)
        self.threshold = float(np.quantile(val_scores.loc[val_mask, "score"], 0.98))
        self.z_threshold = 3.0

    @staticmethod
    def _cdf(sorted_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(sorted_ref, values, side="right") / len(sorted_ref)

    def score(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        feature_df, _ = prepare_anomaly_feature_frame(raw_df, self.config)
        xs = self.scaler.transform(feature_df[self.features].to_numpy(dtype=float))
        iso = self._cdf(self._train_iso, -self.iso.decision_function(xs))
        ae = self._cdf(self._train_ae, np.mean((xs - self.ae.predict(xs)) ** 2, axis=1))
        out = pd.DataFrame({"timestamp": feature_df["timestamp"], "iso": iso, "ae": ae})
        out["score"] = 0.55 * iso + 0.45 * ae
        z_cols = [c for c in feature_df.columns if c.endswith("_roll_z_24")]
        out["max_abs_z"] = feature_df[z_cols].abs().max(axis=1).to_numpy()
        z_by = feature_df[z_cols].abs()
        out["z_driver"] = z_by.idxmax(axis=1).str.replace("_roll_z_24", "", regex=False).to_numpy()
        return out

    def flags(self, scores: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (scores["score"].to_numpy() >= self.threshold), (scores["max_abs_z"].to_numpy() >= self.z_threshold)


def _inject(raw_df: pd.DataFrame, inj: dict, starts: list[pd.Timestamp]) -> tuple[pd.DataFrame, list[list[pd.Timestamp]]]:
    df = raw_df.copy()
    ts = pd.to_datetime(df["timestamp"])
    windows = []
    for start in starts:
        hours = [start + pd.Timedelta(hours=h) for h in range(inj["hours"])]
        mask = ts.isin(hours)
        col = inj["column"]
        if inj["kind"] == "scale":
            df.loc[mask, col] = df.loc[mask, col] * (1 + inj["value"])
        elif inj["kind"] == "shift":
            df.loc[mask, col] = df.loc[mask, col] + inj["value"]
        elif inj["kind"] == "freeze":
            prev = df.loc[ts == start - pd.Timedelta(hours=1), col]
            if len(prev):
                df.loc[mask, col] = float(prev.iloc[0])
        windows.append(hours)
    return df, windows


def evaluate(detector: LeakFreeDetector, raw_df: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp,
             residuals: pd.DataFrame | None = None, seed: int = 7) -> dict:
    clean = detector.score(raw_df)
    cts = pd.to_datetime(clean["timestamp"])
    test_mask = ((cts >= test_start) & (cts <= test_end)).to_numpy()
    model_flag, z_flag = detector.flags(clean)
    weeks = test_mask.sum() / 168.0
    result = {
        "threshold": round(detector.threshold, 4),
        "threshold_rule": "98th percentile of validation-window scores (detector fit on training only)",
        "test_hours": int(test_mask.sum()),
        "clean_window": {
            "model_flag_rate": float(model_flag[test_mask].mean()),
            "model_flags_per_week": float(model_flag[test_mask].sum() / weeks),
            "zscore_flag_rate": float(z_flag[test_mask].mean()),
            "zscore_flags_per_week": float(z_flag[test_mask].sum() / weeks),
        },
        "injection_tests": [],
    }

    # Do flagged hours coincide with the forecaster being surprised?
    if residuals is not None:
        merged = clean.merge(residuals, on="timestamp", how="inner")
        if len(merged):
            m_flag = merged["score"].to_numpy() >= detector.threshold
            err = merged["norm_abs_error"].to_numpy()
            result["forecast_error_link"] = {
                "mean_norm_error_flagged": float(err[m_flag].mean()) if m_flag.any() else None,
                "mean_norm_error_unflagged": float(err[~m_flag].mean()),
                "flagged_hours": int(m_flag.sum()),
            }

    rng = np.random.default_rng(seed)
    test_hours = cts[test_mask].reset_index(drop=True)
    candidates = test_hours[24:-8].reset_index(drop=True)
    for inj in INJECTIONS:
        # 12 incidents spaced >= 30 h apart so their effects do not overlap.
        picks = []
        order = rng.permutation(len(candidates))
        for i in order:
            t = candidates.iloc[i]
            if all(abs((t - p).total_seconds()) >= 30 * 3600 for p in picks):
                picks.append(t)
            if len(picks) == 12:
                break
        injected_df, windows = _inject(raw_df, inj, sorted(picks))
        sc = detector.score(injected_df)
        sts = pd.to_datetime(sc["timestamp"])
        mf, zf = detector.flags(sc)
        caught_m = caught_z = 0
        for hours in windows:
            span = sts.isin(hours + [hours[-1] + pd.Timedelta(hours=1)]).to_numpy()
            caught_m += bool(mf[span].any())
            caught_z += bool(zf[span].any())
        result["injection_tests"].append(
            {
                "id": inj["id"],
                "label": inj["label"],
                "incidents": len(windows),
                "detected_by_model": caught_m,
                "detected_by_zscore": caught_z,
                "recall_model": caught_m / len(windows),
                "recall_zscore": caught_z / len(windows),
            }
        )
    tot = sum(r["incidents"] for r in result["injection_tests"])
    result["overall_recall_model"] = sum(r["detected_by_model"] for r in result["injection_tests"]) / tot
    result["overall_recall_zscore"] = sum(r["detected_by_zscore"] for r in result["injection_tests"]) / tot
    result["scores"] = clean.loc[test_mask, ["timestamp", "score", "max_abs_z", "z_driver"]].reset_index(drop=True)
    return result
