"""Counterfactual what-if scenarios run through the full deployed prediction chain.

A scenario edits the observed history (e.g. traffic x 0.8 from shortly before
the test window onward), rebuilds every engineered feature, re-runs the SARIMA
expert walk-forward, the network, the calibration and the SARIMA blend, and
compares the result with the unmodified forecast.

Training rows are never touched, so every fitted scaler, the SARIMA parameters
and the network weights stay identical; only the inputs change. The result is
the *model's* sensitivity, not a proven causal effect, and the dashboard says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = [
    {"id": "traffic_-30", "driver": "traffic_flow", "kind": "scale", "value": -0.30, "label": "Traffic -30%"},
    {"id": "traffic_-20", "driver": "traffic_flow", "kind": "scale", "value": -0.20, "label": "Traffic -20%"},
    {"id": "traffic_-10", "driver": "traffic_flow", "kind": "scale", "value": -0.10, "label": "Traffic -10%"},
    {"id": "traffic_+10", "driver": "traffic_flow", "kind": "scale", "value": 0.10, "label": "Traffic +10%"},
    {"id": "traffic_+20", "driver": "traffic_flow", "kind": "scale", "value": 0.20, "label": "Traffic +20%"},
    {"id": "traffic_+30", "driver": "traffic_flow", "kind": "scale", "value": 0.30, "label": "Traffic +30%"},
    {"id": "temperature_-2", "driver": "temperature", "kind": "shift", "value": -2.0, "label": "Temperature -2 C"},
    {"id": "temperature_-1", "driver": "temperature", "kind": "shift", "value": -1.0, "label": "Temperature -1 C"},
    {"id": "temperature_+1", "driver": "temperature", "kind": "shift", "value": 1.0, "label": "Temperature +1 C"},
    {"id": "temperature_+2", "driver": "temperature", "kind": "shift", "value": 2.0, "label": "Temperature +2 C"},
    {"id": "humidity_+10", "driver": "humidity", "kind": "scale", "value": 0.10, "label": "Humidity +10%"},
    {"id": "humidity_-10", "driver": "humidity", "kind": "scale", "value": -0.10, "label": "Humidity -10%"},
]


def _apply(raw_df: pd.DataFrame, scenario: dict, start: pd.Timestamp) -> pd.DataFrame:
    df = raw_df.copy()
    mask = pd.to_datetime(df["timestamp"]) >= start
    col = scenario["driver"]
    if scenario["kind"] == "scale":
        df.loc[mask, col] = df.loc[mask, col] * (1.0 + scenario["value"])
    else:
        df.loc[mask, col] = df.loc[mask, col] + scenario["value"]
    return df


def run_whatif(runtime, baseline) -> dict:
    """baseline: SplitPredictions for the unmodified test window."""
    test_ts = pd.to_datetime(baseline.timestamps).reset_index(drop=True)
    train_end_ts = pd.to_datetime(runtime.datasets["train_df"]["timestamp"]).iloc[-1]
    # Start the edit far enough back that every lag/rolling input in the test
    # window (max lag 179 h) sees the scenario, but never inside training.
    start = max(test_ts.iloc[0] - pd.Timedelta(hours=200), train_end_ts + pd.Timedelta(hours=1))
    hours = np.asarray([t.hour for t in test_ts])
    out = {
        "method": "full-chain counterfactual (features, SARIMA, network, calibration, blend re-run)",
        "edit_starts": start.isoformat(),
        "window": {"start": test_ts.iloc[0].isoformat(), "end": test_ts.iloc[-1].isoformat(), "hours": int(len(test_ts))},
        "caveat": "Model sensitivity learned from Feb-Aug 2025 Bengaluru data, not a measured causal effect.",
        "scenarios": [],
    }
    for sc in SCENARIOS:
        modified = runtime.predict_with_raw(_apply(runtime.raw_df, sc, start), "test")
        if len(modified.final) != len(baseline.final):
            raise RuntimeError("Scenario changed the number of test rows; aborting.")
        effects = {}
        for i, target in enumerate(runtime.targets):
            delta = modified.final[:, i] - baseline.final[:, i]
            net_delta = modified.network[:, i] - baseline.network[:, i]
            base_mean = float(np.mean(baseline.final[:, i]))
            by_hour = {str(h): round(float(delta[hours == h].mean()), 4) for h in range(24) if (hours == h).any()}
            effects[target] = {
                "mean_change": round(float(delta.mean()), 4),
                "mean_change_pct": round(float(delta.mean() / base_mean * 100), 3) if base_mean else None,
                "p05": round(float(np.percentile(delta, 5)), 4),
                "p95": round(float(np.percentile(delta, 95)), 4),
                "network_mean_change": round(float(net_delta.mean()), 4),
                "by_hour": by_hour,
            }
        out["scenarios"].append({**sc, "effects": effects})
    out["daily_associations"] = daily_associations(runtime.raw_df, train_end_ts)
    out["findings"] = _findings(out)
    return out


def _findings(result: dict) -> list[str]:
    lines = []
    by_id = {s["id"]: s for s in result["scenarios"]}
    t20 = by_id.get("traffic_-20")
    if t20:
        aqi = t20["effects"].get("aqi", {})
        ele = t20["effects"].get("electricity_demand", {})
        lines.append(
            f"Hour-ahead model: cutting traffic 20% changes the AQI forecast by {aqi.get('mean_change', 0):+.3f} AQI points "
            f"and electricity by {ele.get('mean_change', 0):+.1f} MW ({ele.get('mean_change_pct', 0):+.2f}%). "
            "One hour ahead, each signal is driven by its own recent history; cross-domain effects are negligible."
        )
    tp2 = by_id.get("temperature_+2")
    if tp2:
        ele = tp2["effects"].get("electricity_demand", {})
        lines.append(
            f"Hour-ahead model: a 2 C warmer spell changes electricity demand by {ele.get('mean_change', 0):+.1f} MW "
            f"({ele.get('mean_change_pct', 0):+.2f}%)."
        )
    for a in result.get("daily_associations", []):
        lines.append(a["sentence"])
    return lines


DAILY_PAIRS = [
    ("temperature", "electricity_demand", 1.0, "C", "MW"),
    ("traffic_flow", "aqi", 1000.0, "veh/hr", "AQI points"),
    ("traffic_flow", "electricity_demand", 1000.0, "veh/hr", "MW"),
    ("humidity", "aqi", 10.0, "%", "AQI points"),
]
LABELS = {"temperature": "temperature", "electricity_demand": "electricity demand", "traffic_flow": "traffic", "aqi": "AQI", "humidity": "humidity"}


def daily_associations(raw_df: pd.DataFrame, train_end: pd.Timestamp, n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Observed day-level associations (training period only) with a block-bootstrap 95% CI.

    Day-level is the resolution at which AQI and temperature were recorded, so
    this is where cross-domain structure can actually show up. Correlational.
    """
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] <= train_end]
    daily = df.groupby(df["timestamp"].dt.date).mean(numeric_only=True)
    rng = np.random.default_rng(seed)
    out = []
    for x, y, per, xu, yu in DAILY_PAIRS:
        if x not in daily or y not in daily:
            continue
        d = daily[[x, y]].dropna()
        xv, yv = d[x].to_numpy(), d[y].to_numpy()
        slope = float(np.polyfit(xv, yv, 1)[0]) * per
        corr = float(np.corrcoef(xv, yv)[0, 1])
        n, block = len(d), 7
        boots = []
        for _ in range(n_boot):
            starts = rng.integers(0, n - block, size=int(np.ceil(n / block)))
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            if np.std(xv[idx]) < 1e-9:
                continue
            boots.append(float(np.polyfit(xv[idx], yv[idx], 1)[0]) * per)
        lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
        significant = lo is not None and (lo > 0 or hi < 0)
        per_txt = f"{per:g}{'' if xu == '%' else ' '}{xu}"
        sentence = (
            f"Observed daily data (Feb-Jun 2025): a day with {LABELS[x]} higher by {per_txt} averages "
            f"{slope:+.1f} {yu}{'' if y == 'aqi' else ' ' + LABELS[y]} (95% CI {lo:+.1f} to {hi:+.1f}; r={corr:.2f}). "
            + ("Clear association." if significant else "Not distinguishable from zero.")
        )
        out.append({"x": x, "y": y, "per": per, "x_unit": xu, "y_unit": yu, "slope": slope, "ci95": [lo, hi],
                    "correlation": corr, "days": n, "significant": significant, "sentence": sentence})
    return out
