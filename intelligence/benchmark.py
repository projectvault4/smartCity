"""Honest skill benchmark for the deployed hybrid.

Every comparison uses the same held-out timestamps. Baselines:
  - persistence      : y[t-1]   (what you get by assuming nothing changes)
  - seasonal naive   : y[t-24]  (same hour yesterday)
  - SARIMA           : the one-step statistical expert (fit on train, walk-forward)

Skill = 1 - MAE_model / MAE_baseline (positive = model is better).
Rolling-origin: the window is cut into non-overlapping blocks so we can report
how often the model wins, not just one average. A Diebold-Mariano test with a
Newey-West variance (lag 24, hourly autocorrelation) says whether the
difference is statistically meaningful.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TARGET_LABELS = {
    "traffic_flow": "Traffic flow",
    "aqi": "AQI",
    "temperature": "Temperature",
    "electricity_demand": "Electricity demand",
}
TARGET_UNITS = {"traffic_flow": "veh/hr", "aqi": "AQI", "temperature": "C", "electricity_demand": "MW"}


def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _skill(model_mae: float, base_mae: float) -> float | None:
    if base_mae <= 1e-12:
        return None
    return float(1.0 - model_mae / base_mae)


def diebold_mariano(e_model: np.ndarray, e_base: np.ndarray, lag: int = 24) -> dict:
    """Two-sided DM test on absolute-error loss with a Newey-West variance."""
    d = np.abs(e_model) - np.abs(e_base)
    n = len(d)
    mean_d = float(d.mean())
    dc = d - mean_d
    gamma0 = float(np.dot(dc, dc) / n)
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * float(np.dot(dc[k:], dc[:-k]) / n)
    if var <= 0:
        return {"statistic": None, "p_value": None}
    stat = mean_d / math.sqrt(var / n)
    p = math.erfc(abs(stat) / math.sqrt(2.0))
    return {"statistic": round(stat, 3), "p_value": float(p)}


def _baselines(full_actual: pd.DataFrame, timestamps: pd.Series, target: str) -> dict[str, np.ndarray]:
    series = full_actual.set_index("timestamp")[target]
    ts = pd.to_datetime(timestamps)
    return {
        "persistence": series.reindex(ts - pd.Timedelta(hours=1)).to_numpy(dtype=float),
        "seasonal_naive_24h": series.reindex(ts - pd.Timedelta(hours=24)).to_numpy(dtype=float),
    }


def build_benchmark(pred_frame: pd.DataFrame, full_actual: pd.DataFrame, targets: list[str], block_hours: int = 72) -> dict:
    out: dict = {"targets": {}, "block_hours": block_hours}
    ts = pd.to_datetime(pred_frame["timestamp"])
    out["window"] = {"start": ts.iloc[0].isoformat(), "end": ts.iloc[-1].isoformat(), "hours": int(len(ts))}

    for target in targets:
        actual = pred_frame[f"actual_{target}"].to_numpy(dtype=float)
        model = pred_frame[f"predicted_{target}"].to_numpy(dtype=float)
        candidates = {"sarima": pred_frame[f"sarima_{target}"].to_numpy(dtype=float)}
        candidates.update(_baselines(full_actual, ts, target))
        valid = np.isfinite(actual) & np.isfinite(model)
        for v in candidates.values():
            valid &= np.isfinite(v)

        a, m = actual[valid], model[valid]
        rows = {"ForeSightX hybrid": {"MAE": _mae(a, m), "RMSE": _rmse(a, m)}}
        comparisons = {}
        for name, base in candidates.items():
            b = base[valid]
            rows[name] = {"MAE": _mae(a, b), "RMSE": _rmse(a, b)}
            blocks = []
            for start in range(0, len(a) - block_hours + 1, block_hours):
                sl = slice(start, start + block_hours)
                blocks.append(_skill(_mae(a[sl], m[sl]), _mae(a[sl], b[sl])))
            blocks = [x for x in blocks if x is not None]
            comparisons[name] = {
                "skill_mae": _skill(rows["ForeSightX hybrid"]["MAE"], rows[name]["MAE"]),
                "skill_rmse": _skill(rows["ForeSightX hybrid"]["RMSE"], rows[name]["RMSE"]),
                "block_skills": [round(x, 4) for x in blocks],
                "block_win_rate": float(np.mean([x > 0 for x in blocks])) if blocks else None,
                "dm_test": diebold_mariano(a - m, a - b),
            }

        # Hours where the observed value actually moved. For signals recorded
        # at daily resolution (AQI, temperature) most hours repeat the last
        # value, so this isolates whether the model adds anything when it matters.
        prev = candidates["persistence"][valid]
        changed = np.abs(a - prev) > 1e-6
        changed_block = None
        if changed.sum() >= 10:
            changed_block = {
                "hours": int(changed.sum()),
                "model_MAE": _mae(a[changed], m[changed]),
                "persistence_MAE": _mae(a[changed], prev[changed]),
                "skill_vs_persistence": _skill(_mae(a[changed], m[changed]), _mae(a[changed], prev[changed])),
            }

        out["targets"][target] = {
            "label": TARGET_LABELS.get(target, target),
            "unit": TARGET_UNITS.get(target, ""),
            "hours_evaluated": int(valid.sum()),
            "share_hours_unchanged": float(1.0 - changed.mean()),
            "models": rows,
            "comparisons": comparisons,
            "changed_hours": changed_block,
        }

    out["daily_resolution"] = _daily_resolution(pred_frame, full_actual, [t for t in ("aqi", "temperature") if t in targets])
    out["verdicts"] = _verdicts(out)
    return out


def _daily_resolution(pred_frame: pd.DataFrame, full_actual: pd.DataFrame, targets: list[str]) -> dict:
    """AQI and temperature come from daily records, so also score them per day.

    Like-for-like: the model's daily mean of one-step forecasts vs the daily
    mean of one-step persistence (both only use information up to t-1).
    Day-ahead reference: yesterday's daily mean (a harder bar for a 1-step model
    to be compared against, reported for context).
    """
    out = {}
    ts = pd.to_datetime(pred_frame["timestamp"])
    full = full_actual.set_index("timestamp")
    for target in targets:
        frame = pd.DataFrame(
            {
                "date": ts.dt.date,
                "actual": pred_frame[f"actual_{target}"].to_numpy(),
                "model": pred_frame[f"predicted_{target}"].to_numpy(),
                "persist": full[target].reindex(ts - pd.Timedelta(hours=1)).to_numpy(),
            }
        )
        counts = frame.groupby("date").size()
        days = counts[counts >= 20].index
        daily = frame[frame["date"].isin(days)].groupby("date").mean()
        yesterday = daily["actual"].shift(1)
        mask = yesterday.notna()
        out[target] = {
            "days": int(len(daily)),
            "model_daily_MAE": _mae(daily["actual"], daily["model"]),
            "persistence_daily_MAE": _mae(daily["actual"], daily["persist"]),
            "skill_vs_persistence": _skill(_mae(daily["actual"], daily["model"]), _mae(daily["actual"], daily["persist"])),
            "yesterday_mean_MAE": _mae(daily["actual"][mask], yesterday[mask]) if mask.any() else None,
        }
    return out


def _verdicts(bench: dict) -> list[str]:
    lines = []
    for target, info in bench["targets"].items():
        label = info["label"]
        parts = []
        for name, short in (("persistence", "last-hour"), ("seasonal_naive_24h", "same-hour-yesterday"), ("sarima", "SARIMA")):
            comp = info["comparisons"][name]
            skill = comp["skill_mae"]
            p = comp["dm_test"]["p_value"]
            if skill is None:
                continue
            sig = " (significant)" if p is not None and p < 0.05 else " (not significant)"
            word = "better" if skill > 0 else "worse"
            parts.append(f"{abs(skill) * 100:.0f}% {word} than {short}{sig}")
        lines.append(f"{label}: " + "; ".join(parts) + ".")
    return lines
