"""Build every decision-intelligence artifact from the deployed Bengaluru hybrid.

    venv/bin/python build_intelligence.py            # full build (~10 min, SHAP is the slow part)
    venv/bin/python build_intelligence.py --fast     # fewer SHAP samples, for quick iteration

Writes JSON to outputs/intelligence/ (served by backend /api/intelligence/*):
  manifest.json            provenance, reproduction check, versions
  benchmark.json           skill vs last-hour / same-hour-yesterday / SARIMA, rolling blocks, DM tests
  conformal.json           adaptive conformal interval coverage vs plain split conformal
  shap_summary.json        SHAP drivers on the real hybrid (global, by hour, final-forecast mix)
  whatif.json              full-chain counterfactual scenarios
  anomaly_eval.json        leakage-free anomaly detector vs z-score, injected-incident recall
  replay.json              alert pipeline replay + scorecard vs simple rules
  outlook_uncertainty.json residuals used for live exceedance probabilities on the 2026 outlook
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))

import numpy as np
import pandas as pd
import shap
import sklearn
import torch

from intelligence import load_policy
from intelligence.anomaly import LeakFreeDetector, evaluate as evaluate_anomaly
from intelligence.benchmark import build_benchmark
from intelligence.conformal import adaptive_intervals, static_intervals
from intelligence.explain import compute_shap, final_mix, summarize, top_drivers
from intelligence.replay import run_replay
from intelligence.whatif import run_whatif
from utils.hybrid_runtime import HybridRuntime

OUT = Path("outputs") / "intelligence"


def _clean(obj):
    """Make numpy/pandas values JSON-safe and round floats."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_clean(payload), handle, separators=(",", ":"))
    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)


def _seasonal_profile_residuals(runtime, test) -> dict:
    """Backtest of a long-range outlook: day-of-week x hour medians from all data
    before the test window vs what actually happened in the test window. Used as
    the uncertainty of the live 2026 outlook, which is a long-range forecast."""
    raw = runtime.raw_df.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    test_ts = pd.to_datetime(test.timestamps).reset_index(drop=True)
    hist = raw[raw["timestamp"] < test_ts.iloc[0]]
    prof = hist.groupby([hist["timestamp"].dt.dayofweek, hist["timestamp"].dt.hour])[runtime.targets].median()
    keys = list(zip(test_ts.dt.dayofweek, test_ts.dt.hour))
    expected = prof.reindex(keys).to_numpy()
    raw_resid = test.actual - expected
    # The held-out window is the monsoon, so the raw residuals carry a
    # season-specific bias (e.g. AQI far below the Feb-Jun profile). That bias
    # would be wrong in other seasons, so only the spread is reused live.
    bias = np.nanmean(raw_resid, axis=0)
    resid = raw_resid - bias
    typical = raw.groupby(raw["timestamp"].dt.hour)[runtime.targets].median()
    out = {
        "method": "day-of-week x hour median profile from data before the held-out window, scored on the held-out window",
        "targets": runtime.targets,
        "hours": int(len(resid)),
        "residuals": np.round(resid, 3).tolist(),
        "hour_of_day": test_ts.dt.hour.astype(int).tolist(),
        "typical_by_hour": {t: {str(h): round(float(typical.loc[h, t]), 3) for h in typical.index} for t in runtime.targets},
        "typical_basis": f"median of observed Bengaluru data {raw['timestamp'].min().date()} to {raw['timestamp'].max().date()}",
        "centering": "season-specific mean bias removed; spread kept",
        "per_target": {},
    }
    for i, t in enumerate(runtime.targets):
        r = resid[:, i]
        out["per_target"][t] = {
            "mae": float(np.mean(np.abs(raw_resid[:, i]))),
            "mae_centered": float(np.mean(np.abs(r))),
            "removed_bias": float(bias[i]),
            "q05": float(np.quantile(r, 0.05)),
            "q95": float(np.quantile(r, 0.95)),
            "abs_q90": float(np.quantile(np.abs(r), 0.9)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Smaller SHAP sample for quick iteration.")
    parser.add_argument("--skip-whatif", action="store_true")
    parser.add_argument("--refresh-shap", action="store_true", help="Recompute SHAP even if a cache exists.")
    args = parser.parse_args()

    policy = load_policy()
    alpha = 1.0 - float(policy["interval_coverage"])
    _write("policy.json", policy)  # the backend reads the policy from the artifact folder
    print("Loading deployed hybrid...", flush=True)
    rt = HybridRuntime("bangalore")
    targets = rt.targets
    train = rt.predict_split("train")
    val = rt.predict_split("val")
    test = rt.predict_split("test")
    test_frame = test.frame(targets)
    full_actual = rt.raw_df[["timestamp", *targets]].copy()
    full_actual["timestamp"] = pd.to_datetime(full_actual["timestamp"])
    repro = rt.verify_against_saved()

    _write("manifest.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "city": "bangalore",
        "model": "TFT + GRU residual hybrid (6-expert gate) + validation calibration + SARIMA blend",
        "checkpoint": "outputs/bangalore/checkpoints/tft_gru_residual_hybrid.pt",
        "data_file": str(rt.config.data_file),
        "splits": {
            s: {"start": pd.to_datetime(p.timestamps).iloc[0].isoformat(), "end": pd.to_datetime(p.timestamps).iloc[-1].isoformat(), "hours": int(len(p.final))}
            for s, p in (("train", train), ("validation", val), ("test", test))
        },
        "sarima_blend_weight_on_hybrid": dict(zip(targets, rt.sarima_alpha.tolist())),
        "reproduction_check": {
            **repro,
            "note": "All intelligence artifacts use this re-run of the checkpoint on the current data file. "
                    "The older saved CSV was produced before the data file was last rebuilt, hence small differences.",
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "shap": shap.__version__, "sklearn": sklearn.__version__},
    })

    print("Skill benchmark...", flush=True)
    _write("benchmark.json", build_benchmark(test_frame, full_actual, targets))

    print("Conformal intervals...", flush=True)
    intervals, conf = {}, {"coverage_target": policy["interval_coverage"], "targets": {}}
    for i, t in enumerate(targets):
        seed = val.actual[:, i] - val.final[:, i]
        aci = adaptive_intervals(test.actual[:, i], test.final[:, i], seed, alpha)
        stat = static_intervals(test.actual[:, i], test.final[:, i], seed, alpha)
        intervals[t] = aci
        hrs = pd.to_datetime(test.timestamps).dt.hour.to_numpy()
        conf["targets"][t] = {
            "adaptive": {"coverage": aci["coverage"], "mean_width": aci["mean_width"]},
            "static_split": stat,
            "width_by_hour": {str(h): float(np.mean((aci["upper"] - aci["lower"])[hrs == h])) for h in range(24) if (hrs == h).any()},
            "latest_half_width": float((aci["upper"][-1] - aci["lower"][-1]) / 2),
        }
    _write("conformal.json", conf)

    print("SHAP on the hybrid network (slow)...", flush=True)
    cache = OUT / "cache" / f"shap_{'fast' if args.fast else 'full'}.npz"
    if cache.exists() and not args.refresh_shap:
        z = np.load(cache, allow_pickle=True)
        shap_res = {k: z[k] for k in z.files}
        shap_res["driver_keys"] = list(shap_res["driver_keys"])
        shap_res["timestamps"] = pd.Series(pd.to_datetime(shap_res["timestamps"]))
        print(f"  using cached SHAP values from {cache}", flush=True)
    else:
        shap_res = compute_shap(rt, "test", n_background=48 if args.fast else 96, nsamples=24 if args.fast else 64)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            **{k: v for k, v in shap_res.items() if k not in ("timestamps", "driver_keys")},
            driver_keys=np.array(shap_res["driver_keys"]),
            timestamps=pd.to_datetime(shap_res["timestamps"]).astype(str).to_numpy(),
        )
    shap_sum = summarize(rt, shap_res)
    mix = final_mix(rt)
    shap_rows = {}
    for ti, t in enumerate(targets):
        gain = mix[t]["network_to_final_gain"]
        shap_rows[t] = [top_drivers(shap_res["drivers"][k, ti, :], shap_res["driver_keys"], gain, target=t) for k in range(len(test.final))]
    shap_sum["per_prediction_count"] = len(test.final)
    _write("shap_summary.json", shap_sum)

    print("Leakage-free anomaly detector...", flush=True)
    train_end = pd.to_datetime(rt.datasets["train_df"]["timestamp"]).iloc[-1]
    val_end = pd.to_datetime(rt.datasets["val_df"]["timestamp"]).iloc[-1]
    det = LeakFreeDetector(rt.raw_df, rt.config, train_end, val_end)
    scale = np.array([np.mean(np.abs(val.actual[:, i] - val.final[:, i])) or 1.0 for i in range(len(targets))])
    norm_err = (np.abs(test.actual - test.final) / scale).mean(axis=1)
    resid_df = pd.DataFrame({"timestamp": pd.to_datetime(test.timestamps).reset_index(drop=True), "norm_abs_error": norm_err})
    tts = pd.to_datetime(test.timestamps)
    anom = evaluate_anomaly(det, rt.raw_df, tts.iloc[0], tts.iloc[-1], resid_df)
    anomaly_scores = anom.pop("scores")
    _write("anomaly_eval.json", anom)

    print("Alert replay...", flush=True)
    replay = run_replay(policy, targets, val, test, train, intervals, shap_rows, anomaly_scores, det.threshold)
    _write("replay.json", replay)

    print("Outlook uncertainty for the live 2026 outlook...", flush=True)
    _write("outlook_uncertainty.json", _seasonal_profile_residuals(rt, test))

    if not args.skip_whatif:
        print("What-if counterfactuals (12 full-chain re-runs)...", flush=True)
        _write("whatif.json", run_whatif(rt, test))

    print("Done.")


if __name__ == "__main__":
    main()
