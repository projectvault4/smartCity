"""SHAP explanations computed on the real TFT+GRU hybrid network.

We use shap.GradientExplainer (expected gradients) on the network's six input
tensors: three temporal windows (last N hours, same hours yesterday, same hours
last week; each N x F engineered features), the daily/weekly seasonal-naive
experts and the SARIMA expert.

Attributions are summed into human-level drivers (traffic, AQI, temperature,
humidity, electricity, calendar, seasonal memory, SARIMA) and converted from the
model's scaled target space into original units with the target scaler.

The deployed forecast is not the raw network output: it is calibrated and then
blended with SARIMA (validation-selected weights). We report that mix
explicitly so the explanation never overstates what the neural part decides.
"""

from __future__ import annotations

import numpy as np
import shap
import torch

from utils.hybrid_runtime import GROUP_KEYS

INPUT_KEYS = (*GROUP_KEYS, "seasonal_stat")
CONTEXT_LABELS = {
    "closeness": "Recent hours",
    "period": "Same time yesterday",
    "trend": "Same time last week",
    "seasonal_daily": "Yesterday's value (seasonal memory)",
    "seasonal_weekly": "Last week's value (seasonal memory)",
    "seasonal_stat": "SARIMA statistical expert",
}
DRIVER_LABELS = {
    "traffic_flow": "Traffic",
    "aqi": "Air quality history",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "electricity_demand": "Electricity demand",
    "calendar": "Time of day / weekday",
    "seasonal_memory": "Seasonal memory (yesterday / last week)",
    "sarima": "SARIMA seasonal pattern",
}
CALENDAR = {"hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "season_sin", "season_cos"}


class _Wrapped(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *xs):
        return self.model(dict(zip(INPUT_KEYS, xs)))


def _feature_domain(name: str, domains: list[str]) -> str:
    if name in CALENDAR:
        return "calendar"
    for d in sorted(domains, key=len, reverse=True):
        if name == d or name.startswith(d + "_"):
            return d
    return "calendar"


def final_mix(runtime) -> dict:
    """Approximate linear share of each component in the deployed forecast."""
    mix = {}
    for i, t in enumerate(runtime.targets):
        spec = runtime.calibration_specs[i]
        alpha = float(runtime.sarima_alpha[i])
        lag = float(spec.get("postprocess", {}).get("alpha", 0.0)) if spec.get("postprocess", {}).get("method") == "lag_blend_with_previous_observation" else 0.0
        coefs = spec.get("params", {}).get("coefficients", [0, 1, 0, 0])
        c_net = float(coefs[1]) if len(coefs) > 1 else 1.0
        c_tft = float(coefs[2]) if len(coefs) > 2 else 0.0
        mix[t] = {
            "neural_hybrid": round(alpha * (1 - lag) * (c_net + c_tft), 4),
            "last_observation": round(alpha * lag, 4),
            "sarima": round(1 - alpha, 4),
            "network_to_final_gain": alpha * (1 - lag) * c_net,
        }
    return mix


def compute_shap(runtime, split: str = "test", n_background: int = 96, nsamples: int = 64, seed: int = 0) -> dict:
    model = runtime.model.eval()
    wrapped = _Wrapped(model).eval()
    train = runtime.datasets["train_tpt"]
    groups = runtime.datasets[f"{split}_tpt"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train["target"]), size=min(n_background, len(train["target"])), replace=False)
    background = [torch.tensor(train[k][idx]) for k in INPUT_KEYS]
    explainer = shap.GradientExplainer(wrapped, background, batch_size=64)

    n = len(groups["target"])
    per_input = {k: [] for k in INPUT_KEYS}
    for start in range(0, n, 64):
        xb = [torch.tensor(groups[k][start : start + 64]) for k in INPUT_KEYS]
        values = explainer.shap_values(xb, nsamples=nsamples, rseed=seed)
        for k, v in zip(INPUT_KEYS, values):
            per_input[k].append(np.asarray(v))
    per_input = {k: np.concatenate(v, axis=0) for k, v in per_input.items()}

    with torch.no_grad():
        fx = wrapped(*[torch.tensor(groups[k]) for k in INPUT_KEYS]).numpy()
        base = wrapped(*background).numpy().mean(axis=0)

    feature_names = runtime.datasets["processor"].feature_columns
    domains = list(runtime.config.domain_columns)
    feat_domain = np.array([_feature_domain(f, domains) for f in feature_names])
    scale = _target_scale(runtime)  # scaled -> original units per target

    T = len(runtime.targets)
    driver_keys = list(DRIVER_LABELS)
    drivers = np.zeros((n, T, len(driver_keys)))
    contexts = np.zeros((n, T, len(INPUT_KEYS)))
    for ci, key in enumerate(INPUT_KEYS):
        arr = per_input[key]
        if arr.ndim == 4:  # (n, L, F, T)
            by_feat = arr.sum(axis=1)  # (n, F, T)
            contexts[:, :, ci] = by_feat.sum(axis=1)
            for di, dk in enumerate(driver_keys):
                mask = feat_domain == dk
                if mask.any():
                    drivers[:, :, di] += by_feat[:, mask, :].sum(axis=1)
        else:  # (n, T_in, T_out) expert inputs
            total = arr.sum(axis=1)
            contexts[:, :, ci] = total
            dk = "sarima" if key == "seasonal_stat" else "seasonal_memory"
            drivers[:, :, driver_keys.index(dk)] += total

    drivers *= scale[None, :, None]
    contexts *= scale[None, :, None]
    additivity = np.abs(contexts.sum(axis=2) - (fx - base) * scale[None, :])
    return {
        "drivers": drivers,
        "contexts": contexts,
        "driver_keys": driver_keys,
        "base_value": base * scale + _target_min(runtime),
        "network_output": fx * scale + _target_min(runtime),
        "additivity_mean_abs_error": additivity.mean(axis=0),
        "network_output_std": (fx * scale).std(axis=0),
        "timestamps": groups["timestamp"],
    }


def _target_scale(runtime) -> np.ndarray:
    s = runtime.datasets["processor"].target_scaler
    if hasattr(s, "data_range_"):
        return np.asarray(s.data_range_, dtype=float)
    return np.asarray(s.scale_, dtype=float)


def _target_min(runtime) -> np.ndarray:
    s = runtime.datasets["processor"].target_scaler
    if hasattr(s, "data_min_"):
        return np.asarray(s.data_min_, dtype=float)
    return np.asarray(s.mean_, dtype=float)


TARGET_NOUNS = {"traffic_flow": "traffic", "aqi": "AQI", "temperature": "temperature", "electricity_demand": "electricity demand"}


def driver_label(key: str, target: str | None = None) -> str:
    """Plain-language label; the SARIMA expert is the target's own daily cycle."""
    noun = TARGET_NOUNS.get(target or "", "")
    if key == "sarima" and noun:
        return f"Usual daily {noun} cycle (SARIMA)"
    if key == "seasonal_memory" and noun:
        return f"{noun[0].upper() + noun[1:]} at this hour yesterday / last week"
    if key == target or (key == "aqi" and target == "aqi"):
        return f"Recent {noun} readings"
    return DRIVER_LABELS[key]


def top_drivers(drivers_row: np.ndarray, driver_keys: list[str], gain: float, k: int = 3, target: str | None = None) -> list[dict]:
    """Top-k drivers for one prediction, in final-forecast units."""
    vals = drivers_row * gain
    order = np.argsort(-np.abs(vals))
    total = float(np.abs(vals).sum()) or 1.0
    out = []
    for i in order[:k]:
        out.append(
            {
                "key": driver_keys[i],
                "label": driver_label(driver_keys[i], target),
                "effect": round(float(vals[i]), 3),
                "direction": "up" if vals[i] >= 0 else "down",
                "share": round(float(abs(vals[i]) / total), 3),
            }
        )
    return out


def summarize(runtime, result: dict) -> dict:
    mix = final_mix(runtime)
    drivers = result["drivers"]
    keys = result["driver_keys"]
    ts = result["timestamps"].reset_index(drop=True)
    hours = np.asarray([t.hour for t in ts])
    out = {"method": "shap.GradientExplainer (expected gradients) on the TFT+GRU hybrid network", "targets": {}}
    for ti, target in enumerate(runtime.targets):
        mean_abs = np.abs(drivers[:, ti, :]).mean(axis=0)
        total = float(mean_abs.sum()) or 1.0
        global_rank = sorted(
            [
                {"key": k, "label": driver_label(k, target), "mean_abs_effect": round(float(v), 4), "share": round(float(v / total), 4)}
                for k, v in zip(keys, mean_abs)
            ],
            key=lambda r: -r["share"],
        )
        ctx = np.abs(result["contexts"][:, ti, :]).mean(axis=0)
        ctx_total = float(ctx.sum()) or 1.0
        by_hour = {}
        for h in range(24):
            m = hours == h
            if not m.any():
                continue
            mean_signed = drivers[m, ti, :].mean(axis=0)
            by_hour[str(h)] = top_drivers(mean_signed, keys, mix[target]["network_to_final_gain"], target=target)
        out["targets"][target] = {
            "final_mix": {k: v for k, v in mix[target].items() if k != "network_to_final_gain"},
            "global_drivers": global_rank,
            "time_context": sorted(
                [{"key": k, "label": CONTEXT_LABELS[k], "share": round(float(v / ctx_total), 4)} for k, v in zip(INPUT_KEYS, ctx)],
                key=lambda r: -r["share"],
            ),
            "typical_drivers_by_hour": by_hour,
            "additivity_check": {
                "mean_abs_gap": round(float(result["additivity_mean_abs_error"][ti]), 4),
                "network_output_std": round(float(result["network_output_std"][ti]), 4),
            },
        }
    return out
