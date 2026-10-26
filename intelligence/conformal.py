"""Adaptive conformal prediction intervals and exceedance probabilities.

For every forecast hour we only use residuals observed *before* that hour:

  interval_t = yhat_t +/- q_t,   q_t = quantile_{1-alpha_t}(|r| over the trailing window)

alpha_t is updated online (Adaptive Conformal Inference, Gibbs & Candes 2021):

  alpha_{t+1} = alpha_t + gamma * (alpha - miss_t)

so coverage self-corrects when the error regime shifts. The same trailing set of
signed residuals gives a conformal predictive distribution, from which

  P(Y_t > threshold) = mean( yhat_t + r_i > threshold )

For multi-condition events the residual *vectors* are used jointly, so the
correlation between targets is kept.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _quantile_abs(window: np.ndarray, level: float) -> float:
    if len(window) == 0:
        return float("nan")
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(np.abs(window), level, method="higher"))


def adaptive_intervals(
    actual: np.ndarray,
    predicted: np.ndarray,
    seed_residuals: np.ndarray,
    alpha: float,
    window: int = 168,
    gamma: float = 0.01,
) -> dict:
    """Online ACI for a single target. Returns lower/upper bounds and coverage."""
    hist = deque(seed_residuals[-window:].tolist(), maxlen=window)
    a_t = alpha
    n = len(actual)
    lower = np.empty(n)
    upper = np.empty(n)
    covered = np.zeros(n, dtype=bool)
    for t in range(n):
        q = _quantile_abs(np.asarray(hist), 1.0 - a_t) if a_t > 0 else float(np.max(np.abs(hist)))
        lower[t] = predicted[t] - q
        upper[t] = predicted[t] + q
        covered[t] = lower[t] <= actual[t] <= upper[t]
        a_t = a_t + gamma * (alpha - (0.0 if covered[t] else 1.0))
        hist.append(float(actual[t] - predicted[t]))
    return {
        "lower": lower,
        "upper": upper,
        "covered": covered,
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(upper - lower)),
    }


def static_intervals(actual: np.ndarray, predicted: np.ndarray, calib_residuals: np.ndarray, alpha: float) -> dict:
    """Plain split conformal (fixed width from the calibration set) for comparison."""
    q = _quantile_abs(calib_residuals, 1.0 - alpha)
    covered = (predicted - q <= actual) & (actual <= predicted + q)
    return {"coverage": float(covered.mean()), "mean_width": float(2 * q)}


def exceedance_probabilities(
    actual: np.ndarray,
    predicted: np.ndarray,
    seed_residuals: np.ndarray,
    conditions: list[tuple[int, float]],
    window: int = 168,
) -> np.ndarray:
    """P(all conditions exceeded) for each hour using trailing residual vectors.

    actual/predicted: (N, T); seed_residuals: (M, T); conditions: [(target_idx, threshold)].
    """
    hist = deque([row for row in seed_residuals[-window:]], maxlen=window)
    n = len(actual)
    probs = np.empty(n)
    for t in range(n):
        res = np.asarray(hist)
        hit = np.ones(len(res), dtype=bool)
        for idx, thr in conditions:
            hit &= (predicted[t, idx] + res[:, idx]) > thr
        probs[t] = float(hit.mean()) if len(res) else 0.0
        hist.append(actual[t] - predicted[t])
    return probs


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes.astype(float)) ** 2))


def reliability_table(probs: np.ndarray, outcomes: np.ndarray, bins=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)) -> list[dict]:
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "hours": int(mask.sum()),
                "mean_predicted": float(probs[mask].mean()),
                "observed_rate": float(outcomes[mask].mean()),
            }
        )
    return rows
