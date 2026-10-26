"""One alert pipeline, replayed hour by hour over the held-out window.

For each event type in policy.json and each forecast hour t:

  1. probability  p_t = P(event at t | data up to t-1) from conformal residuals
  2. alert hour   if p_t >= cutoff (cutoff picked on the VALIDATION window by F1)
  3. notification sent at the first alert hour after `cooldown_hours` of silence,
     or when the probability escalates by >= 0.25 within a cooldown
  4. explanation  top drivers from SHAP for that exact prediction, the forecast
     interval, and a profile-specific action (same object feeds SMS, in-app,
     voice and AI summary)

Scored against what actually happened, alongside three simple rules a city
could use instead:
  - last-hour rule        : alert for t if the event was happening at t-1
  - same-hour-yesterday   : alert for t if the event happened at t-24
  - point forecast        : alert if the forecast itself crosses the threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from intelligence.conformal import brier_score, exceedance_probabilities, reliability_table


def _episodes(flags: np.ndarray) -> list[tuple[int, int]]:
    eps, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            eps.append((start, i - 1))
            start = None
    if start is not None:
        eps.append((start, len(flags) - 1))
    return eps


def _notifications(alert: np.ndarray, probs: np.ndarray | None, cooldown: int) -> list[int]:
    sent, last, last_p = [], -10**9, 0.0
    for t, a in enumerate(alert):
        if not a:
            continue
        p = float(probs[t]) if probs is not None else 1.0
        if t - last >= cooldown or (probs is not None and p - last_p >= 0.25):
            sent.append(t)
            last, last_p = t, p
    return sent


def score_rule(alert: np.ndarray, truth: np.ndarray, cooldown: int, probs: np.ndarray | None = None, hours: float | None = None) -> dict:
    hours = hours or len(truth)
    tp = int((alert & truth).sum())
    fp = int((alert & ~truth).sum())
    fn = int((~alert & truth).sum())
    sent = _notifications(alert, probs, cooldown)
    correct = [t for t in sent if truth[t : t + cooldown].any()]
    eps = _episodes(truth)
    onset_warned = sum(1 for s, _ in eps if alert[s])
    caught = sum(1 for s, e in eps if alert[s : e + 1].any())
    weeks = hours / 168.0
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if (tp + fn) else None)
    return {
        "hour_precision": precision,
        "hour_recall": recall,
        "hour_f1": f1,
        "notifications_sent": len(sent),
        "notifications_correct": len(correct),
        "notification_precision": (len(correct) / len(sent)) if sent else None,
        "false_alarms_per_week": (len(sent) - len(correct)) / weeks,
        "event_episodes": len(eps),
        "episodes_caught": caught,
        "episodes_warned_before_onset": onset_warned,
    }


def _choose_cutoff(probs: np.ndarray, truth: np.ndarray, default: float) -> tuple[float, str]:
    if truth.sum() < 3:
        return default, "default (too few validation events to tune)"
    best, best_f1 = default, -1.0
    for c in np.arange(0.2, 0.81, 0.05):
        a = probs >= c
        tp = (a & truth).sum(); fp = (a & ~truth).sum(); fn = (~a & truth).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1 + 1e-9:
            best, best_f1 = float(round(c, 2)), f1
    return best, f"max F1 on validation window (F1={best_f1:.2f})"


def _fmt_time(ts: pd.Timestamp) -> str:
    h = ts.hour % 12 or 12
    return f"{h} {'AM' if ts.hour < 12 else 'PM'}"


def _fmt_val(target: str, v: float) -> str:
    if target in ("traffic_flow", "electricity_demand"):
        return f"{v:,.0f}"
    return f"{v:.1f}" if target == "temperature" else f"{v:.0f}"


def build_message(policy: dict, event_id: str, profile: str, ts: pd.Timestamp, prob: float, target: str,
                  values: dict, low: float, high: float, driver_label: str) -> str:
    """values: {target: forecast} for every target in the event's conditions (lead first)."""
    ev = policy["events"][event_id]
    units = policy["units"]
    actions = policy["actions"].get(event_id, {})
    action = actions.get(profile) or actions.get("general") or next(iter(actions.values()), "")
    value_text = " and ".join(f"{_fmt_val(t, v)} {units[t]}" for t, v in values.items())
    return policy["message_template"].format(
        label=ev["label"], time=_fmt_time(ts), probability=int(round(prob * 100)),
        value=value_text, low=_fmt_val(target, low), high=f"{_fmt_val(target, high)} {units[target]}",
        driver=driver_label, action=action,
    ).replace("  ", " ")


def run_replay(policy: dict, targets: list[str], val_pred, test_pred, train_pred, intervals: dict,
               shap_rows: dict, anomaly_scores: pd.DataFrame | None, anomaly_threshold: float | None) -> dict:
    """val_pred / test_pred / train_pred: SplitPredictions; intervals: target -> adaptive_intervals output."""
    cooldown = int(policy["cooldown_hours"])
    tidx = {t: i for i, t in enumerate(targets)}
    seed_val = (train_pred.actual - train_pred.final)
    seed_test = (val_pred.actual - val_pred.final)
    ts = pd.to_datetime(test_pred.timestamps).reset_index(drop=True)
    n = len(ts)
    out = {"cooldown_hours": cooldown, "events": {}, "window": {"start": ts.iloc[0].isoformat(), "end": ts.iloc[-1].isoformat(), "hours": n}}
    hour_rows = [{"timestamp": ts.iloc[i].isoformat(), "alerts": []} for i in range(n)]

    # Hourly series for the player.
    for t in targets:
        i = tidx[t]
        for k in range(n):
            hour_rows[k][t] = {
                "actual": round(float(test_pred.actual[k, i]), 3),
                "forecast": round(float(test_pred.final[k, i]), 3),
                "low": round(float(intervals[t]["lower"][k]), 3),
                "high": round(float(intervals[t]["upper"][k]), 3),
            }
    if anomaly_scores is not None:
        amap = dict(zip(pd.to_datetime(anomaly_scores["timestamp"]), anomaly_scores["score"]))
        for k in range(n):
            s = amap.get(ts.iloc[k] - pd.Timedelta(hours=1))
            hour_rows[k]["anomaly_score"] = None if s is None else round(float(s), 4)
            hour_rows[k]["anomaly_flag"] = bool(s is not None and anomaly_threshold is not None and s >= anomaly_threshold)

    for ev_id, ev in policy["events"].items():
        conds = [(tidx[c["target"]], float(c["threshold"])) for c in ev["conditions"]]
        lead = ev["conditions"][0]["target"]
        li = tidx[lead]

        def truth_of(pred):
            m = np.ones(len(pred.actual), dtype=bool)
            for i, thr in conds:
                m &= pred.actual[:, i] > thr
            return m

        def point_of(pred):
            m = np.ones(len(pred.final), dtype=bool)
            for i, thr in conds:
                m &= pred.final[:, i] > thr
            return m

        p_val = exceedance_probabilities(val_pred.actual, val_pred.final, seed_val, conds)
        truth_val = truth_of(val_pred)
        cutoff, how = _choose_cutoff(p_val, truth_val, float(policy["default_probability_cutoff"]))

        p = exceedance_probabilities(test_pred.actual, test_pred.final, seed_test, conds)
        truth = truth_of(test_pred)
        alert = p >= cutoff
        prev_truth = np.r_[False, truth[:-1]]
        # Same hour yesterday, looked up from the full observed series.
        yest = np.zeros(n, dtype=bool)
        full_truth = dict(zip(ts, truth))
        val_ts = pd.to_datetime(val_pred.timestamps).reset_index(drop=True)
        full_truth.update(dict(zip(val_ts, truth_val)))
        for k in range(n):
            yest[k] = bool(full_truth.get(ts.iloc[k] - pd.Timedelta(hours=24), False))

        out["events"][ev_id] = {
            "label": ev["label"],
            "severity": ev["severity"],
            "conditions": ev["conditions"],
            "cutoff": cutoff,
            "cutoff_rule": how,
            "validation_event_hours": int(truth_val.sum()),
            "test_event_hours": int(truth.sum()),
            "brier_score": brier_score(p, truth),
            "brier_climatology": brier_score(np.full(n, truth_val.mean()), truth),
            "reliability": reliability_table(p, truth),
            "rules": {
                "ForeSightX (probabilistic)": score_rule(alert, truth, cooldown, p),
                "Point forecast crosses threshold": score_rule(point_of(test_pred), truth, cooldown),
                "Last-hour rule": score_rule(prev_truth, truth, cooldown),
                "Same-hour-yesterday rule": score_rule(yest, truth, cooldown),
            },
        }

        sent = _notifications(alert, p, cooldown)
        for k in sent:
            drivers = (shap_rows.get(lead) or [None] * n)[k] or []
            driver_label = drivers[0]["label"] if drivers else "recent pattern"
            if len(drivers) > 1:
                driver_label += f", then {drivers[1]['label'][0].lower()}{drivers[1]['label'][1:]}"
            values = {c["target"]: float(test_pred.final[k, tidx[c["target"]]]) for c in ev["conditions"]}
            msgs = {
                prof: build_message(policy, ev_id, prof, ts.iloc[k], float(p[k]), lead, values,
                                    float(intervals[lead]["lower"][k]), float(intervals[lead]["upper"][k]), driver_label)
                for prof in ev["audience"]
            }
            hour_rows[k]["alerts"].append(
                {
                    "event": ev_id, "label": ev["label"], "severity": ev["severity"],
                    "probability": round(float(p[k]), 3), "audience": ev["audience"],
                    "drivers": drivers, "messages": msgs,
                    "outcome": "correct" if truth[k : k + cooldown].any() else "false_alarm",
                }
            )
        for k in range(n):
            hour_rows[k].setdefault("probabilities", {})[ev_id] = round(float(p[k]), 3)
            hour_rows[k].setdefault("event_happened", {})[ev_id] = bool(truth[k])

    out["scorecard"] = _scorecard(out)
    out["hours"] = hour_rows
    return out


def _scorecard(out: dict) -> dict:
    agg = {}
    for ev in out["events"].values():
        for rule, r in ev["rules"].items():
            a = agg.setdefault(rule, {"notifications_sent": 0, "notifications_correct": 0, "event_episodes": 0,
                                      "episodes_caught": 0, "episodes_warned_before_onset": 0, "false_alarms": 0})
            a["notifications_sent"] += r["notifications_sent"]
            a["notifications_correct"] += r["notifications_correct"]
            a["false_alarms"] += r["notifications_sent"] - r["notifications_correct"]
            a["event_episodes"] += r["event_episodes"]
            a["episodes_caught"] += r["episodes_caught"]
            a["episodes_warned_before_onset"] += r["episodes_warned_before_onset"]
    weeks = out["window"]["hours"] / 168.0
    for a in agg.values():
        a["notification_precision"] = a["notifications_correct"] / a["notifications_sent"] if a["notifications_sent"] else None
        a["false_alarms_per_week"] = a["false_alarms"] / weeks
        a["onset_warning_rate"] = a["episodes_warned_before_onset"] / a["event_episodes"] if a["event_episodes"] else None
    return agg
