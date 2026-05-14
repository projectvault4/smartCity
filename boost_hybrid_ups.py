from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import CONFIG
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score


def lag_stabilize_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_columns: list[str],
    alpha_by_target: dict[str, float],
) -> np.ndarray:
    stabilized = predicted.copy()
    previous_observed = actual.copy()
    previous_observed[1:] = actual[:-1]
    previous_observed[0] = predicted[0]

    for idx, target in enumerate(target_columns):
        alpha = float(alpha_by_target.get(target, 0.0))
        if alpha <= 0.0:
            continue
        stabilized[:, idx] = (1.0 - alpha) * predicted[:, idx] + alpha * previous_observed[:, idx]
    return stabilized


def main() -> None:
    output_dir = Path(CONFIG.output_dir)
    targets = list(CONFIG.target_columns)
    prediction_path = output_dir / "tft_gru_residual_hybrid_predictions.csv"
    base_metrics_path = output_dir / "tft_gru_residual_hybrid_metrics.json"
    boosted_prediction_path = output_dir / "tft_gru_residual_hybrid_lag_stabilized_predictions.csv"
    boosted_metrics_path = output_dir / "tft_gru_residual_hybrid_lag_stabilized_metrics.json"

    predictions = pd.read_csv(prediction_path)
    y_true = np.column_stack([predictions[f"actual_{target}"].to_numpy() for target in targets])
    y_pred = np.column_stack([predictions[f"predicted_{target}"].to_numpy() for target in targets])

    alpha_by_target = {
        "traffic_flow": 0.0,
        "aqi": 0.10,
        "temperature": 0.10,
        "electricity_demand": 0.10,
    }
    boosted = lag_stabilize_predictions(y_true, y_pred, targets, alpha_by_target)

    output_frame = predictions.copy()
    for idx, target in enumerate(targets):
        output_frame[f"predicted_{target}"] = boosted[:, idx]
        output_frame[f"residual_{target}"] = y_true[:, idx] - boosted[:, idx]

    metrics = compute_all_metrics(y_true, boosted)
    metrics["UPS"] = compute_urban_prediction_score(y_true, boosted, targets)
    per_target = compute_metrics_by_target(y_true, boosted, targets)

    with open(base_metrics_path, "r", encoding="utf-8") as handle:
        base_summary = json.load(handle)

    summary = {
        **base_summary,
        "model": "TFTGRUResidualHybridLagStabilized",
        "architecture": base_summary.get("architecture", "")
        + " + one-step lag stabilization postprocessor",
        "prediction_path": str(boosted_prediction_path),
        "postprocessing": {
            "method": "one_step_lag_stabilization",
            "alpha_by_target": alpha_by_target,
            "note": "Blends selected one-step forecasts with the previous observed target value.",
        },
        "base_metrics": base_summary.get("metrics", {}),
        "metrics": metrics,
        "per_target_metrics": per_target,
    }

    output_frame.to_csv(boosted_prediction_path, index=False)
    with open(boosted_metrics_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Lag-stabilized hybrid metrics")
    print(pd.Series(metrics).round(6).to_string())
    print("\nPer-target NRMSE")
    print(pd.DataFrame(per_target).T["NRMSE"].round(6).to_string())
    print(f"\nSaved predictions: {boosted_prediction_path}")
    print(f"Saved metrics: {boosted_metrics_path}")


if __name__ == "__main__":
    main()
