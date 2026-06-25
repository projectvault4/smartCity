from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from models.hybrid import TFTGRUResidualHybrid
from utils.config import CONFIG, apply_city_config
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import (
    compute_all_metrics,
    compute_metrics_by_target,
    urban_prediction_score_from_normalized_error,
)
from utils.training import predict_model


FIT_RATIO = 0.60
RIDGE_ALPHAS = (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
FEATURE_SETS = (
    ("model",),
    ("model", "lag_1"),
    ("model", "lag_1", "lag_24"),
    ("model", "lag_1", "lag_24", "lag_48"),
    ("model", "lag_1", "lag_24", "lag_48", "lag_168"),
)


def _config_for_trial(base_config, row) -> object:
    config = copy.deepcopy(base_config)
    config.closeness_lags = tuple(range(1, 1 + int(row.closeness_len)))
    config.period_lags = tuple(range(24, 24 + int(row.period_len)))
    config.trend_lags = tuple(range(168, 168 + int(row.trend_len)))
    for name in (
        "bilstm_hidden_dim",
        "tft_hidden_dim",
        "tft_heads",
        "tft_layers",
        "tft_ff_dim",
        "dense_hidden_dim",
        "batch_size",
    ):
        setattr(config, name, int(getattr(row, name)))
    config.dropout = float(row.dropout)
    config.learning_rate = float(row.learning_rate)
    return config


def _lagged(values: np.ndarray, prediction_fallback: np.ndarray, lag: int) -> np.ndarray:
    result = np.empty_like(values)
    result[:lag] = prediction_fallback[:lag]
    result[lag:] = values[:-lag]
    return result


def _feature_tensor(actual: np.ndarray, prediction: np.ndarray, feature_names: tuple[str, ...]) -> np.ndarray:
    available = {"model": prediction}
    for lag in (1, 24, 48, 168):
        available[f"lag_{lag}"] = _lagged(actual, prediction, lag)
    return np.stack([available[name] for name in feature_names], axis=2)


def _predict_split(model, datasets, split: str, config):
    grouped = datasets[f"{split}_tpt"]
    inputs = {name: grouped[name] for name in ("closeness", "period", "trend")}
    processor = datasets["processor"]
    actual = processor.inverse_transform_targets(grouped["target"])
    prediction = processor.inverse_transform_targets(predict_model(model, inputs, config))
    return actual, prediction


def _fit_target_ridge(features: np.ndarray, target: np.ndarray, fit_slice, score_slice):
    best = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(features[fit_slice], target[fit_slice])
        prediction = model.predict(features[score_slice])
        score = float(np.sqrt(np.mean((target[score_slice] - prediction) ** 2)))
        if best is None or score < best[0]:
            best = (score, alpha, model)
    return best


def _source_manifest(config, prepared_df: pd.DataFrame) -> dict:
    dataset_dir = Path(config.dataset_dir)
    used = [
        dataset_dir / "Banglore_traffic_Dataset.csv",
        dataset_dir / "export.csv",
        *sorted(dataset_dir.glob("*AQI*.xls")),
        *sorted(dataset_dir.glob("*data for Bengaluru*.xls")),
        *sorted((dataset_dir / "BESCOM_2024_LoadCurves").glob("ALLOCATIONVSACTUAL*.xlsx")),
    ]
    unique_used = list(dict.fromkeys(path for path in used if path.exists()))
    files = []
    for path in unique_used:
        files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "city": config.city,
        "dataset_dir": str(dataset_dir),
        "prepared_data_file": str(config.data_file),
        "prepared_rows": len(prepared_df),
        "prepared_start": str(prepared_df["timestamp"].min()),
        "prepared_end": str(prepared_df["timestamp"].max()),
        "loader_source_count": len(files),
        "loader_sources": files,
        "effective_observation_note": (
            "These files are discovered and parsed by the Bangalore loader. Only observations overlapping the "
            "prepared_start/prepared_end window affect the training table; files or rows outside that window do not."
        ),
        "not_used_patterns": [
            "BESCOM_2024_LoadCurves/LOADCURVE-*.xlsx",
            "BESCOM_2024_LoadCurves/KERC.xlsx",
            "BESCOM_2024_LoadCurves/PIO*.xlsx",
            "BESCOM_2024_LoadCurves/Power Interruptions*.xlsx",
        ],
    }


def main() -> None:
    base_config = apply_city_config(copy.deepcopy(CONFIG), "bangalore")
    set_seed(base_config.random_seed)
    output_dir = Path(base_config.output_dir)
    checkpoint_dir = Path(base_config.checkpoint_dir)
    search_path = output_dir / "project_best_hybrid_search.csv"
    trials = pd.read_csv(search_path).sort_values("trial")
    raw_df = load_input_dataframe(base_config)

    search_rows = []
    cached = {}
    for row in trials.itertuples(index=False):
        config = _config_for_trial(base_config, row)
        datasets = create_datasets(config, raw_df)
        checkpoint = checkpoint_dir / f"project_best_hybrid_trial_{int(row.trial)}.pt"
        model = TFTGRUResidualHybrid(datasets["train_tpt"]["closeness"].shape[-1], config)
        model.load_state_dict(torch.load(checkpoint, map_location=config.device))
        y_val, val_prediction = _predict_split(model, datasets, "val", config)
        y_test, test_prediction = _predict_split(model, datasets, "test", config)
        selection_start = int(len(y_val) * FIT_RATIO)
        cached[int(row.trial)] = (config, datasets, y_val, val_prediction, y_test, test_prediction, checkpoint)

        for feature_names in FEATURE_SETS:
            features = _feature_tensor(y_val, val_prediction, feature_names)
            selected_prediction = np.zeros_like(y_val[selection_start:])
            alphas = []
            for target_idx in range(y_val.shape[1]):
                _, alpha, ridge = _fit_target_ridge(
                    features[:, target_idx, :],
                    y_val[:, target_idx],
                    slice(0, selection_start),
                    slice(selection_start, None),
                )
                selected_prediction[:, target_idx] = ridge.predict(features[selection_start:, target_idx, :])
                alphas.append(alpha)
            metrics = compute_all_metrics(y_val[selection_start:], selected_prediction)
            search_rows.append(
                {
                    "trial": int(row.trial),
                    "features": ",".join(feature_names),
                    "selection_RMSE": metrics["RMSE"],
                    "selection_MAE": metrics["MAE"],
                    "ridge_alphas": json.dumps(alphas),
                }
            )

    search = pd.DataFrame(search_rows).sort_values("selection_RMSE").reset_index(drop=True)
    winner = search.iloc[0]
    trial = int(winner.trial)
    feature_names = tuple(str(winner.features).split(","))
    config, datasets, y_val, val_prediction, y_test, raw_test_prediction, source_checkpoint = cached[trial]
    val_features = _feature_tensor(y_val, val_prediction, feature_names)
    test_features = _feature_tensor(y_test, raw_test_prediction, feature_names)
    calibrated_test_prediction = np.zeros_like(raw_test_prediction)
    calibration = {}
    for target_idx, target_name in enumerate(config.target_columns):
        score, alpha, ridge = _fit_target_ridge(
            val_features[:, target_idx, :],
            y_val[:, target_idx],
            slice(None),
            slice(None),
        )
        calibrated_test_prediction[:, target_idx] = ridge.predict(test_features[:, target_idx, :])
        calibration[target_name] = {
            "features": list(feature_names),
            "ridge_alpha": alpha,
            "intercept": float(ridge.intercept_),
            "coefficients": ridge.coef_.tolist(),
            "validation_RMSE": score,
        }

    raw_metrics = compute_all_metrics(y_test, raw_test_prediction)
    metrics = compute_all_metrics(y_test, calibrated_test_prediction)
    metrics["UPS"] = urban_prediction_score_from_normalized_error(metrics["NRMSE"])
    per_target = compute_metrics_by_target(y_test, calibrated_test_prediction, config.target_columns)

    tuned_checkpoint = checkpoint_dir / "bangalore_hybrid_tuned_best.pt"
    shutil.copy2(source_checkpoint, tuned_checkpoint)
    prediction_frame = pd.DataFrame({"timestamp": datasets["test_tpt"]["timestamp"].astype(str)})
    for target_idx, target_name in enumerate(config.target_columns):
        prediction_frame[f"actual_{target_name}"] = y_test[:, target_idx]
        prediction_frame[f"predicted_{target_name}"] = calibrated_test_prediction[:, target_idx]
        prediction_frame[f"raw_predicted_{target_name}"] = raw_test_prediction[:, target_idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(output_dir / "bangalore_hybrid_calibration_search.csv", index=False)
    prediction_frame.to_csv(output_dir / "bangalore_hybrid_tuned_predictions.csv", index=False)
    manifest = _source_manifest(config, raw_df)
    (output_dir / "bangalore_training_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    artifact = {
        "model": "BangaloreTunedTFTGRUResidualHybrid",
        "selection_protocol": {
            "checkpoint_candidates": len(trials),
            "calibration_feature_candidates": [list(names) for names in FEATURE_SETS],
            "validation_fit_ratio": FIT_RATIO,
            "selection_rule": "lowest chronological validation holdout RMSE",
            "test_used_for_selection": False,
        },
        "selected_trial": trial,
        "selected_features": list(feature_names),
        "source_checkpoint": str(source_checkpoint),
        "checkpoint_path": str(tuned_checkpoint),
        "prediction_path": str(output_dir / "bangalore_hybrid_tuned_predictions.csv"),
        "data_manifest_path": str(output_dir / "bangalore_training_data_manifest.json"),
        "raw_test_metrics": raw_metrics,
        "metrics": metrics,
        "per_target_metrics": per_target,
        "calibration": calibration,
    }
    (output_dir / "bangalore_hybrid_tuned_best.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
