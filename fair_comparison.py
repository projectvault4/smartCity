from __future__ import annotations

"""Evaluate every model on one identical test window for a fair comparison.

All deep baselines, the classical SARIMA/ARIMA models, and the TFT+GRU+SARIMA
hybrid are scored on exactly the same set of test timestamps (the intersection
of the hybrid's prediction window and the standard test split), using the same
ground-truth targets. Results are printed and saved.
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from evaluate import evaluate_models  # noqa: E402
from train import build_models, checkpoint_name_for_model  # noqa: E402
from utils.baselines import _predict_statistical_baselines, _merge_temporal_inputs  # noqa: E402
from models.recurrent_baselines import PlainGRU, PlainLSTM  # noqa: E402
from utils.config import CONFIG, apply_city_config  # noqa: E402
from utils.data_utils import create_datasets, load_input_dataframe, set_seed  # noqa: E402
from utils.metrics import compute_all_metrics, compute_urban_prediction_score  # noqa: E402
from utils.training import predict_model  # noqa: E402

TARGETS_KEYS = ("closeness", "period", "trend")


def _metrics_row(y_true, y_pred, targets):
    m = compute_all_metrics(y_true, y_pred)
    m["UPS"] = compute_urban_prediction_score(y_true, y_pred, targets)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair single-window model comparison.")
    parser.add_argument("--city", default="bangalore")
    args = parser.parse_args()

    config = apply_city_config(copy.deepcopy(CONFIG), args.city)
    set_seed(config.random_seed)
    targets = list(config.target_columns)

    raw = load_input_dataframe(config)
    ds = create_datasets(config, raw)
    processor = ds["processor"]
    test_ts = pd.to_datetime(ds["test_tpt"]["timestamp"]).reset_index(drop=True)
    y_true_full = processor.inverse_transform_targets(ds["test_tpt"]["target"])

    # Hybrid predictions define the reference window.
    hyb = pd.read_csv(Path(config.output_dir) / "tft_gru_residual_hybrid_predictions.csv")
    hyb_ts = pd.to_datetime(hyb["timestamp"])
    common = sorted(set(test_ts).intersection(set(hyb_ts)))
    common_idx = pd.Index(common)
    print(f"Fair comparison window: {len(common)} timestamps ({common[0]} -> {common[-1]})")

    # Ground truth on the common window, ordered by common timestamps.
    yt_df = pd.DataFrame(y_true_full, columns=targets, index=test_ts).reindex(common_idx)
    y_true = yt_df.to_numpy(dtype=float)

    def align(pred_array, index):
        return pd.DataFrame(pred_array, columns=targets, index=pd.to_datetime(index)).reindex(common_idx).to_numpy(float)

    predictions = {}

    # Deep models from checkpoints (skip any that do not load).
    input_dim = ds["train_tpt"]["closeness"].shape[-1]
    models = build_models(input_dim, config)
    x_test = {k: ds["test_tpt"][k] for k in TARGETS_KEYS}
    for name in ("BiLSTM", "TFT", "Informer", "PatchTST"):
        ckpt = Path(config.checkpoint_dir) / f"{checkpoint_name_for_model(name)}.pt"
        if not ckpt.exists():
            continue
        try:
            models[name].load_state_dict(torch.load(ckpt, map_location="cpu"))
            scaled = predict_model(models[name], x_test, config)
            predictions[name] = align(processor.inverse_transform_targets(scaled), test_ts)
        except Exception as exc:
            print(f"  skip {name}: {exc}")

    # LSTM / GRU recurrent baselines.
    x_test_seq = _merge_temporal_inputs(ds["test_tpt"])
    rec = {
        "LSTM": PlainLSTM(input_dim, config.bilstm_hidden_dim, config.bilstm_layers, config.dropout, len(targets)),
        "GRU": PlainGRU(input_dim, config.bilstm_hidden_dim, config.bilstm_layers, config.dropout, len(targets)),
    }
    for name, model in rec.items():
        ckpt = Path(config.checkpoint_dir) / f"{name.lower()}.pt"
        if not ckpt.exists():
            continue
        try:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            scaled = predict_model(model, x_test_seq, config)
            predictions[name] = align(processor.inverse_transform_targets(scaled), test_ts)
        except Exception as exc:
            print(f"  skip {name}: {exc}")

    # Classical statistical baselines (aligned to test_tpt timestamps already).
    stat_preds, _ = _predict_statistical_baselines(ds, config)
    for name, arr in stat_preds.items():
        predictions[name] = align(arr, test_ts)

    # The proposed hybrid (already on its own window).
    hyb_pred = hyb[[f"predicted_{t}" for t in targets]].to_numpy(dtype=float)
    predictions["Hybrid"] = align(hyb_pred, hyb_ts)

    rows = {name: _metrics_row(y_true, pred, targets) for name, pred in predictions.items()}
    table = pd.DataFrame(rows).T[["MAE", "MAPE", "RMSE", "NRMSE", "UPS"]].sort_values("UPS", ascending=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\nFAIR COMPARISON (identical test window)\n")
    print(table.to_string())

    out_csv = Path(config.output_dir) / "fair_comparison.csv"
    table.to_csv(out_csv)
    print(f"\nSaved: {out_csv}")

    # Sync these fair numbers into summary.json so downstream tables match.
    summary_path = Path(config.output_dir) / "summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path))
        summary.setdefault("offline_metrics", {})
        for name, m in rows.items():
            summary["offline_metrics"][name] = {k: float(v) for k, v in m.items()}
        json.dump(summary, open(summary_path, "w"), indent=2)
        print(f"Updated {summary_path} offline_metrics for: {', '.join(rows.keys())}")


if __name__ == "__main__":
    main()
