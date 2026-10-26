"""Reproduce the deployed Bengaluru TFT+GRU residual hybrid end to end.

The canonical predictions in ``outputs/bangalore/tft_gru_residual_hybrid_predictions.csv``
are produced by ``train_tft_gru_residual_hybrid.py`` as:

    network (TFT + GRU + 6-expert gate + residual head)
      -> validation-fitted OLS calibration (+ optional previous-observation blend)
      -> validation-selected convex blend with the SARIMA one-step expert

This module rebuilds exactly that chain from the saved checkpoint and the saved
calibration/blend parameters so downstream analysis (SHAP, conformal intervals,
counterfactual what-if, alert replay) runs on the *same* model that produced the
headline numbers, not on a stand-in.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.hybrid import TFTGRUResidualHybrid
from utils.config import CONFIG, apply_city_config
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.statistical_expert import compute_statistical_expert
from utils.training import predict_model

GROUP_KEYS = ("closeness", "period", "trend", "seasonal_daily", "seasonal_weekly")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_config(city: str = "bangalore"):
    """Config identical to the one used when the canonical hybrid was trained."""
    config = apply_city_config(copy.deepcopy(CONFIG), city)
    best_tft = _load_json(Path(config.output_dir) / "project_best_tft_best.json")
    seq_len = int(best_tft.get("seq_len", config.seq_len))
    config.seq_len = seq_len
    config.closeness_lags = tuple(range(1, 1 + seq_len))
    config.period_lags = tuple(range(24, 24 + seq_len))
    config.trend_lags = tuple(range(24 * 7, 24 * 7 + seq_len))
    config.tft_hidden_dim = int(best_tft.get("hidden_dim", config.tft_hidden_dim))
    config.tft_heads = int(best_tft.get("heads", config.tft_heads))
    config.tft_layers = int(best_tft.get("layers", config.tft_layers))
    config.tft_ff_dim = int(best_tft.get("ff_dim", config.tft_ff_dim))
    config.dropout = float(best_tft.get("dropout", config.dropout))
    config.batch_size = int(best_tft.get("batch_size", config.batch_size))
    # CPU keeps inference deterministic and lets SHAP take gradients reliably.
    config.device = "cpu"
    return config


def _linear_design_hybrid(final_pred: np.ndarray, tft_pred: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(final_pred)), final_pred, tft_pred, final_pred * tft_pred])


def _previous_observed(actual: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    previous = actual.copy()
    previous[1:] = actual[:-1]
    previous[0] = fallback[0]
    return previous


@torch.no_grad()
def _tft_branch_predict(model, x: dict, batch_size: int = 256) -> np.ndarray:
    model.eval()
    n = len(next(iter(x.values())))
    out = []
    for start in range(0, n, batch_size):
        xb = {k: torch.as_tensor(v[start : start + batch_size]) for k, v in x.items()}
        context = model.tft_branch.encode(xb)
        out.append(model.tft_branch.regressor(context).numpy())
    return np.concatenate(out, axis=0)


@dataclass
class SplitPredictions:
    timestamps: pd.Series
    actual: np.ndarray  # original units, shape (N, T)
    network: np.ndarray  # raw network output, original units
    tft: np.ndarray
    sarima: np.ndarray
    final: np.ndarray  # deployed prediction (calibrated + SARIMA blend)

    def frame(self, targets: list[str]) -> pd.DataFrame:
        df = pd.DataFrame({"timestamp": pd.to_datetime(self.timestamps).reset_index(drop=True)})
        for i, t in enumerate(targets):
            df[f"actual_{t}"] = self.actual[:, i]
            df[f"predicted_{t}"] = self.final[:, i]
            df[f"network_{t}"] = self.network[:, i]
            df[f"sarima_{t}"] = self.sarima[:, i]
        return df


class HybridRuntime:
    """Loads data + checkpoint once and exposes the deployed prediction chain."""

    def __init__(self, city: str = "bangalore"):
        self.config = build_config(city)
        set_seed(self.config.random_seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.targets = list(self.config.target_columns)
        self.output_dir = Path(self.config.output_dir)
        self.summary = _load_json(self.output_dir / "tft_gru_residual_hybrid_metrics.json")
        self.calibration_specs = [
            self.summary["calibration"]["target_specs"][t] for t in self.targets
        ]
        blend = (self.summary.get("sarima_blend") or {}).get("alpha_by_target", {})
        self.sarima_alpha = np.array([float(blend.get(t, 1.0)) for t in self.targets])
        self.raw_df = load_input_dataframe(self.config)
        self.datasets = None
        self.model = None
        self._prepare(self.raw_df)
        self._load_model()

    # ------------------------------------------------------------------ setup
    def _prepare(self, raw_df: pd.DataFrame):
        datasets = create_datasets(self.config, raw_df)
        stat_raw = compute_statistical_expert(
            datasets["prepared_df"], self.targets, train_end=len(datasets["train_df"]), model="sarima"
        )
        processor = datasets["processor"]
        stat_scaled = pd.DataFrame(
            processor.target_scaler.transform(stat_raw[self.targets].to_numpy(dtype=float)),
            index=stat_raw.index,
            columns=self.targets,
        )
        for split in ("train", "val", "test"):
            groups = datasets[f"{split}_tpt"]
            ts = pd.to_datetime(groups["timestamp"]).reset_index(drop=True)
            groups["seasonal_stat"] = np.nan_to_num(stat_scaled.reindex(ts).to_numpy(dtype=np.float32))
            groups["sarima_orig"] = np.nan_to_num(stat_raw.reindex(ts)[self.targets].to_numpy(dtype=float))
        self.datasets = datasets
        return datasets

    def _load_model(self):
        input_dim = self.datasets["train_tpt"]["closeness"].shape[-1]
        model = TFTGRUResidualHybrid(input_dim=input_dim, config=self.config)
        ckpt = self.output_dir / "checkpoints" / "tft_gru_residual_hybrid.pt"
        state = torch.load(ckpt, map_location="cpu")
        # Trained with --original-scale-loss: the loss-weight buffer is part of
        # the state dict but does not affect inference.
        if "target_loss_weights" in state:
            model.register_buffer("target_loss_weights", state["target_loss_weights"])
        model.load_state_dict(state)
        model.eval()
        self.model = model

    # -------------------------------------------------------------- helpers
    @staticmethod
    def model_inputs(groups: dict) -> dict:
        return {k: groups[k] for k in (*GROUP_KEYS, "seasonal_stat")}

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        return self.datasets["processor"].inverse_transform_targets(scaled)

    def _calibrate(self, network: np.ndarray, tft: np.ndarray, previous: np.ndarray) -> np.ndarray:
        out = np.zeros_like(network)
        for i, spec in enumerate(self.calibration_specs):
            method = spec["method"]
            params = spec.get("params", {})
            if method == "raw_hybrid":
                out[:, i] = network[:, i]
            elif method == "tft":
                out[:, i] = tft[:, i]
            elif method == "blend":
                out[:, i] = params["alpha_hybrid"] * network[:, i] + params["alpha_tft"] * tft[:, i]
            elif method in {"ols", "ridge"}:
                out[:, i] = _linear_design_hybrid(network[:, i], tft[:, i]) @ np.asarray(params["coefficients"])
            else:
                raise ValueError(f"Unknown calibration method {method}")
            post = spec.get("postprocess", {})
            if post.get("method") == "lag_blend_with_previous_observation":
                a = float(post.get("alpha", 0.0))
                out[:, i] = (1 - a) * out[:, i] + a * previous[:, i]
        return out

    def predict_split(self, split: str = "test", datasets: dict | None = None) -> SplitPredictions:
        datasets = datasets or self.datasets
        groups = datasets[f"{split}_tpt"]
        x = self.model_inputs(groups)
        with torch.no_grad():
            network = self.inverse(predict_model(self.model, x, self.config))
        tft = self.inverse(_tft_branch_predict(self.model, x))
        actual = self.inverse(groups["target"])
        previous = np.column_stack(
            [_previous_observed(actual[:, i], network[:, i]) for i in range(actual.shape[1])]
        )
        calibrated = self._calibrate(network, tft, previous)
        sarima = groups["sarima_orig"]
        final = self.sarima_alpha * calibrated + (1 - self.sarima_alpha) * sarima
        return SplitPredictions(groups["timestamp"], actual, network, tft, sarima, final)

    def predict_with_raw(self, raw_df: pd.DataFrame, split: str = "test") -> SplitPredictions:
        """Re-run the full chain on a modified raw frame (used for counterfactuals).

        Scalers are fitted on the training split only, so perturbing rows that
        fall in the test period leaves every fitted transform unchanged.
        """
        original = self.datasets
        try:
            perturbed = self._prepare(raw_df)
            return self.predict_split(split, perturbed)
        finally:
            self.datasets = original

    def verify_against_saved(self) -> dict:
        saved = pd.read_csv(self.output_dir / "tft_gru_residual_hybrid_predictions.csv")
        pred = self.predict_split("test")
        rep = pred.frame(self.targets)
        n = min(len(saved), len(rep))
        diffs = {
            t: float(np.max(np.abs(saved[f"predicted_{t}"].to_numpy()[-n:] - rep[f"predicted_{t}"].to_numpy()[-n:])))
            for t in self.targets
        }
        return {"rows_saved": len(saved), "rows_reproduced": len(rep), "max_abs_diff": diffs}
