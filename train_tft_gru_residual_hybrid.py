from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.hybrid import TFTGRUResidualHybrid
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score
from utils.training import create_loader, predict_model, train_model


def _window(length: int, anchor: int) -> tuple[int, ...]:
    return tuple(range(anchor, anchor + length))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_project_best_tft_settings(config, args):
    best_tft = _load_json(Path(config.output_dir) / "project_best_tft_best.json")
    if not best_tft:
        return "project-best TFT config not found"

    seq_len = int(best_tft.get("seq_len", config.seq_len))
    config.seq_len = seq_len
    config.closeness_lags = _window(seq_len, 1)
    config.period_lags = _window(seq_len, 24)
    config.trend_lags = _window(seq_len, 24 * 7)
    config.tft_hidden_dim = int(best_tft.get("hidden_dim", config.tft_hidden_dim))
    config.tft_heads = int(best_tft.get("heads", config.tft_heads))
    config.tft_layers = int(best_tft.get("layers", config.tft_layers))
    config.tft_ff_dim = int(best_tft.get("ff_dim", config.tft_ff_dim))
    config.dropout = float(best_tft.get("dropout", config.dropout))
    config.learning_rate = float(best_tft.get("learning_rate", config.learning_rate))
    if args.batch_size is None:
        config.batch_size = int(best_tft.get("batch_size", config.batch_size))
    if args.epochs is None:
        config.epochs = 20
    config.patience = max(config.patience, 5)
    config.lr_scheduler_patience = max(config.lr_scheduler_patience, 3)
    return "using project-best TFT config"


def _target_scale_loss_weights(processor, power: float = 2.0) -> torch.Tensor:
    scaler = processor.target_scaler
    if hasattr(scaler, "data_range_"):
        scales = np.asarray(scaler.data_range_, dtype=np.float32)
    elif hasattr(scaler, "scale_"):
        scales = 1.0 / np.asarray(scaler.scale_, dtype=np.float32)
    else:
        scales = np.ones(len(processor.target_columns), dtype=np.float32)
    weights = np.maximum(scales, 1e-6) ** power
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _load_tft_branch(model, config, freeze: bool):
    best_tft = _load_json(Path(config.output_dir) / "project_best_tft_best.json")
    checkpoint = Path(config.checkpoint_dir) / "project_best_tft.pt"
    if not checkpoint.exists() and best_tft.get("checkpoint_path"):
        checkpoint = Path(best_tft["checkpoint_path"])
    if not checkpoint.exists():
        print("Project-best TFT checkpoint not found; training hybrid from scratch.", flush=True)
        return None

    model.tft_branch.load_state_dict(torch.load(checkpoint, map_location=config.device))
    if freeze:
        for param in model.tft_branch.parameters():
            param.requires_grad = False
    print(f"Loaded TFT branch from {checkpoint} | freeze_tft={freeze}", flush=True)
    return str(checkpoint)


@torch.no_grad()
def _predict_tft_branch(model, x, config):
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()
    first = next(iter(x.values())) if isinstance(x, dict) else x
    loader = create_loader(
        x,
        np.zeros((len(first), len(config.target_columns)), dtype=np.float32),
        batch_size=config.batch_size,
        shuffle=False,
    )
    preds = []
    for xb, _ in loader:
        if isinstance(xb, dict):
            xb = {key: value.to(device) for key, value in xb.items()}
        else:
            xb = xb.to(device)
        context = model.tft_branch.encode(xb)
        preds.append(model.tft_branch.regressor(context).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _target_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _linear_design(final_pred: np.ndarray, tft_pred: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(final_pred)), final_pred, tft_pred, final_pred - tft_pred])


def _fit_ridge(design: np.ndarray, target: np.ndarray, ridge_lambda: float) -> np.ndarray:
    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _fit_validation_calibration(y_val, final_val, tft_val):
    specs = []
    calibrated_val = np.zeros_like(final_val)
    alpha_grid = np.linspace(0.0, 1.0, 51)
    ridge_grid = (0.01, 0.1, 1.0, 10.0, 100.0)

    for target_idx in range(y_val.shape[1]):
        y_target = y_val[:, target_idx]
        final_target = final_val[:, target_idx]
        tft_target = tft_val[:, target_idx]
        design = _linear_design(final_target, tft_target)

        candidates = [
            {
                "method": "raw_hybrid",
                "prediction": final_target,
                "params": {},
            },
            {
                "method": "tft",
                "prediction": tft_target,
                "params": {},
            },
        ]
        for alpha in alpha_grid:
            prediction = alpha * final_target + (1.0 - alpha) * tft_target
            candidates.append(
                {
                    "method": "blend",
                    "prediction": prediction,
                    "params": {"alpha_hybrid": float(alpha), "alpha_tft": float(1.0 - alpha)},
                }
            )

        ols_coef, *_ = np.linalg.lstsq(design, y_target, rcond=None)
        candidates.append(
            {
                "method": "ols",
                "prediction": design @ ols_coef,
                "params": {"coefficients": ols_coef.tolist()},
            }
        )
        for ridge_lambda in ridge_grid:
            ridge_coef = _fit_ridge(design, y_target, ridge_lambda)
            candidates.append(
                {
                    "method": "ridge",
                    "prediction": design @ ridge_coef,
                    "params": {"lambda": ridge_lambda, "coefficients": ridge_coef.tolist()},
                }
            )

        best = min(candidates, key=lambda item: _target_rmse(y_target, item["prediction"]))
        calibrated_val[:, target_idx] = best["prediction"]
        specs.append(
            {
                "target_index": target_idx,
                "method": best["method"],
                "params": best["params"],
                "val_RMSE": _target_rmse(y_target, best["prediction"]),
                "raw_val_RMSE": _target_rmse(y_target, final_target),
                "tft_val_RMSE": _target_rmse(y_target, tft_target),
            }
        )
    return specs, calibrated_val


def _apply_validation_calibration(final_pred, tft_pred, specs):
    calibrated = np.zeros_like(final_pred)
    for target_idx, spec in enumerate(specs):
        final_target = final_pred[:, target_idx]
        tft_target = tft_pred[:, target_idx]
        method = spec["method"]
        params = spec.get("params", {})
        if method == "raw_hybrid":
            calibrated[:, target_idx] = final_target
        elif method == "tft":
            calibrated[:, target_idx] = tft_target
        elif method == "blend":
            calibrated[:, target_idx] = params["alpha_hybrid"] * final_target + params["alpha_tft"] * tft_target
        elif method in {"ols", "ridge"}:
            design = _linear_design(final_target, tft_target)
            calibrated[:, target_idx] = design @ np.asarray(params["coefficients"])
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    return calibrated


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the TFT + GRU residual-correction hybrid.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout.")
    parser.add_argument("--gru-hidden-dim", type=int, default=None, help="Override GRU hidden dimension.")
    parser.add_argument("--dense-hidden-dim", type=int, default=None, help="Override fusion/residual hidden dimension.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    parser.add_argument(
        "--original-scale-loss",
        action="store_true",
        help="Weight target losses by original target scale to optimize aggregate original-scale RMSE.",
    )
    parser.add_argument("--loss-weight-power", type=float, default=2.0, help="Power for --original-scale-loss weights.")
    parser.add_argument("--checkpoint-name", default="tft_gru_residual_hybrid", help="Checkpoint stem under outputs/checkpoints.")
    parser.add_argument(
        "--no-project-best-tft-init",
        action="store_true",
        help="Do not initialize/freeze the TFT branch from the tuned project-best TFT checkpoint.",
    )
    parser.add_argument("--unfreeze-tft", action="store_true", help="Fine-tune the initialized TFT branch too.")
    parser.add_argument("--no-calibration", action="store_true", help="Disable validation-learned output calibration.")
    parser.add_argument(
        "--no-ups-routing",
        action="store_true",
        help="Disable TFT fallback routing for targets that receive near-zero original-scale loss weight.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Load the checkpoint and retune/evaluate outputs without training.")
    parser.add_argument(
        "--keep-best",
        action="store_true",
        help="Do not overwrite the canonical metrics/predictions unless the new overall RMSE is better.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = copy.deepcopy(CONFIG)
    tft_setting_note = None
    if not args.no_project_best_tft_init:
        tft_setting_note = _apply_project_best_tft_settings(config, args)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.gru_hidden_dim is not None:
        config.bilstm_hidden_dim = args.gru_hidden_dim
    if args.dense_hidden_dim is not None:
        config.dense_hidden_dim = args.dense_hidden_dim
    if args.patience is not None:
        config.patience = args.patience

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.random_seed)
    if tft_setting_note:
        print(tft_setting_note, flush=True)

    print("Loading dataset...", flush=True)
    raw_df = load_input_dataframe(config)
    print(f"Loaded {len(raw_df)} raw rows", flush=True)
    print("Preparing temporal train/validation/test groups...", flush=True)
    datasets = create_datasets(config, raw_df)
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    print(
        "Prepared groups | "
        f"train={len(datasets['train_tpt']['target'])}, "
        f"val={len(datasets['val_tpt']['target'])}, "
        f"test={len(datasets['test_tpt']['target'])}, "
        f"input_dim={input_dim}",
        flush=True,
    )
    print("Building TFT + GRU residual hybrid...", flush=True)
    model = TFTGRUResidualHybrid(input_dim=input_dim, config=config)
    if args.original_scale_loss:
        model.register_buffer("target_loss_weights", _target_scale_loss_weights(datasets["processor"], args.loss_weight_power))
        print(
            "Using original-scale target loss weights | "
            + ", ".join(
                f"{target}={weight:.3f}"
                for target, weight in zip(config.target_columns, model.target_loss_weights.detach().cpu().tolist())
            ),
            flush=True,
        )
    tft_init_checkpoint = None
    if not args.no_project_best_tft_init:
        tft_init_checkpoint = _load_tft_branch(model, config, freeze=not args.unfreeze_tft)

    train_groups = datasets["train_tpt"]
    val_groups = datasets["val_tpt"]
    test_groups = datasets["test_tpt"]
    train_x = {key: train_groups[key] for key in ("closeness", "period", "trend")}
    val_x = {key: val_groups[key] for key in ("closeness", "period", "trend")}
    test_x = {key: test_groups[key] for key in ("closeness", "period", "trend")}

    checkpoint_path = Path(config.checkpoint_dir) / f"{args.checkpoint_name}.pt"
    best_val_loss = None
    history = None
    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for --eval-only: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        previous_metrics = _load_json(Path(config.output_dir) / "tft_gru_residual_hybrid_metrics.json")
        best_val_loss = previous_metrics.get("best_val_loss")
        history = previous_metrics.get("history")
        print(f"Loaded existing hybrid checkpoint for eval-only mode: {checkpoint_path}", flush=True)
    else:
        result = train_model(
            model=model,
            model_name=args.checkpoint_name,
            train_data=(train_x, train_groups["target"]),
            val_data=(val_x, val_groups["target"]),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
            verbose=True,
        )
        checkpoint_path = result.checkpoint_path
        best_val_loss = result.best_val_loss
        history = result.history

    print("Evaluating on test split...", flush=True)
    processor = datasets["processor"]
    y_val = processor.inverse_transform_targets(val_groups["target"])
    y_true = processor.inverse_transform_targets(test_groups["target"])
    val_pred = processor.inverse_transform_targets(predict_model(model, val_x, config))
    y_pred_raw = processor.inverse_transform_targets(predict_model(model, test_x, config))
    y_pred = y_pred_raw
    calibration = None
    raw_val_metrics = compute_all_metrics(y_val, val_pred)
    tft_val = None
    tft_test = None
    use_ups_routing = args.original_scale_loss and not args.no_ups_routing

    if not args.no_calibration or use_ups_routing:
        print("Fitting validation residual calibration...", flush=True)
        if tft_init_checkpoint:
            _load_tft_branch(model, config, freeze=True)
        tft_val = processor.inverse_transform_targets(_predict_tft_branch(model, val_x, config))
        tft_test = processor.inverse_transform_targets(_predict_tft_branch(model, test_x, config))

    if not args.no_calibration:
        calibration_specs, calibrated_val = _fit_validation_calibration(y_val, val_pred, tft_val)
        calibrated_val_metrics = compute_all_metrics(y_val, calibrated_val)
        if calibrated_val_metrics["RMSE"] <= raw_val_metrics["RMSE"]:
            y_pred = _apply_validation_calibration(y_pred_raw, tft_test, calibration_specs)
            calibration = {
                "enabled": True,
                "selected": True,
                "target_specs": dict(zip(config.target_columns, calibration_specs)),
                "raw_val_metrics": raw_val_metrics,
                "calibrated_val_metrics": calibrated_val_metrics,
            }
            print(
                "Using calibrated predictions | "
                f"val RMSE {raw_val_metrics['RMSE']:.4f} -> {calibrated_val_metrics['RMSE']:.4f}",
                flush=True,
            )
        else:
            calibration = {
                "enabled": True,
                "selected": False,
                "raw_val_metrics": raw_val_metrics,
                "calibrated_val_metrics": calibrated_val_metrics,
            }
            print("Calibration did not improve validation RMSE; using raw hybrid predictions.", flush=True)

    routing = None
    if use_ups_routing:
        loss_weights = _target_scale_loss_weights(processor, args.loss_weight_power).numpy()
        routed_targets = []
        for idx, target in enumerate(config.target_columns):
            if loss_weights[idx] < 0.01:
                y_pred[:, idx] = tft_test[:, idx]
                routed_targets.append(target)
        if routed_targets:
            routing = {
                "enabled": True,
                "strategy": "tft_fallback_for_near_zero_loss_weight_targets",
                "loss_weight_threshold": 0.01,
                "loss_weights": {
                    target: float(weight)
                    for target, weight in zip(config.target_columns, loss_weights)
                },
                "routed_targets": routed_targets,
            }
            print(
                "Using TFT fallback routing for UPS balance | "
                + ", ".join(routed_targets),
                flush=True,
            )

    metrics = compute_all_metrics(y_true, y_pred)
    metrics["UPS"] = compute_urban_prediction_score(y_true, y_pred, config.target_columns)
    per_target = compute_metrics_by_target(y_true, y_pred, config.target_columns)

    output_frame = pd.DataFrame({"timestamp": test_groups["timestamp"].reset_index(drop=True)})
    for idx, target in enumerate(config.target_columns):
        output_frame[f"actual_{target}"] = y_true[:, idx]
        output_frame[f"predicted_{target}"] = y_pred[:, idx]
        output_frame[f"residual_{target}"] = y_true[:, idx] - y_pred[:, idx]

    predictions_path = Path(config.output_dir) / "tft_gru_residual_hybrid_predictions.csv"
    metrics_path = Path(config.output_dir) / "tft_gru_residual_hybrid_metrics.json"
    summary = {
        "model": "TFTGRUResidualHybrid",
        "architecture": "TemporalFusionTransformer + GRU + target-wise residual correction",
        "checkpoint_path": str(checkpoint_path),
        "prediction_path": str(predictions_path),
        "targets": list(config.target_columns),
        "target_output_map": {target: idx for idx, target in enumerate(config.target_columns)},
        "best_val_loss": best_val_loss,
        "tft_init_checkpoint": tft_init_checkpoint,
        "calibration": calibration,
        "routing": routing,
        "metrics": metrics,
        "per_target_metrics": per_target,
        "history": history,
    }

    previous_summary = _load_json(metrics_path)
    previous_rmse = previous_summary.get("metrics", {}).get("RMSE")
    should_save = True
    if args.keep_best and previous_rmse is not None and metrics["RMSE"] >= float(previous_rmse):
        should_save = False
        print(
            f"Keeping previous best canonical outputs | "
            f"new RMSE={metrics['RMSE']:.4f}, previous RMSE={float(previous_rmse):.4f}",
            flush=True,
        )

    if should_save:
        print("Saving mapped predictions and metrics...", flush=True)
        output_frame.to_csv(predictions_path, index=False)
        if checkpoint_path.name != "tft_gru_residual_hybrid.pt" and not args.eval_only:
            canonical_checkpoint = Path(config.checkpoint_dir) / "tft_gru_residual_hybrid.pt"
            canonical_checkpoint.write_bytes(Path(checkpoint_path).read_bytes())
            summary["checkpoint_path"] = str(canonical_checkpoint)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(summary if should_save else previous_summary, handle, indent=2)

    print("TFT + GRU residual hybrid trained")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(pd.DataFrame(per_target).T[["MAE", "RMSE", "NRMSE"]].round(4).to_string())


if __name__ == "__main__":
    main()
