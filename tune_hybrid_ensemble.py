from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from engine.adaptive_ensemble import AdaptiveEnsemble
from models.bilstm import EnhancedBiLSTM
from models.hybrid import AdaptiveHybridModel
from models.transformer import AdvancedTimeSeriesTransformer
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics, compute_urban_prediction_score
from utils.training import predict_model, train_model


def _load_fixed_models(input_dim: int, config):
    output_dim = len(config.target_columns)
    models = {
        "BiLSTM": EnhancedBiLSTM(
            input_dim=input_dim,
            hidden_dim=config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            dropout=config.dropout,
            output_dim=output_dim,
        ),
        "Transformer": AdvancedTimeSeriesTransformer(
            input_dim=input_dim,
            d_model=config.transformer_d_model,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        ),
    }
    for name, model in models.items():
        checkpoint = Path(config.checkpoint_dir) / f"{name.lower()}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {checkpoint}. Run `python3 main.py` before tuning the Hybrid + AdaptiveEnsemble setup."
            )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return models


def _dataset_groups(grouped):
    return (
        {key: grouped[key] for key in ("closeness", "period", "trend")},
        grouped["target"],
    )


def run_search(base_config=CONFIG):
    set_seed(base_config.random_seed)
    raw_df = load_input_dataframe(base_config)
    datasets = create_datasets(base_config, raw_df)
    input_dim = datasets["train_tpt"]["closeness"].shape[-1]
    fixed_models = _load_fixed_models(input_dim, base_config)

    train_data = _dataset_groups(datasets["train_tpt"])
    val_x, y_val_scaled = _dataset_groups(datasets["val_tpt"])
    processor = datasets["processor"]
    y_val = processor.inverse_transform_targets(y_val_scaled)

    results = []
    best = None
    total_runs = 3 * 2 * 2
    run_idx = 0

    for dense_hidden_dim in (32, 64, 96):
        for dropout in (0.1, 0.2):
            for ensemble_error_window in (48, 72):
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] Testing dense_hidden_dim={dense_hidden_dim}, dropout={dropout}, ensemble_error_window={ensemble_error_window}",
                    flush=True,
                )
                config = copy.deepcopy(base_config)
                config.dense_hidden_dim = dense_hidden_dim
                config.dropout = dropout
                config.ensemble_error_window = ensemble_error_window

                hybrid = AdaptiveHybridModel(input_dim=input_dim, config=config)
                model_name = f"hybrid_d{dense_hidden_dim}_do{str(dropout).replace('.', '')}_ew{ensemble_error_window}"
                result = train_model(
                    model=hybrid,
                    model_name=model_name,
                    train_data=train_data,
                    val_data=(val_x, y_val_scaled),
                    config=config,
                    checkpoint_dir=Path(config.checkpoint_dir),
                )

                val_predictions = {}
                for name, fixed_model in fixed_models.items():
                    val_scaled = predict_model(fixed_model, val_x, config)
                    val_predictions[name] = processor.inverse_transform_targets(val_scaled)

                hybrid_val_scaled = predict_model(hybrid, val_x, config)
                hybrid_val_pred = processor.inverse_transform_targets(hybrid_val_scaled)
                val_predictions["Hybrid"] = hybrid_val_pred

                ensemble = AdaptiveEnsemble(
                    model_names=["BiLSTM", "Transformer", "Hybrid"],
                    error_window=ensemble_error_window,
                )
                ensemble.update_errors(y_val, val_predictions)
                ensemble.fit_meta_learner(y_val, val_predictions)
                ensemble_val_pred = ensemble.predict(val_predictions)

                hybrid_metrics = compute_all_metrics(y_val, hybrid_val_pred)
                ensemble_metrics = compute_all_metrics(y_val, ensemble_val_pred)
                ensemble_ups = compute_urban_prediction_score(y_val, ensemble_val_pred, config.target_columns)

                row = {
                    "dense_hidden_dim": dense_hidden_dim,
                    "dropout": dropout,
                    "ensemble_error_window": ensemble_error_window,
                    "hybrid_MAE": hybrid_metrics["MAE"],
                    "hybrid_RMSE": hybrid_metrics["RMSE"],
                    "ensemble_MAE": ensemble_metrics["MAE"],
                    "ensemble_RMSE": ensemble_metrics["RMSE"],
                    "ensemble_NRMSE": ensemble_metrics["NRMSE"],
                    "ensemble_UPS": ensemble_ups,
                    "checkpoint_path": str(result.checkpoint_path),
                }
                results.append(row)
                print(
                    "    Validation -> "
                    f"Hybrid RMSE={row['hybrid_RMSE']:.4f}, "
                    f"Ensemble RMSE={row['ensemble_RMSE']:.4f}, "
                    f"Ensemble MAE={row['ensemble_MAE']:.4f}, "
                    f"UPS={row['ensemble_UPS']:.4f}",
                    flush=True,
                )

                if best is None or (row["ensemble_RMSE"], row["ensemble_MAE"]) < (best["ensemble_RMSE"], best["ensemble_MAE"]):
                    best = row
                    print("    New best configuration found", flush=True)

    results_df = pd.DataFrame(results).sort_values(
        ["ensemble_RMSE", "ensemble_MAE", "ensemble_UPS", "dense_hidden_dim"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    results_path = Path(base_config.output_dir) / "hybrid_ensemble_search_results.csv"
    results_df.to_csv(results_path, index=False)

    best_path = Path(base_config.output_dir) / "best_hybrid_ensemble_config.json"
    with open(best_path, "w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    best_checkpoint = Path(base_config.checkpoint_dir) / "hybrid_best_search.pt"
    shutil.copy2(best["checkpoint_path"], best_checkpoint)

    return results_df, best, results_path, best_path, best_checkpoint


def main():
    results_df, best, results_path, best_path, best_checkpoint = run_search(CONFIG)
    print("\nHybrid + AdaptiveEnsemble Search Results")
    print(
        results_df[
            [
                "dense_hidden_dim",
                "dropout",
                "ensemble_error_window",
                "hybrid_MAE",
                "hybrid_RMSE",
                "ensemble_MAE",
                "ensemble_RMSE",
                "ensemble_NRMSE",
                "ensemble_UPS",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nBest Configuration")
    print(json.dumps(best, indent=2))
    print(f"\nResults table saved to {results_path}")
    print(f"Best config saved to {best_path}")
    print(f"Best checkpoint copied to {best_checkpoint}")


if __name__ == "__main__":
    main()
