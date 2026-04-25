from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd

from models.bilstm import EnhancedBiLSTM
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics
from utils.training import predict_model, train_model


def _search_config(base_config, hidden_dim: int, seq_len: int, batch_size: int):
    config = copy.deepcopy(base_config)
    config.bilstm_hidden_dim = hidden_dim
    config.seq_len = seq_len
    config.batch_size = batch_size
    config.closeness_lags = tuple(range(1, seq_len + 1))
    config.period_lags = tuple(range(24, 24 + seq_len))
    config.trend_lags = tuple(range(24 * 7, 24 * 7 + seq_len))
    return config


def _dataset_groups(grouped):
    return (
        {key: grouped[key] for key in ("closeness", "period", "trend")},
        grouped["target"],
    )


def run_bilstm_search(base_config=CONFIG):
    base_config.output_dir.mkdir(parents=True, exist_ok=True)
    base_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seed(base_config.random_seed)

    raw_df = load_input_dataframe(base_config)
    search_results = []
    best_result = None
    total_runs = 3 * 2 * 2
    run_idx = 0

    for hidden_dim in (8, 16, 24):
        for seq_len in (8, 24):
            for batch_size in (32, 50):
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] Testing BiLSTM hidden_dim={hidden_dim}, seq_len={seq_len}, batch_size={batch_size}",
                    flush=True,
                )
                config = _search_config(base_config, hidden_dim, seq_len, batch_size)
                datasets = create_datasets(config, raw_df)
                input_dim = datasets["train_tpt"]["closeness"].shape[-1]
                model = EnhancedBiLSTM(
                    input_dim=input_dim,
                    hidden_dim=config.bilstm_hidden_dim,
                    num_layers=config.bilstm_layers,
                    dropout=config.dropout,
                    output_dim=len(config.target_columns),
                )
                run_name = f"bilstm_h{hidden_dim}_s{seq_len}_b{batch_size}"
                result = train_model(
                    model=model,
                    model_name=run_name,
                    train_data=_dataset_groups(datasets["train_tpt"]),
                    val_data=_dataset_groups(datasets["val_tpt"]),
                    config=config,
                    checkpoint_dir=Path(config.checkpoint_dir),
                )

                val_x, y_val_scaled = _dataset_groups(datasets["val_tpt"])
                val_pred_scaled = predict_model(model, val_x, config)
                y_val = datasets["processor"].inverse_transform_targets(y_val_scaled)
                y_pred = datasets["processor"].inverse_transform_targets(val_pred_scaled)
                metrics = compute_all_metrics(y_val, y_pred)

                row = {
                    "hidden_dim": hidden_dim,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "NRMSE": metrics["NRMSE"],
                    "checkpoint_path": str(result.checkpoint_path),
                }
                search_results.append(row)
                print(
                    f"    Validation -> MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, NRMSE={row['NRMSE']:.4f}",
                    flush=True,
                )

                if best_result is None or (row["RMSE"], row["MAE"]) < (best_result["RMSE"], best_result["MAE"]):
                    best_result = row
                    print(
                        "    New best configuration found",
                        flush=True,
                    )

    results_df = pd.DataFrame(search_results).sort_values(["RMSE", "MAE", "hidden_dim", "seq_len", "batch_size"])
    results_path = Path(base_config.output_dir) / "bilstm_hyperparameter_search_results.csv"
    results_df.to_csv(results_path, index=False)

    best_config_path = Path(base_config.output_dir) / "best_bilstm_hyperparameters.json"
    with open(best_config_path, "w", encoding="utf-8") as handle:
        json.dump(best_result, handle, indent=2)

    best_checkpoint = Path(base_config.checkpoint_dir) / "bilstm_best_search.pt"
    shutil.copy2(best_result["checkpoint_path"], best_checkpoint)

    return results_df, best_result, results_path, best_config_path, best_checkpoint


def main():
    results_df, best_result, results_path, best_config_path, best_checkpoint = run_bilstm_search(CONFIG)
    print("\nBiLSTM Hyperparameter Search Results")
    print(results_df[["hidden_dim", "seq_len", "batch_size", "MAE", "RMSE", "NRMSE"]].round(4).to_string(index=False))
    print("\nBest Configuration")
    print(
        f"hidden_dim={best_result['hidden_dim']}, "
        f"seq_len={best_result['seq_len']}, "
        f"batch_size={best_result['batch_size']}, "
        f"MAE={best_result['MAE']:.4f}, RMSE={best_result['RMSE']:.4f}"
    )
    print(f"Results table saved to {results_path}")
    print(f"Best config saved to {best_config_path}")
    print(f"Best checkpoint copied to {best_checkpoint}")


if __name__ == "__main__":
    main()
