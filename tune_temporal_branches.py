from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from models.hybrid import TFTGRUResidualHybrid
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics
from utils.training import predict_model, train_model


def _dataset_groups(grouped):
    return (
        {key: grouped[key] for key in ("closeness", "period", "trend")},
        grouped["target"],
    )


def _window(length: int, anchor: int) -> tuple[int, ...]:
    return tuple(range(anchor, anchor + length))


def run_search(base_config=CONFIG):
    set_seed(base_config.random_seed)
    raw_df = load_input_dataframe(base_config)

    results = []
    best = None
    candidate_lengths = [
        (4, 4, 4),
        (8, 8, 8),
        (12, 12, 12),
        (8, 4, 8),
        (8, 12, 8),
        (4, 8, 8),
        (12, 8, 8),
        (8, 8, 4),
        (8, 8, 12),
    ]

    for closeness_len, period_len, trend_len in candidate_lengths:
        config = copy.deepcopy(base_config)
        config.closeness_lags = _window(closeness_len, 1)
        config.period_lags = _window(period_len, 24)
        config.trend_lags = _window(trend_len, 24 * 7)
        config.seq_len = max(closeness_len, period_len, trend_len)
        config.epochs = min(base_config.epochs, 4)
        config.patience = min(base_config.patience, 2)

        datasets = create_datasets(config, raw_df)
        input_dim = datasets["train_tpt"]["closeness"].shape[-1]
        model = TFTGRUResidualHybrid(input_dim=input_dim, config=config)

        train_groups = datasets["train_tpt"]
        val_groups = datasets["val_tpt"]
        train_data = _dataset_groups(train_groups)
        val_x, y_val_scaled = _dataset_groups(val_groups)

        result = train_model(
            model=model,
            model_name=f"hybrid_c{closeness_len}_p{period_len}_t{trend_len}",
            train_data=train_data,
            val_data=(val_x, y_val_scaled),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )

        y_val_pred_scaled = predict_model(model, val_x, config)
        processor = datasets["processor"]
        y_val = processor.inverse_transform_targets(y_val_scaled)
        y_val_pred = processor.inverse_transform_targets(y_val_pred_scaled)
        metrics = compute_all_metrics(y_val, y_val_pred)

        row = {
            "closeness_length": closeness_len,
            "period_length": period_len,
            "trend_length": trend_len,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "NRMSE": metrics["NRMSE"],
            "checkpoint_path": str(result.checkpoint_path),
        }
        results.append(row)
        print(
            f"checked c={closeness_len}, p={period_len}, t={trend_len} -> "
            f"MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, NRMSE={row['NRMSE']:.4f}"
        )

        if best is None or (row["RMSE"], row["MAE"]) < (best["RMSE"], best["MAE"]):
            best = row

    results_df = pd.DataFrame(results).sort_values(["RMSE", "MAE", "closeness_length", "period_length", "trend_length"])
    results_path = Path(base_config.output_dir) / "temporal_branch_search_results.csv"
    results_df.to_csv(results_path, index=False)

    best_path = Path(base_config.output_dir) / "best_temporal_branch_config.json"
    with open(best_path, "w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    return results_df, best, results_path, best_path


def main():
    results_df, best, results_path, best_path = run_search(CONFIG)
    print("\nTemporal Branch Length Search Results")
    print(results_df[["closeness_length", "period_length", "trend_length", "MAE", "RMSE", "NRMSE"]].round(4).to_string(index=False))
    print("\nBest Configuration")
    print(json.dumps(best, indent=2))
    print(f"\nResults table saved to {results_path}")
    print(f"Best config saved to {best_path}")


if __name__ == "__main__":
    main()
