from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import pandas as pd

from models.bilstm import EnhancedBiLSTM
from models.hybrid import AdaptiveHybridModel
from models.informer import InformerForecastModel
from models.patchtst import PatchTSTForecastModel
from models.transformer import TemporalFusionTransformer
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score
from utils.training import predict_model, train_model


def _dataset_groups(grouped):
    return (
        {key: grouped[key] for key in ("closeness", "period", "trend")},
        grouped["target"],
    )


def _window(length: int, anchor: int) -> tuple[int, ...]:
    return tuple(range(anchor, anchor + length))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _reference_metrics(output_dir: Path) -> dict:
    summary = _load_json(output_dir / "summary.json")
    return summary.get("offline_metrics", {})


def _merge_with_existing_summary(summary_path: Path, selected: list[str], updated_results: dict) -> dict:
    canonical_order = ["BiLSTM", "TFT", "Hybrid", "Informer", "PatchTST"]
    existing_summary = _load_json(summary_path)
    existing_results = existing_summary.get("results", {})

    merged_results = {}
    for model_name in canonical_order:
        if model_name in updated_results:
            merged_results[model_name] = updated_results[model_name]
        elif model_name in existing_results:
            merged_results[model_name] = existing_results[model_name]

    merged_models = [model_name for model_name in canonical_order if model_name in merged_results]
    return {
        "search_protocol": {
            "selection_rule": "lowest validation RMSE, MAE as tie-breaker",
            "training_budget": {
                "epochs": 16,
                "patience": 5,
            },
            "models": merged_models,
            "updated_models": selected,
        },
        "results": merged_results,
    }


def _base_training_config(base_config):
    config = copy.deepcopy(base_config)
    config.epochs = 16
    config.patience = 5
    config.lr_scheduler_patience = max(base_config.lr_scheduler_patience, 2)
    config.lr_scheduler_factor = min(base_config.lr_scheduler_factor, 0.5)
    return config


def _validation_and_test_metrics(model, datasets, config):
    processor = datasets["processor"]
    val_x, y_val_scaled = _dataset_groups(datasets["val_tpt"])
    test_x, y_test_scaled = _dataset_groups(datasets["test_tpt"])

    val_pred_scaled = predict_model(model, val_x, config)
    test_pred_scaled = predict_model(model, test_x, config)

    y_val = processor.inverse_transform_targets(y_val_scaled)
    y_test = processor.inverse_transform_targets(y_test_scaled)
    y_val_pred = processor.inverse_transform_targets(val_pred_scaled)
    y_test_pred = processor.inverse_transform_targets(test_pred_scaled)

    val_metrics = compute_all_metrics(y_val, y_val_pred)
    val_metrics["UPS"] = compute_urban_prediction_score(y_val, y_val_pred, config.target_columns)

    test_metrics = compute_all_metrics(y_test, y_test_pred)
    test_metrics["UPS"] = compute_urban_prediction_score(y_test, y_test_pred, config.target_columns)

    return {
        "y_test": y_test,
        "test_pred": y_test_pred,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_per_target": compute_metrics_by_target(y_test, y_test_pred, config.target_columns),
    }


def _build_bilstm_model(input_dim: int, config):
    return EnhancedBiLSTM(
        input_dim=input_dim,
        hidden_dim=config.bilstm_hidden_dim,
        num_layers=config.bilstm_layers,
        dropout=config.dropout,
        output_dim=len(config.target_columns),
    )


def _build_tft_model(input_dim: int, config):
    return TemporalFusionTransformer(
        input_dim=input_dim,
        hidden_dim=config.tft_hidden_dim,
        nhead=config.tft_heads,
        num_layers=config.tft_layers,
        dim_feedforward=config.tft_ff_dim,
        dropout=config.dropout,
        output_dim=len(config.target_columns),
    )


def _build_hybrid_model(input_dim: int, config):
    return AdaptiveHybridModel(input_dim=input_dim, config=config)


def _build_informer_model(input_dim: int, config):
    return InformerForecastModel(
        input_dim=input_dim,
        d_model=config.informer_d_model,
        nhead=config.informer_heads,
        num_layers=config.informer_layers,
        dim_feedforward=config.informer_ff_dim,
        dropout=config.dropout,
        output_dim=len(config.target_columns),
    )


def _build_patchtst_model(input_dim: int, config):
    return PatchTSTForecastModel(
        input_dim=input_dim,
        d_model=config.patchtst_d_model,
        nhead=config.patchtst_heads,
        num_layers=config.patchtst_layers,
        dim_feedforward=config.patchtst_ff_dim,
        dropout=config.dropout,
        output_dim=len(config.target_columns),
        patch_len=config.patchtst_patch_len,
        stride=config.patchtst_stride,
    )


def _search_bilstm(base_config, raw_df):
    artifact = _load_json(Path(base_config.output_dir) / "best_bilstm_hyperparameters.json")
    candidate_specs = [
        {
            "hidden_dim": 16,
            "seq_len": 8,
            "batch_size": 64,
            "dropout": 0.10,
            "learning_rate": 1e-3,
        },
        {
            "hidden_dim": 16,
            "seq_len": 12,
            "batch_size": 64,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "hidden_dim": 24,
            "seq_len": 8,
            "batch_size": 64,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "hidden_dim": 24,
            "seq_len": 12,
            "batch_size": 64,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "hidden_dim": 16,
            "seq_len": 8,
            "batch_size": 96,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "hidden_dim": 24,
            "seq_len": 8,
            "batch_size": 96,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "hidden_dim": 32,
            "seq_len": 12,
            "batch_size": 64,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
    ]
    if artifact:
        candidate_specs.insert(
            0,
            {
                "hidden_dim": int(artifact.get("hidden_dim", 16)),
                "seq_len": int(artifact.get("seq_len", 8)),
                "batch_size": max(64, int(artifact.get("batch_size", 32))),
                "dropout": 0.10,
                "learning_rate": 1e-3,
            },
        )

    rows = []
    best = None
    for trial_idx, spec in enumerate(candidate_specs, start=1):
        print(
            f"Starting BiLSTM trial {trial_idx}/{len(candidate_specs)} with "
            f"h={spec['hidden_dim']} seq={spec['seq_len']} batch={spec['batch_size']} "
            f"dropout={spec['dropout']:.2f} lr={spec['learning_rate']:.4g}",
            flush=True,
        )
        config = _base_training_config(base_config)
        config.bilstm_hidden_dim = spec["hidden_dim"]
        config.seq_len = spec["seq_len"]
        config.batch_size = spec["batch_size"]
        config.dropout = spec["dropout"]
        config.learning_rate = spec["learning_rate"]
        config.closeness_lags = _window(spec["seq_len"], 1)
        config.period_lags = _window(spec["seq_len"], 24)
        config.trend_lags = _window(spec["seq_len"], 24 * 7)

        datasets = create_datasets(config, raw_df)
        input_dim = datasets["train_tpt"]["closeness"].shape[-1]
        model = _build_bilstm_model(input_dim, config)
        result = train_model(
            model=model,
            model_name=f"project_best_bilstm_trial_{trial_idx}",
            train_data=_dataset_groups(datasets["train_tpt"]),
            val_data=_dataset_groups(datasets["val_tpt"]),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )
        metrics = _validation_and_test_metrics(model, datasets, config)
        row = {
            "trial": trial_idx,
            **spec,
            **{f"val_{name}": value for name, value in metrics["val_metrics"].items()},
            "checkpoint_path": str(result.checkpoint_path),
        }
        rows.append(row)
        print(
            f"[BiLSTM {trial_idx}/{len(candidate_specs)}] "
            f"h={spec['hidden_dim']} seq={spec['seq_len']} batch={spec['batch_size']} "
            f"dropout={spec['dropout']:.2f} lr={spec['learning_rate']:.4g} -> "
            f"val RMSE={row['val_RMSE']:.4f}, val UPS={row['val_UPS']:.4f}",
            flush=True,
        )
        if best is None or (row["val_RMSE"], row["val_MAE"]) < (best["row"]["val_RMSE"], best["row"]["val_MAE"]):
            best = {"row": row, "test": metrics}

    return rows, best


def _search_tft(base_config, raw_df):
    candidate_specs = [
        {
            "seq_len": 8,
            "hidden_dim": 32,
            "heads": 2,
            "layers": 1,
            "ff_dim": 96,
            "batch_size": 64,
            "dropout": 0.05,
            "learning_rate": 5e-4,
        },
        {
            "seq_len": 8,
            "hidden_dim": 32,
            "heads": 4,
            "layers": 1,
            "ff_dim": 128,
            "batch_size": 64,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "seq_len": 12,
            "hidden_dim": 48,
            "heads": 4,
            "layers": 1,
            "ff_dim": 144,
            "batch_size": 64,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "seq_len": 12,
            "hidden_dim": 48,
            "heads": 4,
            "layers": 2,
            "ff_dim": 192,
            "batch_size": 64,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "seq_len": 8,
            "hidden_dim": 48,
            "heads": 2,
            "layers": 1,
            "ff_dim": 96,
            "batch_size": 96,
            "dropout": 0.05,
            "learning_rate": 5e-4,
        },
        {
            "seq_len": 12,
            "hidden_dim": 64,
            "heads": 4,
            "layers": 2,
            "ff_dim": 192,
            "batch_size": 96,
            "dropout": 0.10,
            "learning_rate": 3e-4,
        },
    ]

    rows = []
    best = None
    for trial_idx, spec in enumerate(candidate_specs, start=1):
        print(
            f"Starting TFT trial {trial_idx}/{len(candidate_specs)} with "
            f"seq={spec['seq_len']} hidden={spec['hidden_dim']} heads={spec['heads']} "
            f"layers={spec['layers']} ff={spec['ff_dim']} "
            f"dropout={spec['dropout']:.2f} lr={spec['learning_rate']:.4g}",
            flush=True,
        )
        config = _base_training_config(base_config)
        config.seq_len = spec["seq_len"]
        config.batch_size = spec["batch_size"]
        config.dropout = spec["dropout"]
        config.learning_rate = spec["learning_rate"]
        config.tft_hidden_dim = spec["hidden_dim"]
        config.tft_heads = spec["heads"]
        config.tft_layers = spec["layers"]
        config.tft_ff_dim = spec["ff_dim"]
        config.closeness_lags = _window(spec["seq_len"], 1)
        config.period_lags = _window(spec["seq_len"], 24)
        config.trend_lags = _window(spec["seq_len"], 24 * 7)

        datasets = create_datasets(config, raw_df)
        input_dim = datasets["train_tpt"]["closeness"].shape[-1]
        model = _build_tft_model(input_dim, config)
        result = train_model(
            model=model,
            model_name=f"project_best_tft_trial_{trial_idx}",
            train_data=_dataset_groups(datasets["train_tpt"]),
            val_data=_dataset_groups(datasets["val_tpt"]),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )
        metrics = _validation_and_test_metrics(model, datasets, config)
        row = {
            "trial": trial_idx,
            **spec,
            **{f"val_{name}": value for name, value in metrics["val_metrics"].items()},
            "checkpoint_path": str(result.checkpoint_path),
        }
        rows.append(row)
        print(
            f"[TFT {trial_idx}/{len(candidate_specs)}] "
            f"seq={spec['seq_len']} hidden={spec['hidden_dim']} heads={spec['heads']} "
            f"layers={spec['layers']} ff={spec['ff_dim']} -> "
            f"val RMSE={row['val_RMSE']:.4f}, val UPS={row['val_UPS']:.4f}",
            flush=True,
        )
        if best is None or (row["val_RMSE"], row["val_MAE"]) < (best["row"]["val_RMSE"], best["row"]["val_MAE"]):
            best = {"row": row, "test": metrics}

    return rows, best


def _search_hybrid(base_config, raw_df):
    branch_artifact = _load_json(Path(base_config.output_dir) / "best_temporal_branch_config.json")
    ensemble_artifact = _load_json(Path(base_config.output_dir) / "best_hybrid_ensemble_config.json")
    candidate_specs = [
        {
            "closeness_len": 4,
            "period_len": 8,
            "trend_len": 4,
            "bilstm_hidden_dim": 16,
            "tft_hidden_dim": 32,
            "tft_heads": 2,
            "tft_layers": 1,
            "tft_ff_dim": 96,
            "dense_hidden_dim": 96,
            "dropout": 0.10,
            "batch_size": 64,
            "learning_rate": 5e-4,
        },
        {
            "closeness_len": 4,
            "period_len": 8,
            "trend_len": 4,
            "bilstm_hidden_dim": 16,
            "tft_hidden_dim": 48,
            "tft_heads": 4,
            "tft_layers": 1,
            "tft_ff_dim": 144,
            "dense_hidden_dim": 96,
            "dropout": 0.10,
            "batch_size": 64,
            "learning_rate": 5e-4,
        },
        {
            "closeness_len": 4,
            "period_len": 8,
            "trend_len": 4,
            "bilstm_hidden_dim": 16,
            "tft_hidden_dim": 32,
            "tft_heads": 2,
            "tft_layers": 2,
            "tft_ff_dim": 128,
            "dense_hidden_dim": 96,
            "dropout": 0.10,
            "batch_size": 64,
            "learning_rate": 5e-4,
        },
        {
            "closeness_len": 4,
            "period_len": 8,
            "trend_len": 4,
            "bilstm_hidden_dim": 24,
            "tft_hidden_dim": 48,
            "tft_heads": 4,
            "tft_layers": 2,
            "tft_ff_dim": 192,
            "dense_hidden_dim": 128,
            "dropout": 0.10,
            "batch_size": 96,
            "learning_rate": 3e-4,
        },
        {
            "closeness_len": 8,
            "period_len": 8,
            "trend_len": 4,
            "bilstm_hidden_dim": 24,
            "tft_hidden_dim": 48,
            "tft_heads": 4,
            "tft_layers": 1,
            "tft_ff_dim": 144,
            "dense_hidden_dim": 128,
            "dropout": 0.10,
            "batch_size": 64,
            "learning_rate": 3e-4,
        },
    ]
    if branch_artifact or ensemble_artifact:
        candidate_specs.insert(
            0,
            {
                "closeness_len": int(branch_artifact.get("closeness_length", 4)),
                "period_len": int(branch_artifact.get("period_length", 8)),
                "trend_len": int(branch_artifact.get("trend_length", 4)),
                "bilstm_hidden_dim": int(_load_json(Path(base_config.output_dir) / "best_bilstm_hyperparameters.json").get("hidden_dim", 16)),
                "tft_hidden_dim": 32,
                "tft_heads": 2,
                "tft_layers": 1,
                "tft_ff_dim": 96,
                "dense_hidden_dim": int(ensemble_artifact.get("dense_hidden_dim", 96)),
                "dropout": float(ensemble_artifact.get("dropout", 0.1)),
                "batch_size": 64,
                "learning_rate": 5e-4,
            },
        )

    rows = []
    best = None
    for trial_idx, spec in enumerate(candidate_specs, start=1):
        print(
            f"Starting Hybrid trial {trial_idx}/{len(candidate_specs)} with "
            f"c={spec['closeness_len']} p={spec['period_len']} t={spec['trend_len']} "
            f"bilstm_h={spec['bilstm_hidden_dim']} tft_h={spec['tft_hidden_dim']} "
            f"dense={spec['dense_hidden_dim']} dropout={spec['dropout']:.2f} "
            f"lr={spec['learning_rate']:.4g}",
            flush=True,
        )
        config = _base_training_config(base_config)
        config.closeness_lags = _window(spec["closeness_len"], 1)
        config.period_lags = _window(spec["period_len"], 24)
        config.trend_lags = _window(spec["trend_len"], 24 * 7)
        config.seq_len = max(spec["closeness_len"], spec["period_len"], spec["trend_len"])
        config.bilstm_hidden_dim = spec["bilstm_hidden_dim"]
        config.tft_hidden_dim = spec["tft_hidden_dim"]
        config.tft_heads = spec["tft_heads"]
        config.tft_layers = spec["tft_layers"]
        config.tft_ff_dim = spec["tft_ff_dim"]
        config.dense_hidden_dim = spec["dense_hidden_dim"]
        config.dropout = spec["dropout"]
        config.batch_size = spec["batch_size"]
        config.learning_rate = spec["learning_rate"]

        datasets = create_datasets(config, raw_df)
        input_dim = datasets["train_tpt"]["closeness"].shape[-1]
        model = _build_hybrid_model(input_dim, config)
        result = train_model(
            model=model,
            model_name=f"project_best_hybrid_trial_{trial_idx}",
            train_data=_dataset_groups(datasets["train_tpt"]),
            val_data=_dataset_groups(datasets["val_tpt"]),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )
        metrics = _validation_and_test_metrics(model, datasets, config)
        row = {
            "trial": trial_idx,
            **spec,
            **{f"val_{name}": value for name, value in metrics["val_metrics"].items()},
            "checkpoint_path": str(result.checkpoint_path),
        }
        rows.append(row)
        print(
            f"[Hybrid {trial_idx}/{len(candidate_specs)}] "
            f"c={spec['closeness_len']} p={spec['period_len']} t={spec['trend_len']} "
            f"bilstm_h={spec['bilstm_hidden_dim']} tft_h={spec['tft_hidden_dim']} "
            f"dense={spec['dense_hidden_dim']} -> "
            f"val RMSE={row['val_RMSE']:.4f}, val UPS={row['val_UPS']:.4f}",
            flush=True,
        )
        if best is None or (row["val_RMSE"], row["val_MAE"]) < (best["row"]["val_RMSE"], best["row"]["val_MAE"]):
            best = {"row": row, "test": metrics}

    return rows, best


def _search_informer(base_config, raw_df):
    candidate_specs = [
        {
            "seq_len": 8,
            "d_model": 32,
            "heads": 4,
            "layers": 1,
            "ff_dim": 96,
            "batch_size": 128,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "seq_len": 12,
            "d_model": 48,
            "heads": 4,
            "layers": 1,
            "ff_dim": 128,
            "batch_size": 128,
            "dropout": 0.10,
            "learning_rate": 3e-4,
        },
    ]

    rows = []
    best = None
    for trial_idx, spec in enumerate(candidate_specs, start=1):
        print(
            f"Starting Informer trial {trial_idx}/{len(candidate_specs)} with "
            f"seq={spec['seq_len']} d_model={spec['d_model']} heads={spec['heads']} "
            f"layers={spec['layers']} ff={spec['ff_dim']} "
            f"dropout={spec['dropout']:.2f} lr={spec['learning_rate']:.4g}",
            flush=True,
        )
        config = _base_training_config(base_config)
        config.epochs = 4
        config.patience = 1
        config.seq_len = spec["seq_len"]
        config.batch_size = spec["batch_size"]
        config.dropout = spec["dropout"]
        config.learning_rate = spec["learning_rate"]
        config.informer_d_model = spec["d_model"]
        config.informer_heads = spec["heads"]
        config.informer_layers = spec["layers"]
        config.informer_ff_dim = spec["ff_dim"]
        config.closeness_lags = _window(spec["seq_len"], 1)
        config.period_lags = _window(spec["seq_len"], 24)
        config.trend_lags = _window(spec["seq_len"], 24 * 7)

        datasets = create_datasets(config, raw_df)
        input_dim = datasets["train_tpt"]["closeness"].shape[-1]
        model = _build_informer_model(input_dim, config)
        result = train_model(
            model=model,
            model_name=f"project_best_informer_trial_{trial_idx}",
            train_data=_dataset_groups(datasets["train_tpt"]),
            val_data=_dataset_groups(datasets["val_tpt"]),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )
        metrics = _validation_and_test_metrics(model, datasets, config)
        row = {
            "trial": trial_idx,
            **spec,
            **{f"val_{name}": value for name, value in metrics["val_metrics"].items()},
            "checkpoint_path": str(result.checkpoint_path),
        }
        rows.append(row)
        print(
            f"[Informer {trial_idx}/{len(candidate_specs)}] "
            f"seq={spec['seq_len']} d_model={spec['d_model']} heads={spec['heads']} "
            f"layers={spec['layers']} -> val RMSE={row['val_RMSE']:.4f}, val UPS={row['val_UPS']:.4f}",
            flush=True,
        )
        if best is None or (row["val_RMSE"], row["val_MAE"]) < (best["row"]["val_RMSE"], best["row"]["val_MAE"]):
            best = {"row": row, "test": metrics}

    return rows, best


def _search_patchtst(base_config, raw_df):
    candidate_specs = [
        {
            "seq_len": 12,
            "d_model": 32,
            "heads": 4,
            "layers": 1,
            "ff_dim": 96,
            "patch_len": 4,
            "stride": 2,
            "batch_size": 128,
            "dropout": 0.10,
            "learning_rate": 5e-4,
        },
        {
            "seq_len": 16,
            "d_model": 48,
            "heads": 4,
            "layers": 1,
            "ff_dim": 128,
            "patch_len": 4,
            "stride": 2,
            "batch_size": 128,
            "dropout": 0.10,
            "learning_rate": 3e-4,
        },
    ]

    rows = []
    best = None
    for trial_idx, spec in enumerate(candidate_specs, start=1):
        print(
            f"Starting PatchTST trial {trial_idx}/{len(candidate_specs)} with "
            f"seq={spec['seq_len']} d_model={spec['d_model']} heads={spec['heads']} "
            f"layers={spec['layers']} patch={spec['patch_len']} stride={spec['stride']} "
            f"dropout={spec['dropout']:.2f} lr={spec['learning_rate']:.4g}",
            flush=True,
        )
        config = _base_training_config(base_config)
        config.epochs = 4
        config.patience = 1
        config.seq_len = spec["seq_len"]
        config.batch_size = spec["batch_size"]
        config.dropout = spec["dropout"]
        config.learning_rate = spec["learning_rate"]
        config.patchtst_d_model = spec["d_model"]
        config.patchtst_heads = spec["heads"]
        config.patchtst_layers = spec["layers"]
        config.patchtst_ff_dim = spec["ff_dim"]
        config.patchtst_patch_len = spec["patch_len"]
        config.patchtst_stride = spec["stride"]
        config.closeness_lags = _window(spec["seq_len"], 1)
        config.period_lags = _window(spec["seq_len"], 24)
        config.trend_lags = _window(spec["seq_len"], 24 * 7)

        datasets = create_datasets(config, raw_df)
        input_dim = datasets["train_tpt"]["closeness"].shape[-1]
        model = _build_patchtst_model(input_dim, config)
        result = train_model(
            model=model,
            model_name=f"project_best_patchtst_trial_{trial_idx}",
            train_data=_dataset_groups(datasets["train_tpt"]),
            val_data=_dataset_groups(datasets["val_tpt"]),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )
        metrics = _validation_and_test_metrics(model, datasets, config)
        row = {
            "trial": trial_idx,
            **spec,
            **{f"val_{name}": value for name, value in metrics["val_metrics"].items()},
            "checkpoint_path": str(result.checkpoint_path),
        }
        rows.append(row)
        print(
            f"[PatchTST {trial_idx}/{len(candidate_specs)}] "
            f"seq={spec['seq_len']} d_model={spec['d_model']} patch={spec['patch_len']} "
            f"stride={spec['stride']} -> val RMSE={row['val_RMSE']:.4f}, val UPS={row['val_UPS']:.4f}",
            flush=True,
        )
        if best is None or (row["val_RMSE"], row["val_MAE"]) < (best["row"]["val_RMSE"], best["row"]["val_MAE"]):
            best = {"row": row, "test": metrics}

    return rows, best


def _write_search_outputs(model_name: str, rows: list[dict], best: dict, output_dir: Path, checkpoint_dir: Path, reference: dict):
    search_df = pd.DataFrame(rows).sort_values(["val_RMSE", "val_MAE"]).reset_index(drop=True)
    csv_path = output_dir / f"project_best_{model_name.lower()}_search.csv"
    search_df.to_csv(csv_path, index=False)

    best_config = dict(best["row"])
    best_path = output_dir / f"project_best_{model_name.lower()}_best.json"
    with open(best_path, "w", encoding="utf-8") as handle:
        json.dump(best_config, handle, indent=2)

    best_checkpoint = checkpoint_dir / f"project_best_{model_name.lower()}.pt"
    shutil.copy2(best["row"]["checkpoint_path"], best_checkpoint)

    test_metrics = best["test"]["test_metrics"]
    previous = reference.get(model_name, {})
    comparison = {}
    if previous:
        for metric_name in ("MAE", "RMSE", "NRMSE", "UPS"):
            if metric_name in previous:
                comparison[metric_name] = {
                    "previous": float(previous[metric_name]),
                    "project_best": float(test_metrics[metric_name]),
                    "delta": float(test_metrics[metric_name] - previous[metric_name]),
                }

    return {
        "search_csv": str(csv_path),
        "best_json": str(best_path),
        "best_checkpoint": str(best_checkpoint),
        "best_validation": best["row"],
        "test_metrics": test_metrics,
        "test_per_target": best["test"]["test_per_target"],
        "comparison_to_current": comparison,
    }


def run_project_best_track(base_config=CONFIG, models: list[str] | None = None):
    selected = models or ["BiLSTM", "TFT", "Hybrid", "Informer", "PatchTST"]
    set_seed(base_config.random_seed)
    base_config.output_dir.mkdir(parents=True, exist_ok=True)
    base_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_input_dataframe(base_config)
    reference = _reference_metrics(Path(base_config.output_dir))
    updated_results = {}

    searchers = {
        "BiLSTM": _search_bilstm,
        "TFT": _search_tft,
        "Hybrid": _search_hybrid,
        "Informer": _search_informer,
        "PatchTST": _search_patchtst,
    }
    for model_name in selected:
        rows, best = searchers[model_name](base_config, raw_df)
        updated_results[model_name] = _write_search_outputs(
            model_name=model_name,
            rows=rows,
            best=best,
            output_dir=Path(base_config.output_dir),
            checkpoint_dir=Path(base_config.checkpoint_dir),
            reference=reference,
        )

    summary_path = Path(base_config.output_dir) / "project_best_summary.json"
    summary = _merge_with_existing_summary(summary_path, selected, updated_results)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary, summary_path


def main():
    parser = argparse.ArgumentParser(description="Focused best-result tuning for BiLSTM, TFT, Hybrid, Informer, and PatchTST.")
    parser.add_argument(
        "--models",
        default="BiLSTM,TFT,Hybrid,Informer,PatchTST",
        help="Comma-separated subset of models to tune.",
    )
    args = parser.parse_args()
    selected = [name.strip() for name in args.models.split(",") if name.strip()]

    summary, summary_path = run_project_best_track(CONFIG, selected)
    print("\nProject-Best Summary")
    for model_name, payload in summary["results"].items():
        test_metrics = payload["test_metrics"]
        print(
            f"{model_name}: "
            f"MAE={test_metrics['MAE']:.4f}, "
            f"RMSE={test_metrics['RMSE']:.4f}, "
            f"NRMSE={test_metrics['NRMSE']:.4f}, "
            f"UPS={test_metrics['UPS']:.4f}"
        )
    print(f"\nSaved project-best summary to {summary_path}")


if __name__ == "__main__":
    main()
