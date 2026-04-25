from __future__ import annotations

from pathlib import Path

from models.bilstm import EnhancedBiLSTM
from models.hybrid import AdaptiveHybridModel
from models.informer import InformerForecastModel
from models.patchtst import PatchTSTForecastModel
from models.transformer import TemporalFusionTransformer
from utils.training import train_model


def checkpoint_name_for_model(name: str) -> str:
    aliases = {
        "AdaptiveHybrid": "hybrid",
        "Hybrid": "hybrid",
        "PatchTST": "patchtst",
    }
    return aliases.get(name, name.lower())


def build_models(input_dim: int, config):
    output_dim = len(config.target_columns)
    return {
        "BiLSTM": EnhancedBiLSTM(
            input_dim=input_dim,
            hidden_dim=config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            dropout=config.dropout,
            output_dim=output_dim,
        ),
        "TFT": TemporalFusionTransformer(
            input_dim=input_dim,
            hidden_dim=config.tft_hidden_dim,
            nhead=config.tft_heads,
            num_layers=config.tft_layers,
            dim_feedforward=config.tft_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        ),
        "Hybrid": AdaptiveHybridModel(input_dim=input_dim, config=config),
        "Informer": InformerForecastModel(
            input_dim=input_dim,
            d_model=config.informer_d_model,
            nhead=config.informer_heads,
            num_layers=config.informer_layers,
            dim_feedforward=config.informer_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        ),
        "PatchTST": PatchTSTForecastModel(
            input_dim=input_dim,
            d_model=config.patchtst_d_model,
            nhead=config.patchtst_heads,
            num_layers=config.patchtst_layers,
            dim_feedforward=config.patchtst_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
            patch_len=config.patchtst_patch_len,
            stride=config.patchtst_stride,
        ),
    }


def train_selected_models(models, datasets, config, model_names: list[str] | None = None):
    train_groups = datasets["train_tpt"]
    val_groups = datasets["val_tpt"]
    x_train, y_train = (
        {key: train_groups[key] for key in ("closeness", "period", "trend")},
        train_groups["target"],
    )
    x_val, y_val = (
        {key: val_groups[key] for key in ("closeness", "period", "trend")},
        val_groups["target"],
    )
    selected_names = model_names or list(models.keys())
    results = {}
    for name in selected_names:
        model = models[name]
        results[name] = train_model(
            model=model,
            model_name=checkpoint_name_for_model(name),
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )
    return results


def train_all_models(datasets, config):
    train_groups = datasets["train_tpt"]
    input_dim = train_groups["closeness"].shape[-1]
    models = build_models(input_dim, config)
    results = train_selected_models(models, datasets, config)
    return models, results
