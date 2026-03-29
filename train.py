from __future__ import annotations

from pathlib import Path

from models.bilstm import EnhancedBiLSTM
from models.hybrid import AdaptiveHybridModel
from models.transformer import AdvancedTimeSeriesTransformer
from utils.training import train_model


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
        "Transformer": AdvancedTimeSeriesTransformer(
            input_dim=input_dim,
            d_model=config.transformer_d_model,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        ),
        "Hybrid": AdaptiveHybridModel(input_dim=input_dim, config=config),
    }


def train_all_models(datasets, config):
    x_train, y_train = datasets["train_seq"]
    x_val, y_val = datasets["val_seq"]
    input_dim = x_train.shape[-1]
    models = build_models(input_dim, config)

    results = {}
    for name, model in models.items():
        results[name] = train_model(
            model=model,
            model_name=name.lower(),
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            config=config,
            checkpoint_dir=Path(config.checkpoint_dir),
        )
    return models, results
