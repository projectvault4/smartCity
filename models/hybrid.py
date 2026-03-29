from __future__ import annotations

import torch
from torch import nn

from models.bilstm import EnhancedBiLSTM
from models.transformer import AdvancedTimeSeriesTransformer


class AdaptiveHybridModel(nn.Module):
    def __init__(self, input_dim: int, config):
        super().__init__()
        output_dim = len(config.target_columns)
        self.bilstm_branch = EnhancedBiLSTM(
            input_dim=input_dim,
            hidden_dim=config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            dropout=config.dropout,
            output_dim=output_dim,
        )
        self.transformer_branch = AdvancedTimeSeriesTransformer(
            input_dim=input_dim,
            d_model=config.transformer_d_model,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        )
        fusion_dim = config.bilstm_hidden_dim * 2 + config.transformer_d_model
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.dense_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_hidden_dim * 2, config.dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_hidden_dim, output_dim),
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        bilstm_context = self.bilstm_branch.encode(x)
        transformer_context = self.transformer_branch.encode(x)
        fused = torch.cat([bilstm_context, transformer_context], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)
        return self.fusion(fused)

    @property
    def latest_feature_weights(self):
        bilstm_weights = self.bilstm_branch.latest_feature_weights
        transformer_weights = self.transformer_branch.latest_feature_weights
        if bilstm_weights is None or transformer_weights is None:
            return None
        return (bilstm_weights + transformer_weights) / 2.0
