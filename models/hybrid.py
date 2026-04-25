from __future__ import annotations

import torch
from torch import nn

from models.bilstm import EnhancedBiLSTM
from models.transformer import TemporalFusionTransformer


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
        self.tft_branch = TemporalFusionTransformer(
            input_dim=input_dim,
            hidden_dim=config.tft_hidden_dim,
            nhead=config.tft_heads,
            num_layers=config.tft_layers,
            dim_feedforward=config.tft_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        )
        fusion_dim = config.bilstm_hidden_dim * 2 + config.tft_hidden_dim
        self.context_norm = nn.LayerNorm(fusion_dim)
        self.context_dropout = nn.Dropout(config.dropout)
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim, config.dense_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_hidden_dim, output_dim),
        )
        self.fusion_refiner = nn.Sequential(
            nn.Linear(fusion_dim + output_dim * 2, config.dense_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_hidden_dim, output_dim),
        )
        self.latest_feature_weights = None
        self.latest_gate_values = None

    def forward(self, x):
        bilstm_context = self.bilstm_branch.encode(x)
        tft_context = self.tft_branch.encode(x)

        bilstm_pred = self.bilstm_branch.regressor(bilstm_context)
        tft_pred = self.tft_branch.regressor(tft_context)

        fused_context = torch.cat([bilstm_context, tft_context], dim=-1)
        fused_context = self.context_norm(fused_context)
        fused_context = self.context_dropout(fused_context)

        gate = torch.sigmoid(self.gate_network(fused_context))
        gated_prediction = gate * bilstm_pred + (1.0 - gate) * tft_pred
        residual_correction = self.fusion_refiner(torch.cat([fused_context, bilstm_pred, tft_pred], dim=-1))
        final_prediction = gated_prediction + residual_correction

        self.latest_gate_values = gate.detach().mean(dim=0).cpu().numpy()

        bilstm_weights = self.bilstm_branch.latest_feature_weights
        tft_weights = self.tft_branch.latest_feature_weights
        if bilstm_weights is not None and tft_weights is not None:
            self.latest_feature_weights = (bilstm_weights + tft_weights) / 2.0

        return final_prediction
