from __future__ import annotations

import torch
from torch import nn

from models.common import FeatureAttention, PositionalEncoding, TemporalAttention


class AdvancedTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        self.feature_attention = FeatureAttention(input_dim)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_attention = TemporalAttention(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )
        self.latest_feature_weights = None
        self.latest_temporal_weights = None

    def encode(self, x):
        attended_x, feature_weights = self.feature_attention(x)
        x_proj = self.input_projection(attended_x)
        encoded = self.positional_encoding(x_proj)
        encoded = self.encoder(encoded)
        encoded = self.norm(encoded)
        context, temporal_weights = self.temporal_attention(encoded)
        context = self.dropout(context)
        self.latest_feature_weights = feature_weights.detach().mean(dim=0).cpu().numpy()
        self.latest_temporal_weights = temporal_weights.detach().mean(dim=0).cpu().numpy()
        return context

    def forward(self, x):
        context = self.encode(x)
        return self.regressor(context)
