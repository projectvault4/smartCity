from __future__ import annotations

import torch
from torch import nn

from models.common import FeatureAttention, TemporalAttention


class EnhancedBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int):
        super().__init__()
        self.feature_attention = FeatureAttention(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.temporal_attention = TemporalAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.latest_feature_weights = None
        self.latest_temporal_weights = None

    def encode(self, x):
        attended_x, feature_weights = self.feature_attention(x)
        outputs, _ = self.lstm(attended_x)
        context, temporal_weights = self.temporal_attention(outputs)
        context = self.dropout(context)
        self.latest_feature_weights = feature_weights.detach().mean(dim=0).cpu().numpy()
        self.latest_temporal_weights = temporal_weights.detach().mean(dim=0).cpu().numpy()
        return context

    def forward(self, x):
        context = self.encode(x)
        return self.regressor(context)
