from __future__ import annotations

import torch
from torch import nn

from models.common import PositionalEncoding, combine_temporal_groups


class _InformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.distill = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=max(1, d_model // 8)),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        if encoded.size(1) <= 2:
            return encoded
        distilled = self.distill(encoded.transpose(1, 2)).transpose(1, 2)
        return distilled


class InformerForecastModel(nn.Module):
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
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        self.pre_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.blocks = nn.ModuleList(
            [
                _InformerEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_dim),
        )

    def forward(self, x):
        sequence = combine_temporal_groups(x)
        hidden = self.position(self.input_projection(sequence))
        if hidden.size(1) >= 16:
            hidden = self.pre_pool(hidden.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        context = self.norm(hidden.mean(dim=1))
        return self.head(self.dropout(context))
