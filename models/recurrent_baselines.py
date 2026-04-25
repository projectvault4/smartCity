from __future__ import annotations

import torch
from torch import nn


class _RecurrentForecastBase(nn.Module):
    recurrent_cls = nn.RNN

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = self.recurrent_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.recurrent(x)
        final_state = output[:, -1, :]
        return self.head(self.dropout(final_state))


class PlainLSTM(_RecurrentForecastBase):
    recurrent_cls = nn.LSTM


class PlainGRU(_RecurrentForecastBase):
    recurrent_cls = nn.GRU
