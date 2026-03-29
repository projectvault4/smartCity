from __future__ import annotations

import math

import torch
from torch import nn


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_hidden = torch.tanh(self.proj(x))
        scores = self.score(attn_hidden).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class FeatureAttention(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_features)

    def forward(self, x):
        pooled = x.mean(dim=1)
        scores = self.fc2(torch.relu(self.fc1(pooled)))
        weights = torch.softmax(scores, dim=-1)
        return x * weights.unsqueeze(1), weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]
