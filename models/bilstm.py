from __future__ import annotations

import torch
from torch import nn

from models.common import FeatureAttention, TemporalAttention, get_temporal_group_inputs


class EnhancedBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int):
        super().__init__()
        self.group_names = ("trend", "period", "closeness")
        self.feature_attention = nn.ModuleDict({name: FeatureAttention(input_dim) for name in self.group_names})
        self.lstm = nn.ModuleDict(
            {
                name: nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=True,
                )
                for name in self.group_names
            }
        )
        self.temporal_attention = nn.ModuleDict(
            {name: TemporalAttention(hidden_dim * 2) for name in self.group_names}
        )
        self.dropout = nn.Dropout(dropout)
        self.group_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 * len(self.group_names), hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.latest_feature_weights = None
        self.latest_temporal_weights = None

    def encode(self, x):
        grouped_inputs = get_temporal_group_inputs(x)
        contexts = []
        feature_weights = []
        temporal_weights = []
        for name in self.group_names:
            attended_x, current_feature_weights = self.feature_attention[name](grouped_inputs[name])
            outputs, _ = self.lstm[name](attended_x)
            context, current_temporal_weights = self.temporal_attention[name](outputs)
            contexts.append(context)
            feature_weights.append(current_feature_weights)
            temporal_weights.append(current_temporal_weights)

        fused_context = torch.cat(contexts, dim=-1)
        fused_context = self.group_fusion(fused_context)
        fused_context = self.dropout(fused_context)
        self.latest_feature_weights = torch.stack(feature_weights).detach().mean(dim=(0, 1)).cpu().numpy()
        self.latest_temporal_weights = {
            name: weights.detach().mean(dim=0).cpu().numpy()
            for name, weights in zip(self.group_names, temporal_weights)
        }
        return fused_context

    def forward(self, x):
        context = self.encode(x)
        return self.regressor(context)
