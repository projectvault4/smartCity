from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.common import TemporalAttention, get_temporal_group_inputs


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = F.elu(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        gated = torch.sigmoid(self.gate(hidden)) * hidden
        return self.norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.prescalers = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(num_features)])
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(hidden_dim, hidden_dim, output_dim=hidden_dim, dropout=dropout) for _ in range(num_features)]
        )
        self.weight_network = GatedResidualNetwork(num_features, hidden_dim, output_dim=num_features, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps, _ = x.shape
        flattened = x.reshape(batch_size * steps, self.num_features)
        raw_weights = self.weight_network(flattened)
        weights = torch.softmax(raw_weights, dim=-1)

        transformed_features = []
        for idx in range(self.num_features):
            single_feature = flattened[:, idx : idx + 1]
            feature_hidden = self.prescalers[idx](single_feature)
            transformed_features.append(self.feature_grns[idx](feature_hidden))

        transformed = torch.stack(transformed_features, dim=1)
        selected = torch.sum(weights.unsqueeze(-1) * transformed, dim=1)
        selected = selected.reshape(batch_size, steps, self.hidden_dim)
        weights = weights.reshape(batch_size, steps, self.num_features)
        return selected, weights


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        self.group_names = ("trend", "period", "closeness")
        self.hidden_dim = hidden_dim
        self.variable_selection = nn.ModuleDict(
            {name: VariableSelectionNetwork(input_dim, hidden_dim, dropout) for name in self.group_names}
        )
        self.sequence_encoder = nn.ModuleDict(
            {
                name: nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )
                for name in self.group_names
            }
        )
        self.temporal_blocks = nn.ModuleDict()
        self.temporal_attention = nn.ModuleDict()
        self.post_attention = nn.ModuleDict()
        for name in self.group_names:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal_blocks[name] = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.temporal_attention[name] = TemporalAttention(hidden_dim)
            self.post_attention[name] = GatedResidualNetwork(hidden_dim, hidden_dim, output_dim=hidden_dim, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.group_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(self.group_names), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, output_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, output_dim), output_dim),
        )
        self.latest_feature_weights = None
        self.latest_temporal_weights = None

    def encode(self, x):
        grouped_inputs = get_temporal_group_inputs(x)
        contexts = []
        feature_weights = []
        temporal_weights = []

        for name in self.group_names:
            selected_inputs, current_feature_weights = self.variable_selection[name](grouped_inputs[name])
            encoded_sequence, _ = self.sequence_encoder[name](selected_inputs)
            temporal_features = self.temporal_blocks[name](encoded_sequence)
            temporal_features = self.post_attention[name](temporal_features)
            context, current_temporal_weights = self.temporal_attention[name](temporal_features)
            contexts.append(context)
            feature_weights.append(current_feature_weights)
            temporal_weights.append(current_temporal_weights)

        fused_context = torch.cat(contexts, dim=-1)
        fused_context = self.group_fusion(fused_context)
        fused_context = self.dropout(fused_context)
        self.latest_feature_weights = torch.stack([weights.mean(dim=1) for weights in feature_weights]).detach().mean(dim=(0, 1)).cpu().numpy()
        self.latest_temporal_weights = {
            name: weights.detach().mean(dim=0).cpu().numpy()
            for name, weights in zip(self.group_names, temporal_weights)
        }
        return fused_context

    def forward(self, x):
        context = self.encode(x)
        return self.regressor(context)


# Backward-compatible alias for older imports.
AdvancedTimeSeriesTransformer = TemporalFusionTransformer
