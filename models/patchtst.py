from __future__ import annotations

import torch
from torch import nn

from models.common import PositionalEncoding, combine_temporal_groups


class PatchTSTForecastModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
        patch_len: int,
        stride: int,
    ):
        super().__init__()
        self.patch_len = max(2, patch_len)
        self.stride = max(1, stride)
        self.patch_projection = nn.Linear(self.patch_len, d_model)
        self.position = PositionalEncoding(d_model)
        self.patch_mixer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.patch_score = nn.Linear(d_model, 1)
        self.channel_fusion = nn.Sequential(
            nn.Linear(input_dim * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
        )
        self.norm = nn.LayerNorm(d_model)

    def _extract_patches(self, series: torch.Tensor) -> torch.Tensor:
        if series.size(-1) < self.patch_len:
            pad_len = self.patch_len - series.size(-1)
            series = nn.functional.pad(series, (0, pad_len), mode="replicate")
        patches = series.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.size(-2) == 0:
            patches = series.unsqueeze(-2)
        return patches

    def forward(self, x):
        sequence = combine_temporal_groups(x)
        channel_first = sequence.transpose(1, 2)
        patches = self._extract_patches(channel_first)
        batch_size, channels, patch_count, patch_len = patches.shape
        tokens = patches.reshape(batch_size * channels, patch_count, patch_len)
        hidden = self.position(self.patch_projection(tokens))
        mixed = self.patch_mixer(hidden)
        scores = torch.softmax(self.patch_score(mixed).squeeze(-1), dim=1)
        pooled = self.norm(torch.sum(mixed * scores.unsqueeze(-1), dim=1)).reshape(batch_size, channels, -1)
        fused = pooled.reshape(batch_size, -1)
        return self.channel_fusion(fused)
