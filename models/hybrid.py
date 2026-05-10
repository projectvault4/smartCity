from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.bilstm import EnhancedBiLSTM
from models.common import combine_temporal_groups
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
        expert_dim = config.dense_hidden_dim
        self.context_norm = nn.LayerNorm(fusion_dim)
        self.context_dropout = nn.Dropout(config.dropout)
        self.bilstm_adapter = nn.Sequential(
            nn.Linear(config.bilstm_hidden_dim * 2, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(expert_dim),
        )
        self.tft_adapter = nn.Sequential(
            nn.Linear(config.tft_hidden_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(expert_dim),
        )
        router_input_dim = expert_dim * 4 + output_dim * 3
        self.fused_expert = nn.Sequential(
            nn.Linear(expert_dim * 4, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(expert_dim, output_dim),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(router_input_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(expert_dim, output_dim * 3),
        )
        self.fusion_refiner = nn.Sequential(
            nn.Linear(expert_dim * 4 + output_dim * 4, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(expert_dim, output_dim),
        )
        self.residual_scale = nn.Parameter(torch.full((output_dim,), 0.1))
        final_gate = self.gate_network[-1]
        nn.init.zeros_(final_gate.weight)
        with torch.no_grad():
            final_gate.bias.view(output_dim, 3)[:, 0] = 2.0
            final_gate.bias.view(output_dim, 3)[:, 1:] = -1.0
        self.latest_feature_weights = None
        self.latest_gate_values = None
        self._last_bilstm_pred = None
        self._last_tft_pred = None
        self._last_fused_expert_pred = None

    def forward(self, x):
        bilstm_context = self.bilstm_branch.encode(x)
        tft_context = self.tft_branch.encode(x)

        bilstm_pred = self.bilstm_branch.regressor(bilstm_context)
        tft_pred = self.tft_branch.regressor(tft_context)

        fused_context = torch.cat([bilstm_context, tft_context], dim=-1)
        fused_context = self.context_norm(fused_context)
        fused_context = self.context_dropout(fused_context)

        bilstm_feature = self.bilstm_adapter(bilstm_context)
        tft_feature = self.tft_adapter(tft_context)
        interaction_feature = bilstm_feature * tft_feature
        disagreement_feature = torch.abs(bilstm_feature - tft_feature)
        expert_context = torch.cat(
            [bilstm_feature, tft_feature, interaction_feature, disagreement_feature],
            dim=-1,
        )

        fused_expert_pred = self.fused_expert(expert_context)
        routing_features = torch.cat(
            [expert_context, bilstm_pred, tft_pred, torch.abs(bilstm_pred - tft_pred)],
            dim=-1,
        )
        gate_logits = self.gate_network(routing_features).view(-1, bilstm_pred.shape[-1], 3)
        gate = torch.softmax(gate_logits, dim=-1)

        expert_predictions = torch.stack([bilstm_pred, tft_pred, fused_expert_pred], dim=-1)
        mixture_prediction = torch.sum(gate * expert_predictions, dim=-1)
        residual_correction = self.fusion_refiner(
            torch.cat(
                [expert_context, mixture_prediction, bilstm_pred, tft_pred, fused_expert_pred],
                dim=-1,
            )
        )
        final_prediction = mixture_prediction + self.residual_scale * residual_correction

        self.latest_gate_values = gate.detach().mean(dim=0).cpu().numpy()
        self._last_bilstm_pred = bilstm_pred
        self._last_tft_pred = tft_pred
        self._last_fused_expert_pred = fused_expert_pred

        bilstm_weights = self.bilstm_branch.latest_feature_weights
        tft_weights = self.tft_branch.latest_feature_weights
        if bilstm_weights is not None and tft_weights is not None:
            self.latest_feature_weights = (bilstm_weights + tft_weights) / 2.0

        return final_prediction

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        main_loss = 0.6 * F.mse_loss(preds, targets) + 0.4 * F.smooth_l1_loss(preds, targets, beta=0.25)
        aux_weight = 0.15
        aux_losses = []
        for branch_pred in (self._last_bilstm_pred, self._last_tft_pred, self._last_fused_expert_pred):
            if branch_pred is None:
                continue
            aux_losses.append(0.5 * F.mse_loss(branch_pred, targets) + 0.5 * F.smooth_l1_loss(branch_pred, targets, beta=0.25))
        if aux_losses:
            main_loss = main_loss + aux_weight * torch.stack(aux_losses).mean()
        return main_loss


class TFTGRUResidualHybrid(nn.Module):
    """TFT + GRU hybrid with target-wise residual correction."""

    def __init__(self, input_dim: int, config):
        super().__init__()
        output_dim = len(config.target_columns)
        self.tft_branch = TemporalFusionTransformer(
            input_dim=input_dim,
            hidden_dim=config.tft_hidden_dim,
            nhead=config.tft_heads,
            num_layers=config.tft_layers,
            dim_feedforward=config.tft_ff_dim,
            dropout=config.dropout,
            output_dim=output_dim,
        )
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            batch_first=True,
            dropout=config.dropout if config.bilstm_layers > 1 else 0.0,
        )
        expert_dim = config.dense_hidden_dim
        self.gru_adapter = nn.Sequential(
            nn.Linear(config.bilstm_hidden_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(expert_dim),
        )
        self.tft_adapter = nn.Sequential(
            nn.Linear(config.tft_hidden_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(expert_dim),
        )
        self.gru_regressor = nn.Sequential(
            nn.Linear(config.bilstm_hidden_dim, max(config.bilstm_hidden_dim // 2, output_dim)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(max(config.bilstm_hidden_dim // 2, output_dim), output_dim),
        )
        fusion_dim = expert_dim * 4 + output_dim * 3
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(expert_dim, output_dim * 3),
        )
        self.fused_expert = nn.Sequential(
            nn.Linear(expert_dim * 4, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(expert_dim, output_dim),
        )
        self.residual_corrector = nn.Sequential(
            nn.Linear(fusion_dim + output_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(expert_dim, output_dim),
        )
        self.residual_scale = nn.Parameter(torch.full((output_dim,), 0.1))
        self.latest_feature_weights = None
        self.latest_gate_values = None
        self._last_tft_pred = None
        self._last_gru_pred = None
        self._last_fused_pred = None

    def forward(self, x):
        tft_context = self.tft_branch.encode(x)
        gru_input = combine_temporal_groups(x)
        gru_output, _ = self.gru(gru_input)
        gru_context = gru_output[:, -1, :]

        tft_pred = self.tft_branch.regressor(tft_context)
        gru_pred = self.gru_regressor(gru_context)

        tft_feature = self.tft_adapter(tft_context)
        gru_feature = self.gru_adapter(gru_context)
        interaction_feature = tft_feature * gru_feature
        disagreement_feature = torch.abs(tft_feature - gru_feature)
        expert_context = torch.cat(
            [tft_feature, gru_feature, interaction_feature, disagreement_feature],
            dim=-1,
        )

        fused_pred = self.fused_expert(expert_context)
        pred_disagreement = torch.abs(tft_pred - gru_pred)
        routing_features = torch.cat([expert_context, tft_pred, gru_pred, pred_disagreement], dim=-1)
        gate_logits = self.gate_network(routing_features).view(-1, tft_pred.shape[-1], 3)
        gate = torch.softmax(gate_logits, dim=-1)

        expert_predictions = torch.stack([tft_pred, gru_pred, fused_pred], dim=-1)
        mixture_prediction = torch.sum(gate * expert_predictions, dim=-1)
        residual_features = torch.cat([routing_features, mixture_prediction], dim=-1)
        residual = self.residual_corrector(residual_features)
        final_prediction = mixture_prediction + self.residual_scale * residual

        self.latest_feature_weights = self.tft_branch.latest_feature_weights
        self.latest_gate_values = gate.detach().mean(dim=0).cpu().numpy()
        self._last_tft_pred = tft_pred
        self._last_gru_pred = gru_pred
        self._last_fused_pred = fused_pred
        return final_prediction

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = getattr(self, "target_loss_weights", None)
        if weights is not None:
            weights = weights.to(device=preds.device, dtype=preds.dtype)
            mse_loss = ((preds - targets) ** 2 * weights).mean()
            smooth_loss = (F.smooth_l1_loss(preds, targets, beta=0.25, reduction="none") * weights).mean()
        else:
            mse_loss = F.mse_loss(preds, targets)
            smooth_loss = F.smooth_l1_loss(preds, targets, beta=0.25)
        loss = 0.65 * mse_loss + 0.35 * smooth_loss
        aux_losses = []
        for branch_pred in (self._last_tft_pred, self._last_gru_pred, self._last_fused_pred):
            if branch_pred is not None:
                if weights is not None:
                    aux_losses.append(((branch_pred - targets) ** 2 * weights).mean())
                else:
                    aux_losses.append(F.mse_loss(branch_pred, targets))
        if aux_losses:
            loss = loss + 0.12 * torch.stack(aux_losses).mean()
        return loss
