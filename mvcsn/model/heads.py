"""Sequence attention pooling, spam classifier head, and obfuscation locator head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Additive attention pooling aggregating token states ``[B, L, D]`` into ``[B, D]``."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, h: torch.Tensor, attn_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool variable-length token states using masked softmax attention weights.

        Args:
            h: Fused token representations, shape ``[B, L, D]``.
            attn_mask: Valid-token boolean mask, shape ``[B, L]``.

        Returns:
            Tuple ``(pooled, weights)`` of shapes ``[B, D]`` and ``[B, L]``.
        """
        scores = self.attn(h).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~attn_mask, -1e4)
        weights = F.softmax(scores, dim=-1) * attn_mask.float()
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # [B, D]
        return pooled, weights


class SpamClassifierHead(nn.Module):
    """Binary sequence-level spam classification MLP returning raw logits ``[B]``."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map pooled message embedding ``[B, D]`` to scalar spam logits ``[B]``."""
        return self.net(z).squeeze(-1)


class ObfuscationLocHead(nn.Module):
    """Auxiliary token-level head predicting per-character obfuscation logits ``[B, L]``.

    Combines the fused representation ``h_fused`` with the context-vs-character
    discrepancy ``h_ctx - e_char`` to localize disguised tokens for both
    multi-task supervision and Gate X-Ray interpretability.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h_fused: torch.Tensor,
        h_ctx: torch.Tensor,
        e_char: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-token obfuscation logits ``[B, L]``."""
        features = torch.cat([h_fused, h_ctx - e_char], dim=-1)
        return self.net(features).squeeze(-1)
