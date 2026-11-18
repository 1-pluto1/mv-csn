"""Attention pooling, sequence spam classifier, and character obfuscation locator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(h).squeeze(-1)
        scores = scores.masked_fill(~attn_mask, -1e4)
        w = F.softmax(scores, dim=-1)
        w = w * attn_mask.float()
        pooled = torch.bmm(w.unsqueeze(1), h).squeeze(1)
        return pooled, w


class SpamClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class ObfuscationLocHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_fused: torch.Tensor, h_ctx: torch.Tensor, e_char: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_fused, h_ctx - e_char], dim=-1)).squeeze(-1)
