"""Context-Aware Obfuscation Gate dynamically weighting context vs. graph views."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextObfuscationGate(nn.Module):
    """Learns per-token simplex weights alpha in Delta^3 over [h_ctx, h_phon, h_glyph]."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        e_char: torch.Tensor,
        h_ctx: torch.Tensor,
        h_phon: torch.Tensor,
        h_glyph: torch.Tensor,
        use_phonetic: bool = True,
        use_glyph: bool = True,
        use_gate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = h_ctx.shape
        if not use_gate:
            weights = [1.0, 1.0 if use_phonetic else 0.0, 1.0 if use_glyph else 0.0]
            alpha = torch.tensor(weights, device=h_ctx.device, dtype=h_ctx.dtype)
            alpha = (alpha / alpha.sum()).view(1, 1, 3).expand(B, L, 3)
        else:
            feat = torch.cat([h_ctx, h_phon, h_glyph, h_ctx - e_char], dim=-1)
            logits = self.mlp(feat)
            if not use_phonetic:
                logits = logits.masked_fill(
                    torch.tensor([False, True, False], device=logits.device).view(1, 1, 3), -1e4
                )
            if not use_glyph:
                logits = logits.masked_fill(
                    torch.tensor([False, False, True], device=logits.device).view(1, 1, 3), -1e4
                )
            alpha = F.softmax(logits, dim=-1)

        fused = (
            alpha[..., 0:1] * h_ctx
            + alpha[..., 1:2] * h_phon
            + alpha[..., 2:3] * h_glyph
        )
        return fused, alpha
