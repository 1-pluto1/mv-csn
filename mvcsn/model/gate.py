"""Context-Aware Obfuscation Gate dynamically weighting context vs. graph views."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextObfuscationGate(nn.Module):
    """Learns per-token simplex weights ``alpha = [alpha_ctx, alpha_phon, alpha_glyph]``.

    Given the discrepancy ``h_ctx - e_char`` between contextual expectation and
    observed character identity alongside both graph-restored streams, the gate
    assigns near-1.0 weight to ``h_ctx`` on clean characters (avoiding neighbor
    semantic drift) and shifts probability mass onto ``h_phon`` / ``h_glyph``
    whenever a character looks out-of-place in context::

        h_fused = alpha_ctx * h_ctx + alpha_phon * h_phon + alpha_glyph * h_glyph
    """

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
        """Fuse context, phonetic-graph, and glyph-graph token streams.

        Args:
            e_char: Multimodal character input anchor, shape ``[B, L, D]``.
            h_ctx: Contextual sequence representation, shape ``[B, L, D]``.
            h_phon: Phonetic-graph GAT output, shape ``[B, L, D]``.
            h_glyph: Visual-glyph-graph GAT output, shape ``[B, L, D]``.
            use_phonetic: Whether View 1 (phonetic) is active (ablation switch).
            use_glyph: Whether View 2 (glyph) is active (ablation switch).
            use_gate: If ``False``, fall back to uniform averaging over active streams.

        Returns:
            Tuple ``(fused, alpha)`` with shapes ``[B, L, D]`` and ``[B, L, 3]``.
        """
        B, L, _ = h_ctx.shape
        if not use_gate:
            weights = [1.0, 1.0 if use_phonetic else 0.0, 1.0 if use_glyph else 0.0]
            alpha = torch.tensor(weights, device=h_ctx.device, dtype=h_ctx.dtype)
            alpha = (alpha / alpha.sum()).view(1, 1, 3).expand(B, L, 3)
        else:
            # Concatenate streams plus explicit context-vs-character mismatch signal.
            feat = torch.cat([h_ctx, h_phon, h_glyph, h_ctx - e_char], dim=-1)
            logits = self.mlp(feat)
            if not use_phonetic or not use_glyph:
                mask_flags = [False, not use_phonetic, not use_glyph]
                view_mask = torch.tensor(mask_flags, device=logits.device).view(1, 1, 3)
                logits = logits.masked_fill(view_mask, -1e4)
            alpha = F.softmax(logits, dim=-1)

        fused = (
            alpha[..., 0:1] * h_ctx
            + alpha[..., 1:2] * h_phon
            + alpha[..., 2:3] * h_glyph
        )
        return fused, alpha
