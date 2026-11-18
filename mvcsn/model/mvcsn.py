"""MV-CSN: Multi-View Context-Gated Character Similarity Network."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from mvcsn.model.encoder import ContextEncoder
from mvcsn.model.gat import ViewGraphAttention
from mvcsn.model.gate import ContextObfuscationGate
from mvcsn.model.heads import AttentionPooling, ObfuscationLocHead, SpamClassifierHead


class MVCSNModel(nn.Module):
    """Dual-view (Phonetic + Rendered-Glyph) Context-Gated Character Similarity Network."""

    def __init__(
        self,
        vocab_size: int,
        phon_nbr_idx: torch.Tensor,
        phon_nbr_sim: torch.Tensor,
        glyph_nbr_idx: torch.Tensor,
        glyph_nbr_sim: torch.Tensor,
        hidden_dim: int = 128,
        num_heads: int = 4,
        gat_heads: int = 2,
        num_transformer_layers: int = 2,
        transformer_ff_dim: int = 256,
        dropout: float = 0.15,
        max_len: int = 256,
        use_phonetic: bool = True,
        use_glyph: bool = True,
        use_gate: bool = True,
        use_loc_head: bool = True,
    ) -> None:
        super().__init__()
        self.use_phonetic = use_phonetic
        self.use_glyph = use_glyph
        self.use_gate = use_gate
        self.use_loc_head = use_loc_head

        self.register_buffer("phon_nbr_idx", phon_nbr_idx.long())
        self.register_buffer("phon_nbr_sim", phon_nbr_sim.float())
        self.register_buffer("glyph_nbr_idx", glyph_nbr_idx.long())
        self.register_buffer("glyph_nbr_sim", glyph_nbr_sim.float())

        self.char_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.encoder = ContextEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.phon_gat = ViewGraphAttention(hidden_dim=hidden_dim, num_heads=gat_heads, dropout=dropout)
        self.glyph_gat = ViewGraphAttention(hidden_dim=hidden_dim, num_heads=gat_heads, dropout=dropout)
        self.gate = ContextObfuscationGate(hidden_dim=hidden_dim, dropout=dropout)
        self.pool = AttentionPooling(hidden_dim=hidden_dim)
        self.classifier = SpamClassifierHead(hidden_dim=hidden_dim, dropout=dropout)
        self.loc_head = ObfuscationLocHead(hidden_dim=hidden_dim, dropout=dropout)

    def encode_chars(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        e_char = self.emb_drop(self.char_emb(input_ids))
        h_ctx = self.encoder(e_char, attn_mask)

        if self.use_phonetic:
            h_phon = self.phon_gat(h_ctx, input_ids, self.char_emb.weight, self.phon_nbr_idx, self.phon_nbr_sim)
        else:
            h_phon = torch.zeros_like(h_ctx)

        if self.use_glyph:
            h_glyph = self.glyph_gat(h_ctx, input_ids, self.char_emb.weight, self.glyph_nbr_idx, self.glyph_nbr_sim)
        else:
            h_glyph = torch.zeros_like(h_ctx)

        h_fused, alpha = self.gate(
            e_char=e_char,
            h_ctx=h_ctx,
            h_phon=h_phon,
            h_glyph=h_glyph,
            use_phonetic=self.use_phonetic,
            use_glyph=self.use_glyph,
            use_gate=self.use_gate,
        )
        return {
            "e_char": e_char,
            "h_ctx": h_ctx,
            "h_phon": h_phon,
            "h_glyph": h_glyph,
            "h_fused": h_fused,
            "alpha": alpha,
        }

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encode_chars(input_ids, attn_mask)
        pooled, pool_w = self.pool(feats["h_fused"], attn_mask)
        spam_logit = self.classifier(pooled)
        obf_logits = self.loc_head(feats["h_fused"], feats["h_ctx"], feats["e_char"])
        return {
            "spam_logit": spam_logit,
            "obf_logits": obf_logits,
            "alpha": feats["alpha"],
            "pool_weights": pool_w,
            "h_fused": feats["h_fused"],
        }
