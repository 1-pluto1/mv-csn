"""MV-CSN v2: Multi-View Context-Gated Character Similarity Network."""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from mvcsn.graph.phonetic import FINALS, INITIALS
from mvcsn.model.encoder import ContextEncoder
from mvcsn.model.gat import ViewGraphAttention
from mvcsn.model.gate import ContextObfuscationGate
from mvcsn.model.heads import AttentionPooling, ObfuscationLocHead, SpamClassifierHead


class MVCSNModel(nn.Module):
    """Dual-view Context-Gated Character Similarity Network with Multimodal Anchor & OOV Bridge."""

    def __init__(
        self,
        vocab_size: int,
        phon_nbr_idx: torch.Tensor,
        phon_nbr_sim: torch.Tensor,
        glyph_nbr_idx: torch.Tensor,
        glyph_nbr_sim: torch.Tensor,
        glyph_z: Optional[torch.Tensor] = None,
        pinyin_ids: Optional[torch.Tensor] = None,
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
        use_multimodal_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.base_vocab_size = vocab_size
        self.use_phonetic = use_phonetic
        self.use_glyph = use_glyph
        self.use_gate = use_gate
        self.use_loc_head = use_loc_head
        self.use_multimodal_anchor = use_multimodal_anchor

        if glyph_z is None:
            glyph_z = torch.zeros((vocab_size, 64), dtype=torch.float32)
        if pinyin_ids is None:
            pinyin_ids = torch.zeros((vocab_size, 3), dtype=torch.long)

        self.register_buffer("phon_nbr_idx", phon_nbr_idx.long())
        self.register_buffer("phon_nbr_sim", phon_nbr_sim.float())
        self.register_buffer("glyph_nbr_idx", glyph_nbr_idx.long())
        self.register_buffer("glyph_nbr_sim", glyph_nbr_sim.float())
        self.register_buffer("glyph_z", glyph_z.float())
        self.register_buffer("pinyin_ids", pinyin_ids.long())
        self._saved_base_buffers: Optional[Dict[str, torch.Tensor]] = None

        self.char_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)

        # ChineseBERT-inspired frozen/projected visual + pinyin character input anchor
        glyph_dim = glyph_z.size(1)
        self.init_emb = nn.Embedding(len(INITIALS) + 2, 16, padding_idx=0)
        self.fin_emb = nn.Embedding(len(FINALS) + 2, 24, padding_idx=0)
        self.tone_emb = nn.Embedding(7, 8, padding_idx=0)
        self.pinyin_proj = nn.Linear(48, hidden_dim)
        self.glyph_proj = nn.Linear(glyph_dim, hidden_dim)
        self.anchor_norm = nn.LayerNorm(hidden_dim)

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

    def set_inference_graph(self, ext_bundle: Dict[str, torch.Tensor]) -> None:
        """Attach SpellGCN-style dynamic OOV graph rows at inference/evaluation time."""
        if self._saved_base_buffers is None:
            self._saved_base_buffers = {
                "phon_nbr_idx": self.phon_nbr_idx.detach().clone(),
                "phon_nbr_sim": self.phon_nbr_sim.detach().clone(),
                "glyph_nbr_idx": self.glyph_nbr_idx.detach().clone(),
                "glyph_nbr_sim": self.glyph_nbr_sim.detach().clone(),
                "glyph_z": self.glyph_z.detach().clone(),
                "pinyin_ids": self.pinyin_ids.detach().clone(),
            }
        dev = self.char_emb.weight.device
        self.phon_nbr_idx = ext_bundle["phon_nbr_idx"].long().to(dev)
        self.phon_nbr_sim = ext_bundle["phon_nbr_sim"].float().to(dev)
        self.glyph_nbr_idx = ext_bundle["glyph_nbr_idx"].long().to(dev)
        self.glyph_nbr_sim = ext_bundle["glyph_nbr_sim"].float().to(dev)
        self.glyph_z = ext_bundle["glyph_z"].float().to(dev)
        self.pinyin_ids = ext_bundle["pinyin_ids"].long().to(dev)

    def reset_inference_graph(self) -> None:
        if self._saved_base_buffers is not None:
            dev = self.char_emb.weight.device
            for k, v in self._saved_base_buffers.items():
                setattr(self, k, v.to(dev))
            self._saved_base_buffers = None

    def encode_chars(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extended OOV indices >= base_vocab_size map to <UNK>=1 for raw char_emb,
        # while retaining full OOV multimodal anchors & graph neighbor lookups.
        base_ids = torch.where(input_ids < self.base_vocab_size, input_ids, torch.ones_like(input_ids))
        e_raw = self.char_emb(base_ids)

        if self.use_multimodal_anchor:
            gz = self.glyph_z[input_ids]
            py = self.pinyin_ids[input_ids]
            py_cat = torch.cat([
                self.init_emb(py[..., 0]),
                self.fin_emb(py[..., 1]),
                self.tone_emb(py[..., 2]),
            ], dim=-1)
            e_char = self.anchor_norm(e_raw + self.emb_drop(self.glyph_proj(gz) + self.pinyin_proj(py_cat)))
        else:
            e_char = self.emb_drop(e_raw)

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
            "pooled": pooled,
            "h_fused": feats["h_fused"],
        }
