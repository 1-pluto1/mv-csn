"""MV-CSN v2 & v3: Multi-View Context-Gated Character Similarity Network."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvcsn.graph.phonetic import FINALS, INITIALS
from mvcsn.model.encoder import ContextEncoder
from mvcsn.model.gat import FusedDualViewGAT, ViewGraphAttention
from mvcsn.model.gate import ContextObfuscationGate
from mvcsn.model.heads import AttentionPooling, ObfuscationLocHead, SpamClassifierHead
from mvcsn.utils.config import MVCSNConfig

_GRAPH_BUFFER_KEYS = (
    "phon_nbr_idx",
    "phon_nbr_sim",
    "glyph_nbr_idx",
    "glyph_nbr_sim",
    "glyph_z",
    "pinyin_ids",
)


class MVCSNModel(nn.Module):
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
        use_dilated_conv: bool = False,
        use_fused_gat: bool = False,
        teacher_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.base_vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gat_heads = gat_heads
        self.num_transformer_layers = num_transformer_layers
        self.transformer_ff_dim = transformer_ff_dim
        self.dropout_rate = dropout
        self.max_len = max_len
        self.use_phonetic = use_phonetic
        self.use_glyph = use_glyph
        self.use_gate = use_gate
        self.use_loc_head = use_loc_head
        self.use_multimodal_anchor = use_multimodal_anchor
        self.use_dilated_conv = use_dilated_conv
        self.use_fused_gat = use_fused_gat
        self.teacher_feat_dim = teacher_feat_dim

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
        self.register_buffer("phon_log_sim", torch.log(torch.clamp(phon_nbr_sim.float(), min=1e-4)), persistent=False)
        self.register_buffer("phon_has_nbr", phon_nbr_sim.float() > 0.0, persistent=False)
        self.register_buffer("glyph_log_sim", torch.log(torch.clamp(glyph_nbr_sim.float(), min=1e-4)), persistent=False)
        self.register_buffer("glyph_has_nbr", glyph_nbr_sim.float() > 0.0, persistent=False)
        self._saved_base_buffers: Optional[Dict[str, torch.Tensor]] = None

        self.char_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)

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
            use_dilated_conv=use_dilated_conv,
        )
        if use_fused_gat:
            self.fused_gat = FusedDualViewGAT(hidden_dim=hidden_dim, num_heads=gat_heads, dropout=dropout)
            self.phon_gat = None
            self.glyph_gat = None
        else:
            self.fused_gat = None
            self.phon_gat = ViewGraphAttention(hidden_dim=hidden_dim, num_heads=gat_heads, dropout=dropout)
            self.glyph_gat = ViewGraphAttention(hidden_dim=hidden_dim, num_heads=gat_heads, dropout=dropout)

        self.gate = ContextObfuscationGate(hidden_dim=hidden_dim, dropout=dropout)
        self.pool = AttentionPooling(hidden_dim=hidden_dim)
        self.classifier = SpamClassifierHead(hidden_dim=hidden_dim, dropout=dropout)
        self.loc_head = ObfuscationLocHead(hidden_dim=hidden_dim, dropout=dropout)

        self.kd_proj = (
            nn.Sequential(nn.Linear(hidden_dim, teacher_feat_dim), nn.LayerNorm(teacher_feat_dim))
            if teacher_feat_dim > 0
            else None
        )

    def _refresh_graph_derivs(self) -> None:
        self.phon_log_sim = torch.log(torch.clamp(self.phon_nbr_sim, min=1e-4))
        self.phon_has_nbr = self.phon_nbr_sim > 0.0
        self.glyph_log_sim = torch.log(torch.clamp(self.glyph_nbr_sim, min=1e-4))
        self.glyph_has_nbr = self.glyph_nbr_sim > 0.0

    def get_config(self, teacher_feat_dim: Optional[int] = None) -> MVCSNConfig:
        return MVCSNConfig(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            gat_heads=self.gat_heads,
            num_transformer_layers=self.num_transformer_layers,
            transformer_ff_dim=self.transformer_ff_dim,
            dropout=self.dropout_rate,
            max_len=self.max_len,
            use_phonetic=self.use_phonetic,
            use_glyph=self.use_glyph,
            use_gate=self.use_gate,
            use_loc_head=self.use_loc_head,
            use_multimodal_anchor=self.use_multimodal_anchor,
            use_dilated_conv=self.use_dilated_conv,
            use_fused_gat=self.use_fused_gat,
            teacher_feat_dim=self.teacher_feat_dim if teacher_feat_dim is None else teacher_feat_dim,
        )

    def load_pretrained_char_emb(self, init_matrix: torch.Tensor) -> None:
        with torch.no_grad():
            mat = init_matrix.to(device=self.char_emb.weight.device, dtype=self.char_emb.weight.dtype)
            rows = min(self.char_emb.weight.size(0), mat.size(0))
            cols = min(self.char_emb.weight.size(1), mat.size(1))
            self.char_emb.weight[:rows, :cols].copy_(mat[:rows, :cols])
            self.char_emb.weight[0].zero_()

    @classmethod
    def from_config(
        cls,
        vocab_size: int,
        graphs: Dict[str, torch.Tensor],
        config: Optional[MVCSNConfig] = None,
    ) -> "MVCSNModel":
        cfg = config or MVCSNConfig()
        return cls(
            vocab_size=vocab_size,
            phon_nbr_idx=graphs["phon_nbr_idx"],
            phon_nbr_sim=graphs["phon_nbr_sim"],
            glyph_nbr_idx=graphs["glyph_nbr_idx"],
            glyph_nbr_sim=graphs["glyph_nbr_sim"],
            glyph_z=graphs.get("glyph_z"),
            pinyin_ids=graphs.get("pinyin_ids"),
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            gat_heads=cfg.gat_heads,
            num_transformer_layers=cfg.num_transformer_layers,
            transformer_ff_dim=cfg.transformer_ff_dim,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            use_phonetic=cfg.use_phonetic,
            use_glyph=cfg.use_glyph,
            use_gate=cfg.use_gate,
            use_loc_head=cfg.use_loc_head,
            use_multimodal_anchor=cfg.use_multimodal_anchor,
            use_dilated_conv=cfg.use_dilated_conv,
            use_fused_gat=cfg.use_fused_gat,
            teacher_feat_dim=cfg.teacher_feat_dim,
        )

    def set_inference_graph(self, ext_bundle: Dict[str, torch.Tensor]) -> None:
        if self._saved_base_buffers is None:
            self._saved_base_buffers = {
                key: getattr(self, key).detach().clone() for key in _GRAPH_BUFFER_KEYS
            }
        dev = self.char_emb.weight.device
        self.phon_nbr_idx = ext_bundle["phon_nbr_idx"].long().to(dev)
        self.phon_nbr_sim = ext_bundle["phon_nbr_sim"].float().to(dev)
        self.glyph_nbr_idx = ext_bundle["glyph_nbr_idx"].long().to(dev)
        self.glyph_nbr_sim = ext_bundle["glyph_nbr_sim"].float().to(dev)
        self.glyph_z = ext_bundle["glyph_z"].float().to(dev)
        self.pinyin_ids = ext_bundle["pinyin_ids"].long().to(dev)
        self._refresh_graph_derivs()

    def reset_inference_graph(self) -> None:
        if self._saved_base_buffers is not None:
            dev = self.char_emb.weight.device
            for key, tensor in self._saved_base_buffers.items():
                setattr(self, key, tensor.to(dev))
            self._saved_base_buffers = None
            self._refresh_graph_derivs()

    @contextmanager
    def temporary_inference_graph(
        self, ext_bundle: Optional[Dict[str, torch.Tensor]]
    ) -> Iterator[None]:
        if ext_bundle is None:
            yield
            return
        self.set_inference_graph(ext_bundle)
        try:
            yield
        finally:
            self.reset_inference_graph()

    def encode_chars(
        self, input_ids: torch.Tensor, attn_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        base_ids = torch.where(
            input_ids < self.base_vocab_size, input_ids, torch.ones_like(input_ids)
        )
        e_raw = self.char_emb(base_ids)

        if self.use_multimodal_anchor:
            gz = self.glyph_z[input_ids]
            py = self.pinyin_ids[input_ids]
            py_cat = torch.cat(
                [
                    self.init_emb(py[..., 0]),
                    self.fin_emb(py[..., 1]),
                    self.tone_emb(py[..., 2]),
                ],
                dim=-1,
            )
            e_char = self.anchor_norm(
                e_raw + self.emb_drop(self.glyph_proj(gz) + self.pinyin_proj(py_cat))
            )
        else:
            e_char = self.emb_drop(e_raw)

        h_ctx = self.encoder(e_char, attn_mask)

        if self.fused_gat is not None:
            h_phon, h_glyph = self.fused_gat(
                h_ctx=h_ctx,
                input_ids=input_ids,
                emb_weight=self.char_emb.weight,
                phon_nbr_idx=self.phon_nbr_idx,
                phon_log_sim=self.phon_log_sim,
                phon_has_nbr=self.phon_has_nbr,
                glyph_nbr_idx=self.glyph_nbr_idx,
                glyph_log_sim=self.glyph_log_sim,
                glyph_has_nbr=self.glyph_has_nbr,
                use_phonetic=self.use_phonetic,
                use_glyph=self.use_glyph,
            )
        else:
            h_phon = (
                self.phon_gat(h_ctx, input_ids, self.char_emb.weight, self.phon_nbr_idx, self.phon_nbr_sim)
                if self.use_phonetic
                else torch.zeros_like(h_ctx)
            )
            h_glyph = (
                self.glyph_gat(h_ctx, input_ids, self.char_emb.weight, self.glyph_nbr_idx, self.glyph_nbr_sim)
                if self.use_glyph
                else torch.zeros_like(h_ctx)
            )

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

    def forward(
        self, input_ids: torch.Tensor, attn_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        feats = self.encode_chars(input_ids, attn_mask)
        pooled, pool_w = self.pool(feats["h_fused"], attn_mask)
        spam_logit = self.classifier(pooled)
        obf_logits = self.loc_head(feats["h_fused"], feats["h_ctx"], feats["e_char"])
        out: Dict[str, torch.Tensor] = {
            "spam_logit": spam_logit,
            "obf_logits": obf_logits,
            "alpha": feats["alpha"],
            "pool_weights": pool_w,
            "pooled": pooled,
            "h_fused": feats["h_fused"],
        }
        if self.kd_proj is not None:
            out["kd_feat"] = F.normalize(self.kd_proj(pooled).float(), p=2, dim=-1)
        return out
