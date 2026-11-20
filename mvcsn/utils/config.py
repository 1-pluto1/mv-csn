"""Typed configuration dataclasses and YAML loader for MV-CSN v2 & v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class MVCSNConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    gat_heads: int = 2
    num_transformer_layers: int = 2
    transformer_ff_dim: int = 256
    dropout: float = 0.15
    max_len: int = 256
    use_phonetic: bool = True
    use_glyph: bool = True
    use_gate: bool = True
    use_loc_head: bool = True
    use_multimodal_anchor: bool = True
    use_dilated_conv: bool = False
    use_fused_gat: bool = False
    teacher_feat_dim: int = 0


MVCSN_V3_CONFIG = MVCSNConfig(
    hidden_dim=192,
    num_heads=6,
    gat_heads=4,
    num_transformer_layers=3,
    transformer_ff_dim=384,
    dropout=0.15,
    max_len=256,
    use_phonetic=True,
    use_glyph=True,
    use_gate=True,
    use_loc_head=True,
    use_multimodal_anchor=True,
    use_dilated_conv=True,
    use_fused_gat=True,
    teacher_feat_dim=768,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 22
    patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    loc_loss_weight: float = 0.40
    gate_prior_weight: float = 0.10
    use_consistency: bool = True
    cons_weight: float = 0.50
    pert_token_prob: float = 0.20


@dataclass(frozen=True)
class DistillConfig:
    """Balanced hyperparameters for unified dual-teacher cross-view distillation into MV-CSN v3."""

    epochs: int = 22
    patience: int = 12
    lr: float = 1.2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    loc_loss_weight: float = 0.50
    gate_prior_weight: float = 0.25
    cons_weight: float = 0.50
    pert_token_prob: float = 0.20
    unk_prob: float = 0.30
    space_inject_prob: float = 0.25
    temp_domain: float = 2.0
    lambda_logit: float = 0.75
    lambda_feat: float = 1.00
    temp_unlabeled: float = 1.5
    lambda_unlabeled: float = 0.85
    use_logit_kd: bool = True
    use_feat_kd: bool = True
    use_cross_view_kd: bool = True
    use_unlabeled_kd: bool = True
    use_space_aug: bool = True
    use_consistency: bool = True


def load_config(path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
