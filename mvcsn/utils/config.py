"""Typed configuration dataclasses and YAML loader for MV-CSN v2.

Centralizing architecture, training, and inference hyperparameters in strongly
typed dataclasses prevents scattered magic numbers across training/evaluation
scripts while remaining 100% compatible with legacy keyword-argument callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class MVCSNConfig:
    """Architecture hyperparameters and ablation switches for :class:`MVCSNModel`.

    Attributes:
        hidden_dim: Shared token/graph hidden dimension ``D`` (default: 128).
        num_heads: Attention heads in the sequence Transformer encoder.
        gat_heads: Attention heads in each 1-hop graph attention view.
        num_transformer_layers: Depth of the sequence context Transformer.
        transformer_ff_dim: Feed-forward inner dimension inside Transformer layers.
        dropout: Dropout probability shared across embeddings, GAT, and heads.
        max_len: Maximum positional-encoding sequence length.
        use_phonetic: Enable View 1 (Phonetic Top-K Graph Attention).
        use_glyph: Enable View 2 (Rendered-Glyph CAE Top-K Graph Attention).
        use_gate: Enable learned per-token Context Obfuscation Gate (False => uniform).
        use_loc_head: Enable auxiliary token-level obfuscation localization head.
        use_multimodal_anchor: Enable ChineseBERT-style glyph + pinyin input anchor.
    """

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


@dataclass(frozen=True)
class TrainConfig:
    """Optimization and multi-task loss hyperparameters for :func:`train_mvcsn`.

    Attributes:
        epochs: Maximum number of training epochs.
        patience: Early-stopping patience measured on validation F1.
        lr: Initial learning rate for AdamW with cosine annealing.
        weight_decay: L2 weight decay coefficient.
        grad_clip: Global gradient norm clipping threshold.
        loc_loss_weight: Weight of token-level obfuscation localization BCE loss.
        gate_prior_weight: Weight of clean/obfuscated gate prior regularization.
        use_consistency: Enable GPU-vectorized RoCBert graph consistency training.
        cons_weight: Weight of prediction + representation consistency loss.
        pert_token_prob: Per-token neighbor/UNK substitution rate for consistency view.
    """

    epochs: int = 20
    patience: int = 6
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    loc_loss_weight: float = 0.40
    gate_prior_weight: float = 0.10
    use_consistency: bool = True
    cons_weight: float = 0.50
    pert_token_prob: float = 0.20


def load_config(path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file into a plain Python dictionary.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed YAML mapping (or empty dict if the file is empty).
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
