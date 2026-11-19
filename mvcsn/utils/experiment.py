"""Shared experiment orchestration helpers (DRY layer for all evaluation scripts).

Consolidates common constants (random seeds, ablation matrix) and record/metric
utilities previously duplicated across ``run_all.py``, ``run_ablations.py``,
and ``run_llm_eval.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord

# Standard 3-seed protocol used across all benchmark and ablation suites.
DEFAULT_SEEDS: Tuple[int, ...] = (42, 1024, 2025)


@dataclass(frozen=True)
class AblationSpec:
    """Declarative specification for one row in the 8-row MV-CSN v2 ablation table."""

    name: str
    use_phonetic: bool = True
    use_glyph: bool = True
    use_gate: bool = True
    use_multimodal_anchor: bool = True
    use_consistency: bool = True
    use_oov_bridge: bool = True
    use_preprocessor: bool = True


# Canonical 8-row ablation study matrix evaluated in Exp 4.
ABLATION_SPECS: Tuple[AblationSpec, ...] = (
    AblationSpec("Full MV-CSN v2"),
    AblationSpec("w/o Phonetic View", use_phonetic=False),
    AblationSpec("w/o Glyph Visual View", use_glyph=False),
    AblationSpec("w/o Context Gate (Uniform)", use_gate=False),
    AblationSpec("w/o RoCBert Consistency", use_consistency=False),
    AblationSpec("w/o Multimodal Anchor", use_multimodal_anchor=False),
    AblationSpec("w/o Dynamic OOV Bridge", use_oov_bridge=False),
    AblationSpec("w/o Split/Noise Preprocessor", use_preprocessor=False),
)


def to_clean_record(
    record: SpamRecord,
    preproc: TextPreprocessor,
    max_len: int = 192,
) -> SpamRecord:
    """Convert a variant :class:`SpamRecord` into its clean-text counterpart.

    Used to build clean-only training/validation splits for the Zero-Shot
    ``Clean -> RealVariant`` evaluation protocol without altering labels or
    template-cluster assignments.

    Args:
        record: Original dataset record containing ``clean_text``.
        preproc: Text preprocessor applied consistently to the clean text.
        max_len: Maximum mask length matching dataset truncation.

    Returns:
        A new :class:`SpamRecord` tagged ``CLEAN`` with an all-zero obfuscation mask.
    """
    clean_raw = record.clean_text if record.clean_text else record.raw_text
    clean_proc = preproc.process(clean_raw, enabled=True) or clean_raw
    return SpamRecord(
        idx=record.idx,
        label=record.label,
        raw_text=clean_raw,
        proc_text=clean_proc,
        clean_text=clean_raw,
        cluster_id=record.cluster_id,
        split=record.split,
        attack_type="CLEAN",
        obf_mask=[0] * min(len(clean_proc), max_len),
    )


def aggregate_metric_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate numeric metrics across multiple random seeds into ``{mean, std}``.

    Args:
        runs: Non-empty sequence of per-seed metric dictionaries.

    Returns:
        Mapping from each numeric metric key to ``{"mean": float, "std": float}``.
    """
    if not runs:
        return {}
    numeric_keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    return {
        key: {
            "mean": float(np.mean([float(r[key]) for r in runs])),
            "std": float(np.std([float(r[key]) for r in runs])),
        }
        for key in numeric_keys
    }


def save_json(payload: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialize a JSON payload to UTF-8 disk with automatic parent creation.

    Args:
        payload: JSON-serializable object.
        path: Output file path.
        indent: Pretty-print indentation spaces.

    Returns:
        The resolved :class:`Path` that was written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
    return out


# Backward-compatible alias for scripts that import agg_stats directly.
agg_stats = aggregate_metric_runs
