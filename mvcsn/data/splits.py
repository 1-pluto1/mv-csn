"""Leakage-free template-cluster deduplicated dataset splits and attack tagging."""

from __future__ import annotations

import difflib
import gzip
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit

from mvcsn.data.preprocessing import PURE_RADICALS, TextPreprocessor, is_cjk

CANON_STRIP_RE = re.compile(r"https?://\S+|www\.\S+|[0-9]+|\W+")


@dataclass
class SpamRecord:
    idx: int
    label: int
    raw_text: str
    proc_text: str
    clean_text: str
    cluster_id: int
    split: str
    attack_type: str
    obf_mask: List[int]


def _is_kana_or_rare(ch: str) -> bool:
    code = ord(ch)
    if 0x3040 <= code <= 0x30FF:
        return True
    if ch in ("髙", "穏", "茺", "Ꝗ", "⑧", "⑥", "③", "え"):
        return True
    return False


def infer_attack_type(raw_text: str, clean_text: str, label: int, preproc: TextPreprocessor) -> str:
    """Offline heuristic categorization for per-attack evaluation breakdowns."""
    if label == 0:
        return "CLEAN"
    norm_raw = preproc.normalize_noise(raw_text)
    has_space_noise = (len(raw_text) - len(norm_raw)) >= 3
    recomp = preproc.recompose_splits(norm_raw)
    has_split = recomp != norm_raw or any(c in PURE_RADICALS for c in raw_text)

    if recomp.strip() == clean_text.strip() and not has_space_noise and not has_split:
        return "CLEAN"

    has_glyph = any(_is_kana_or_rare(c) for c in raw_text)
    sm = difflib.SequenceMatcher(None, recomp, clean_text, autojunk=False)
    has_phon = False
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            seg_r = recomp[i1:i2]
            seg_c = clean_text[j1:j2]
            if any(is_cjk(c) for c in seg_r) and any(is_cjk(c) for c in seg_c):
                has_phon = True
            if any((not c.isascii() and not is_cjk(c)) or (c in "01" and any(ch.isalpha() for ch in seg_c)) for c in seg_r):
                has_glyph = True

    active = [name for name, flag in [("PHON", has_phon), ("GLYPH", has_glyph), ("SPLIT", has_split), ("SPACE", has_space_noise)] if flag]
    if len(active) >= 2:
        return "MIXED"
    if len(active) == 1:
        return active[0]
    return "PHON" if recomp != clean_text else "CLEAN"


def compute_token_obfuscation_mask(proc_text: str, clean_text: str, label: int, max_len: int = 192) -> List[int]:
    """Align processed variant text against clean reference to mark perturbed positions."""
    n = min(len(proc_text), max_len)
    if label == 0 or not clean_text or proc_text == clean_text:
        return [0] * n
    seq_p = proc_text[:max_len]
    seq_c = re.sub(r"\s+", "", clean_text)
    sm = difflib.SequenceMatcher(None, seq_p, seq_c, autojunk=False)
    mask = [0] * n
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            for k in range(i1, min(i2, n)):
                if seq_p[k] not in ",.!?;:'\"，。！？；：":
                    mask[k] = 1
    return mask


def canonical_for_clustering(proc_text: str) -> str:
    """Deterministic canonical signature preserving CJK + lowercase letters."""
    s = CANON_STRIP_RE.sub("", proc_text.lower())
    return s if len(s) >= 3 else proc_text.lower()


def create_template_cluster_splits(
    dataset_path: str | Path = "data/resources/variant_spam_16k.tsv.gz",
    chaizi_path: str | Path = "data/resources/chaizi_map.json.gz",
    sim_threshold: float = 0.65,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    max_len: int = 192,
    seed: int = 42,
    use_preprocessor: bool = True,
) -> Tuple[List[SpamRecord], Dict[str, object]]:
    """Load dataset, cluster near-duplicate templates, and split without cluster leakage."""
    preproc = TextPreprocessor(chaizi_path)
    rows: List[Tuple[int, str, str, str]] = []
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            lbl = int(parts[0])
            raw_t = parts[1]
            clean_t = parts[2] if len(parts) >= 3 else raw_t
            proc_t = preproc.process(raw_t, enabled=use_preprocessor)
            if not proc_t:
                proc_t = raw_t
            rows.append((lbl, raw_t, proc_t, clean_t))

    N = len(rows)
    canon_texts = [canonical_for_clustering(r[3] or r[2]) for r in rows]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2)
    X = vec.fit_transform(canon_texts)
    sim = (X @ X.T).tocsr()
    adj = (sim >= sim_threshold).tolil()

    # Explicitly link identical proc_text, raw_text, or canon_texts even if short/unique
    for key_fn in (lambda r: r[3] or r[2], lambda r: r[2], lambda r: r[1]):
        seen: Dict[str, int] = {}
        for i, r in enumerate(rows):
            k = key_fn(r)
            if k in seen:
                j = seen[k]
                adj[i, j] = 1
                adj[j, i] = 1
            else:
                seen[k] = i

    n_clusters, cluster_ids = connected_components(adj.tocsr(), directed=False, return_labels=True)

    cluster_members: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        cluster_members.setdefault(int(cid), []).append(idx)

    cids = np.array(sorted(cluster_members.keys()))
    c_labels = np.array([1 if sum(rows[i][0] for i in cluster_members[c]) * 2 >= len(cluster_members[c]) else 0 for c in cids])

    test_val_ratio = 1.0 - train_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=seed)
    tr_idx, rest_idx = next(sss1.split(cids, c_labels))

    rest_cids = cids[rest_idx]
    rest_labels = c_labels[rest_idx]
    val_frac_of_rest = val_ratio / test_val_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_frac_of_rest, random_state=seed)
    va_rel, te_rel = next(sss2.split(rest_cids, rest_labels))

    cluster_split: Dict[int, str] = {}
    for c in cids[tr_idx]:
        cluster_split[int(c)] = "train"
    for c in rest_cids[va_rel]:
        cluster_split[int(c)] = "val"
    for c in rest_cids[te_rel]:
        cluster_split[int(c)] = "test"

    records: List[SpamRecord] = []
    split_counts: Dict[str, Dict[str, int]] = {"train": {"0": 0, "1": 0}, "val": {"0": 0, "1": 0}, "test": {"0": 0, "1": 0}}
    attack_counts: Dict[str, int] = {}

    for idx, (lbl, raw_t, proc_t, clean_t) in enumerate(rows):
        cid = int(cluster_ids[idx])
        sp = cluster_split[cid]
        atk = infer_attack_type(raw_t, clean_t, lbl, preproc)
        obf_mask = compute_token_obfuscation_mask(proc_t, clean_t, lbl, max_len=max_len)
        records.append(
            SpamRecord(
                idx=idx,
                label=lbl,
                raw_text=raw_t,
                proc_text=proc_t,
                clean_text=clean_t,
                cluster_id=cid,
                split=sp,
                attack_type=atk,
                obf_mask=obf_mask,
            )
        )
        split_counts[sp][str(lbl)] += 1
        if sp == "test":
            attack_counts[atk] = attack_counts.get(atk, 0) + 1

    meta = {
        "total_rows": len(records),
        "num_clusters": int(n_clusters),
        "split_counts": split_counts,
        "test_attack_counts": attack_counts,
    }
    return records, meta


def build_or_load_splits(
    cache_path: str | Path = "artifacts/splits.json.gz",
    force_rebuild: bool = False,
    **kwargs,
) -> Tuple[List[SpamRecord], Dict[str, object]]:
    p = Path(cache_path)
    if p.exists() and not force_rebuild:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        records = [SpamRecord(**r) for r in data["records"]]
        return records, data["meta"]
    records, meta = create_template_cluster_splits(**kwargs)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump({"records": [asdict(r) for r in records], "meta": meta}, f, ensure_ascii=False)
    return records, meta
