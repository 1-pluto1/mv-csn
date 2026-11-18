"""Vectorized polyphone-aware soft-phonetic character similarity graph (View 1)."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from pypinyin import Style, pinyin

from mvcsn.data.vocab import CharVocab

INITIALS = [
    "", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
]
INIT_TO_ID = {s: i + 1 for i, s in enumerate(INITIALS)}

FINALS = [
    "", "a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu",
    "ie", "ve", "er", "an", "en", "in", "un", "vn", "ang", "eng", "ing", "ong",
    "ia", "iao", "ian", "iang", "iong", "ua", "uo", "uai", "uan", "uang", "ueng",
]
FINAL_TO_ID = {s: i + 1 for i, s in enumerate(FINALS)}

CONFUSABLE_INITIAL_PAIRS: List[Tuple[str, str, float]] = [
    ("zh", "z", 0.85), ("ch", "c", 0.85), ("sh", "s", 0.85),
    ("n", "l", 0.80), ("r", "l", 0.75), ("f", "h", 0.75),
    ("j", "z", 0.70), ("q", "c", 0.70), ("x", "s", 0.70),
    ("b", "p", 0.65), ("d", "t", 0.65), ("g", "k", 0.65),
    ("y", "i", 0.70), ("w", "u", 0.70),
]

CONFUSABLE_FINAL_PAIRS: List[Tuple[str, str, float]] = [
    ("in", "ing", 0.88), ("en", "eng", 0.88), ("an", "ang", 0.88),
    ("ian", "iang", 0.88), ("uan", "uang", 0.88),
    ("i", "v", 0.72), ("e", "o", 0.72), ("ei", "ui", 0.72),
    ("ou", "uo", 0.72), ("un", "vn", 0.75), ("ong", "iong", 0.78),
]

PY_RE = re.compile(r"^([a-z]+?)([1-5]?)$")


def split_syllable(py: str) -> Tuple[str, str, int]:
    py = py.lower().replace("ü", "v").strip()
    m = PY_RE.match(py)
    if not m:
        return "", "", 0
    body, tone_s = m.group(1), m.group(2)
    tone = int(tone_s) if tone_s else 5
    init = ""
    for cand in ("zh", "ch", "sh"):
        if body.startswith(cand):
            init = cand
            body = body[2:]
            break
    else:
        if body and body[0] in INIT_TO_ID and body[0] not in ("a", "e", "o"):
            init = body[0]
            body = body[1:]
    return init, body, tone


def _build_lookup_tables() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_init = len(INITIALS) + 1
    n_fin = len(FINALS) + 1
    init_mat = torch.eye(n_init, dtype=torch.float32)
    init_mat[0, 0] = 0.0
    for a, b, w in CONFUSABLE_INITIAL_PAIRS:
        ia, ib = INIT_TO_ID.get(a, 0), INIT_TO_ID.get(b, 0)
        if ia and ib:
            init_mat[ia, ib] = init_mat[ib, ia] = max(init_mat[ia, ib].item(), w)

    fin_mat = torch.eye(n_fin, dtype=torch.float32)
    fin_mat[0, 0] = 0.0
    for a, b, w in CONFUSABLE_FINAL_PAIRS:
        ia, ib = FINAL_TO_ID.get(a, 0), FINAL_TO_ID.get(b, 0)
        if ia and ib:
            fin_mat[ia, ib] = fin_mat[ib, ia] = max(fin_mat[ia, ib].item(), w)

    tone_mat = torch.full((6, 6), 0.65, dtype=torch.float32)
    tone_mat[0, :] = 0.0
    tone_mat[:, 0] = 0.0
    for t in range(1, 6):
        tone_mat[t, t] = 1.0
    return init_mat, fin_mat, tone_mat


def _load_radicals(chaizi_path: str | Path | None) -> Dict[str, Set[str]]:
    if not chaizi_path or not Path(chaizi_path).exists():
        return {}
    with gzip.open(chaizi_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return {k: set(v) for k, v in data.get("char_to_radicals", {}).items()}


def compute_phonetic_similarity_matrix(
    vocab: CharVocab,
    chaizi_path: str | Path | None = "data/resources/chaizi_map.json.gz",
    radical_bonus: float = 0.08,
    max_readings: int = 3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute [V, V] polyphone + confusion-weighted phonetic similarity matrix."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    V = len(vocab)
    init_ids = torch.zeros((V, max_readings), dtype=torch.long)
    final_ids = torch.zeros((V, max_readings), dtype=torch.long)
    tone_ids = torch.zeros((V, max_readings), dtype=torch.long)
    read_weight = torch.zeros((V, max_readings), dtype=torch.float32)
    valid_read = torch.zeros((V, max_readings), dtype=torch.bool)

    rank_priors = [1.0, 0.88, 0.76]
    for idx, ch in enumerate(vocab.itos):
        if idx < 2 or len(ch) != 1:
            continue
        code = ord(ch)
        if not ((0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)):
            continue
        py = pinyin(ch, style=Style.TONE3, heteronym=True, errors="ignore")
        readings = py[0] if (py and py[0]) else []
        for r_idx, r in enumerate(readings[:max_readings]):
            ini, fin, ton = split_syllable(r)
            if not ini and not fin:
                continue
            ii = INIT_TO_ID.get(ini, 0)
            fi = FINAL_TO_ID.get(fin, 0)
            if fi == 0 and ini == "":
                continue
            init_ids[idx, r_idx] = ii
            final_ids[idx, r_idx] = fi
            tone_ids[idx, r_idx] = ton
            read_weight[idx, r_idx] = rank_priors[r_idx] if r_idx < len(rank_priors) else 0.70
            valid_read[idx, r_idx] = True

    char_rads = _load_radicals(chaizi_path)
    rad_vocab: Dict[str, int] = {}
    rows_r, cols_r = [], []
    for idx, ch in enumerate(vocab.itos):
        for r in char_rads.get(ch, ()):
            rid = rad_vocab.setdefault(r, len(rad_vocab))
            rows_r.append(idx)
            cols_r.append(rid)

    init_mat, fin_mat, tone_mat = [t.to(device) for t in _build_lookup_tables()]
    init_ids = init_ids.to(device)
    final_ids = final_ids.to(device)
    tone_ids = tone_ids.to(device)
    read_weight = read_weight.to(device)
    valid_read = valid_read.to(device)

    rad_mat = torch.zeros((V, max(1, len(rad_vocab))), dtype=torch.float32, device=device)
    if rows_r:
        rad_mat[torch.tensor(rows_r, device=device), torch.tensor(cols_r, device=device)] = 1.0

    sim_out = torch.zeros((V, V), dtype=torch.float32)
    chunk = 512
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        i1 = init_ids[start:end, :, None, None]
        i2 = init_ids[None, None, :, :]
        f1 = final_ids[start:end, :, None, None]
        f2 = final_ids[None, None, :, :]
        t1 = tone_ids[start:end, :, None, None]
        t2 = tone_ids[None, None, :, :]
        w1 = read_weight[start:end, :, None, None]
        w2 = read_weight[None, None, :, :]

        pair_valid = valid_read[start:end, :, None, None] & valid_read[None, None, :, :]
        s_init = init_mat[i1, i2]
        s_fin = fin_mat[f1, f2]
        s_tone = tone_mat[t1, t2]

        prior = torch.sqrt(torch.clamp(w1 * w2, min=0.0))
        score = (0.38 * s_init + 0.47 * s_fin + 0.15 * s_tone) * prior * pair_valid.float()
        best = score.permute(0, 2, 1, 3).reshape(end - start, V, max_readings * max_readings).max(dim=-1).values

        if radical_bonus > 0 and rows_r:
            shared = (rad_mat[start:end] @ rad_mat.T) > 0
            best = torch.where((best > 0.25) & shared, torch.clamp(best + radical_bonus, max=1.0), best)

        sim_out[start:end] = best.cpu()

    sim_out.fill_diagonal_(1.0)
    sim_out[:2, :] = 0.0
    sim_out[:, :2] = 0.0
    return sim_out
