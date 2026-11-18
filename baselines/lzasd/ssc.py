"""Faithful textbook SoundShapeCode (SSC) similarity matrix for LZASD baseline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from pypinyin import Style, pinyin

from mvcsn.data.vocab import CharVocab

INITIALS_LIST = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
]
FINALS_LIST = [
    "a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu",
    "ie", "ve", "er", "an", "en", "in", "un", "vn", "ang", "eng", "ing", "ong",
]


def _encode_char_ssc(
    ch: str,
    structures: Dict[str, str],
    strokes: Dict[str, int],
    four_corner: Dict[str, str],
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int, int, int, int]] | None:
    if len(ch) != 1 or not ("\u4e00" <= ch <= "\u9fff"):
        return None
    if ch not in structures or ch not in strokes:
        return None

    py_res = pinyin(ch, style=Style.TONE3, heteronym=True, errors="ignore")
    if not py_res or not py_res[0]:
        return None
    py = py_res[0][0].lower().replace("ü", "v")
    tone = 0
    if py and py[-1].isdigit():
        tone = int(py[-1])
        py = py[:-1]

    ini = ""
    for cand in INITIALS_LIST:
        if py.startswith(cand):
            ini = cand
            py = py[len(cand) :]
            break

    fin = ""
    for cand in FINALS_LIST:
        if py.endswith(cand):
            fin = cand
            py = py[: -len(cand)]
            break
    coda = py

    s_code = (
        sum(ord(c) * (idx + 1) for idx, c in enumerate(ini)) if ini else 0,
        sum(ord(c) * (idx + 1) for idx, c in enumerate(fin)) if fin else 0,
        1 if coda in ("u", "i") else 0,
        tone,
    )

    st = ord(structures[ch][0])
    fc = four_corner.get(ch, "00000")
    fc_padded = (fc + "00000")[:5]
    fc_digits = [ord(c) for c in fc_padded]
    sk = int(strokes[ch])
    g_code = (st, fc_digits[0], fc_digits[1], fc_digits[2], fc_digits[3], fc_digits[4], sk)
    return s_code, g_code


def compute_ssc_similarity_matrix(
    vocab: CharVocab,
    ssc_lookup_path: str | Path = "data/resources/ssc_lookup.json.gz",
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with gzip.open(ssc_lookup_path, "rt", encoding="utf-8") as f:
        ssc = json.load(f)
    structures: Dict[str, str] = ssc.get("structures", {})
    strokes: Dict[str, int] = ssc.get("strokes", {})
    four_corner: Dict[str, str] = ssc.get("four_corner", {})

    V = len(vocab)
    sound_arr = torch.full((V, 4), -1, dtype=torch.long)
    shape_arr = torch.full((V, 7), -1, dtype=torch.long)
    valid = torch.zeros((V,), dtype=torch.bool)

    for idx, ch in enumerate(vocab.itos):
        if idx < 2:
            continue
        res = _encode_char_ssc(ch, structures, strokes, four_corner)
        if res is not None:
            sc, gc = res
            sound_arr[idx] = torch.tensor(sc, dtype=torch.long)
            shape_arr[idx] = torch.tensor(gc, dtype=torch.long)
            valid[idx] = True

    sound_arr = sound_arr.to(device)
    shape_arr = shape_arr.to(device)
    valid_d = valid.to(device)

    w_sound = torch.tensor([0.4, 0.4, 0.1, 0.1], device=device)
    w_shape = torch.tensor([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10], device=device)

    sim_out = torch.zeros((V, V), dtype=torch.float32)
    chunk = 512
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        eq_s = (sound_arr[start:end, None, :] == sound_arr[None, :, :]).float()
        eq_g = (shape_arr[start:end, None, :] == shape_arr[None, :, :]).float()
        s_sim = (eq_s * w_sound).sum(dim=-1)
        g_sim = (eq_g * w_shape).sum(dim=-1)
        tot = 0.5 * s_sim + 0.5 * g_sim
        both = valid_d[start:end, None] & valid_d[None, :]
        sim_out[start:end] = torch.where(both, tot, torch.zeros_like(tot)).cpu()

    sim_out.fill_diagonal_(1.0)
    sim_out[:2, :] = 0.0
    sim_out[:, :2] = 0.0
    return sim_out, valid
