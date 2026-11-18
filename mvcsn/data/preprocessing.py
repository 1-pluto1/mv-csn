"""Standards-based normalization and multi-component IDS split-character recomposition."""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

PURE_RADICALS: Set[str] = set(
    "氵亻讠扌艹辶宀广疒纟钅阝卩刂忄犭衤礻灬爫耂虍虫页鸟鱼"
    "⺖⺘⺡⺢⺣⺨⺩⺫⺬⺭⺮⺰⺳⺶⺷⺸⺼⺾⻊⻌⻏⻖⻗⻞⻣"
)

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\xad]")


def is_cjk(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)


def normalize_char_selective(ch: str) -> str:
    """Selective UTS #39 / NFKC normalization on fullwidth & enclosed alphanumerics only.

    Preserves CJK variants (髙/穏/茺), Kana (え), and cross-script confusables (Ꝗ/О/Ⅰ)
    for visual graph reasoning while folding circled digits (⑧->8) and fullwidth forms.
    """
    code = ord(ch)
    if code == 0x3000:
        return " "
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    if 0x2460 <= code <= 0x24FF:
        norm = unicodedata.normalize("NFKC", ch)
        if len(norm) == 1 and norm.isalnum():
            return norm
    return ch


class TextPreprocessor:
    """Normalizes formatting noise and recomposes 2-char and 3-char split evasions."""

    def __init__(self, chaizi_path: str | Path = "data/resources/chaizi_map.json.gz") -> None:
        self.pair_to_char: Dict[str, str] = {}
        self.triple_to_char: Dict[str, str] = {}
        self.manual_splits: Dict[str, str] = {}
        self.char_to_radicals: Dict[str, List[str]] = {}
        path = Path(chaizi_path)
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            for pair, target in payload.get("pair_to_char", {}).items():
                if len(pair) == 2 and (
                    pair[0] in PURE_RADICALS
                    or pair[1] in PURE_RADICALS
                    or ("\u2e80" <= pair[0] <= "\u2fdf")
                    or ("\u2e80" <= pair[1] <= "\u2fdf")
                ):
                    self.pair_to_char[pair] = target
            self.triple_to_char = payload.get("triple_to_char", {})
            self.manual_splits = payload.get("manual_splits", {})
            self.char_to_radicals = payload.get("char_to_radicals", {})

        self._sorted_manual = sorted(self.manual_splits.items(), key=lambda kv: len(kv[0]), reverse=True)

    def normalize_noise(self, text: str) -> str:
        text = ZERO_WIDTH_RE.sub("", text)
        text = "".join(normalize_char_selective(c) for c in text)
        filler_chars = set("()[]{}（）【】「」『』《》<>?？*＊|｜·•~～`^_")
        out: List[str] = []
        n = len(text)
        for i, ch in enumerate(text):
            if ch in filler_chars:
                prev_c = text[i - 1] if i > 0 else ""
                next_c = text[i + 1] if i + 1 < n else ""
                if (is_cjk(prev_c) or prev_c.isdigit()) and (is_cjk(next_c) or next_c.isdigit()):
                    continue
                if prev_c == "" or next_c == "" or prev_c.isspace() or next_c.isspace():
                    continue
            if ch.isspace():
                prev_out = out[-1] if out else ""
                next_c = text[i + 1] if i + 1 < n else ""
                if prev_out.isascii() and prev_out.isalpha() and next_c.isascii() and next_c.isalpha():
                    if not (out and out[-1] == " "):
                        out.append(" ")
                continue
            out.append(ch)
        return "".join(out).strip()

    def recompose_splits(self, text: str) -> str:
        for src, dst in self._sorted_manual:
            if src in text:
                text = text.replace(src, dst)
        out: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            if i + 2 < n:
                tri = text[i : i + 3]
                if tri in self.triple_to_char:
                    out.append(self.triple_to_char[tri])
                    i += 3
                    continue
            if i + 1 < n:
                pair = text[i : i + 2]
                if pair in self.pair_to_char:
                    out.append(self.pair_to_char[pair])
                    i += 2
                    continue
            out.append(text[i])
            i += 1
        return "".join(out)

    def process(self, text: str, enabled: bool = True) -> str:
        if not enabled:
            return text.strip()
        return self.recompose_splits(self.normalize_noise(text))

    def process_with_mask(self, text: str, mask: List[int], enabled: bool = True) -> Tuple[str, List[int]]:
        """O(L) deterministic preprocessing preserving aligned token obfuscation mask."""
        if not enabled:
            s = text.strip()
            return s, list(mask[: len(s)])
        filler_chars = set("()[]{}（）【】「」『』《》<>?？*＊|｜·•~～`^_")
        cleaned, cleaned_m = [], []
        for idx, ch in enumerate(text):
            if ZERO_WIDTH_RE.match(ch):
                continue
            norm_ch = normalize_char_selective(ch)
            cleaned.append(norm_ch)
            m_val = mask[idx] if idx < len(mask) else 0
            if norm_ch != ch:
                m_val = 1
            cleaned_m.append(m_val)

        n = len(cleaned)
        norm_c: List[str] = []
        norm_m: List[int] = []
        for i, ch in enumerate(cleaned):
            m = cleaned_m[i]
            if ch in filler_chars:
                prev_c = cleaned[i - 1] if i > 0 else ""
                next_c = cleaned[i + 1] if i + 1 < n else ""
                if (is_cjk(prev_c) or prev_c.isdigit()) and (is_cjk(next_c) or next_c.isdigit()):
                    continue
                if prev_c == "" or next_c == "" or prev_c.isspace() or next_c.isspace():
                    continue
            if ch.isspace():
                prev_out = norm_c[-1] if norm_c else ""
                next_c = cleaned[i + 1] if i + 1 < n else ""
                if prev_out.isascii() and prev_out.isalpha() and next_c.isascii() and next_c.isalpha():
                    if not (norm_c and norm_c[-1] == " "):
                        norm_c.append(" ")
                        norm_m.append(m)
                continue
            norm_c.append(ch)
            norm_m.append(m)

        s = "".join(norm_c)
        for src, dst in self._sorted_manual:
            if src in s:
                s = s.replace(src, dst)
                if len(s) != len(norm_m):
                    norm_m = [1 if any(norm_m[max(0, k - 1) : k + 2]) else 0 for k in range(len(s))]

        out_chars: List[str] = []
        out_mask: List[int] = []
        i = 0
        L = len(s)
        while i < L:
            if i + 2 < L:
                tri = s[i : i + 3]
                if tri in self.triple_to_char:
                    out_chars.append(self.triple_to_char[tri])
                    out_mask.append(max(norm_m[i : i + 3]) if i + 2 < len(norm_m) else 1)
                    i += 3
                    continue
            if i + 1 < L:
                pair = s[i : i + 2]
                if pair in self.pair_to_char:
                    out_chars.append(self.pair_to_char[pair])
                    m1 = norm_m[i] if i < len(norm_m) else 0
                    m2 = norm_m[i + 1] if (i + 1) < len(norm_m) else 0
                    out_mask.append(max(m1, m2))
                    i += 2
                    continue
            out_chars.append(s[i])
            out_mask.append(norm_m[i] if i < len(norm_m) else 0)
            i += 1
        return "".join(out_chars), out_mask
