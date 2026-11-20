"""Standards-based selective normalization, canonical punctuation, and IDS split-character recomposition."""

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

FILLER_CHARS: Set[str] = set("()[]{}（）【】「」『』《》<>?？*＊|｜·•~～`^_")

# Zero-width & invisible Unicode formatting characters (incl. U+2060-U+206F invisible math ops)
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\xad]")


def is_cjk(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)


def normalize_char_selective(ch: str) -> str:
    """Selective UTS #39 / NFKC folding preserving CJK variants, Kana, and cross-script homoglyphs."""
    if ch == "。":
        return "."
    if ch == "、":
        return ","
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


def normalize_canonical_punct(text: str) -> str:
    """Strip invisible format chars and canonicalize fullwidth ASCII/CJK punctuation while preserving text."""
    s = ZERO_WIDTH_RE.sub("", text)
    return "".join(normalize_char_selective(c) for c in s).strip()


def strip_to_pure_script(text: str) -> str:
    """Keep strictly CJK ideographs, Latin letters, and digits (Exp 2b punctuation-free stress track)."""
    s = normalize_canonical_punct(text)
    out = [c for c in s if is_cjk(c) or (c.isascii() and c.isalnum())]
    return "".join(out)


def _should_drop_filler(ch: str, prev_c: str, next_c: str) -> bool:
    if ch not in FILLER_CHARS:
        return False
    if (is_cjk(prev_c) or prev_c.isdigit()) and (is_cjk(next_c) or next_c.isdigit()):
        return True
    return prev_c == "" or next_c == "" or prev_c.isspace() or next_c.isspace()


def _should_keep_space(prev_out: str, next_c: str) -> bool:
    return prev_out.isascii() and prev_out.isalpha() and next_c.isascii() and next_c.isalpha()


class TextPreprocessor:
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

        self._sorted_manual = sorted(
            self.manual_splits.items(), key=lambda kv: len(kv[0]), reverse=True
        )

    def normalize_noise(self, text: str) -> str:
        cleaned = "".join(normalize_char_selective(c) for c in ZERO_WIDTH_RE.sub("", text))
        out: List[str] = []
        n = len(cleaned)
        for i, ch in enumerate(cleaned):
            prev_c = cleaned[i - 1] if i > 0 else ""
            next_c = cleaned[i + 1] if i + 1 < n else ""
            if _should_drop_filler(ch, prev_c, next_c):
                continue
            if ch.isspace():
                prev_out = out[-1] if out else ""
                if _should_keep_space(prev_out, next_c) and not (out and out[-1] == " "):
                    out.append(" ")
                continue
            out.append(ch)
        return "".join(out).strip()

    def recompose_splits(self, text: str) -> str:
        for src, dst in self._sorted_manual:
            if src in text:
                text = text.replace(src, dst)
        out: List[str] = []
        i, n = 0, len(text)
        while i < n:
            if i + 2 < n and text[i : i + 3] in self.triple_to_char:
                out.append(self.triple_to_char[text[i : i + 3]])
                i += 3
            elif i + 1 < n and text[i : i + 2] in self.pair_to_char:
                out.append(self.pair_to_char[text[i : i + 2]])
                i += 2
            else:
                out.append(text[i])
                i += 1
        return "".join(out)

    def process(self, text: str, enabled: bool = True) -> str:
        if not enabled:
            return text.strip()
        return self.recompose_splits(self.normalize_noise(text))

    def process_with_mask(
        self, text: str, mask: List[int], enabled: bool = True
    ) -> Tuple[str, List[int]]:
        if not enabled:
            stripped = text.strip()
            return stripped, list(mask[: len(stripped)])

        cleaned: List[str] = []
        cleaned_m: List[int] = []
        for idx, ch in enumerate(text):
            if ZERO_WIDTH_RE.match(ch):
                continue
            norm_ch = normalize_char_selective(ch)
            cleaned.append(norm_ch)
            m_val = mask[idx] if idx < len(mask) else 0
            if norm_ch != ch:
                m_val = 1
            cleaned_m.append(m_val)

        norm_c: List[str] = []
        norm_m: List[int] = []
        n = len(cleaned)
        for i, ch in enumerate(cleaned):
            m = cleaned_m[i]
            prev_c = cleaned[i - 1] if i > 0 else ""
            next_c = cleaned[i + 1] if i + 1 < n else ""
            if _should_drop_filler(ch, prev_c, next_c):
                continue
            if ch.isspace():
                prev_out = norm_c[-1] if norm_c else ""
                if _should_keep_space(prev_out, next_c) and not (norm_c and norm_c[-1] == " "):
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
                    norm_m = [
                        1 if any(norm_m[max(0, k - 1) : k + 2]) else 0 for k in range(len(s))
                    ]

        out_chars: List[str] = []
        out_mask: List[int] = []
        i, L = 0, len(s)
        while i < L:
            if i + 2 < L and s[i : i + 3] in self.triple_to_char:
                out_chars.append(self.triple_to_char[s[i : i + 3]])
                out_mask.append(max(norm_m[i : i + 3]) if i + 2 < len(norm_m) else 1)
                i += 3
            elif i + 1 < L and s[i : i + 2] in self.pair_to_char:
                out_chars.append(self.pair_to_char[s[i : i + 2]])
                m1 = norm_m[i] if i < len(norm_m) else 0
                m2 = norm_m[i + 1] if (i + 1) < len(norm_m) else 0
                out_mask.append(max(m1, m2))
                i += 2
            else:
                out_chars.append(s[i])
                out_mask.append(norm_m[i] if i < len(norm_m) else 0)
                i += 1
        return "".join(out_chars), out_mask
