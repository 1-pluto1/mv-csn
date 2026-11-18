"""Deterministic noise normalization and IDS split-character recomposition."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Set

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


def to_halfwidth(ch: str) -> str:
    code = ord(ch)
    if code == 0x3000:
        return " "
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    return ch


class TextPreprocessor:
    """Normalizes formatting noise and recomposes split-character evasions."""

    def __init__(self, chaizi_path: str | Path = "data/resources/chaizi_map.json.gz") -> None:
        self.pair_to_char: Dict[str, str] = {}
        self.manual_splits: Dict[str, str] = {}
        self.char_to_radicals: Dict[str, List[str]] = {}
        path = Path(chaizi_path)
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            raw_pairs = payload.get("pair_to_char", {})
            for pair, target in raw_pairs.items():
                if len(pair) == 2 and (
                    pair[0] in PURE_RADICALS
                    or pair[1] in PURE_RADICALS
                    or ("\u2e80" <= pair[0] <= "\u2fdf")
                    or ("\u2e80" <= pair[1] <= "\u2fdf")
                ):
                    self.pair_to_char[pair] = target
            self.manual_splits = payload.get("manual_splits", {})
            self.char_to_radicals = payload.get("char_to_radicals", {})

        self._sorted_manual = sorted(self.manual_splits.items(), key=lambda kv: len(kv[0]), reverse=True)

    def normalize_noise(self, text: str) -> str:
        """Strip obfuscation spacing, zero-width chars, and filler brackets."""
        text = ZERO_WIDTH_RE.sub("", text)
        text = "".join(to_halfwidth(c) for c in text)
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
        """Deterministically recombine split-character and symbol substitutions."""
        for src, dst in self._sorted_manual:
            if src in text:
                text = text.replace(src, dst)
        out: List[str] = []
        i = 0
        n = len(text)
        while i < n:
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
        """Run full preprocessing pipeline unless disabled for ablation."""
        if not enabled:
            return text.strip()
        return self.recompose_splits(self.normalize_noise(text))
