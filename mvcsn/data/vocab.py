"""Leakage-free character vocabulary built strictly from training text."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class CharVocab:
    """Character-to-ID bidirectional table with reserved ``<PAD>=0`` and ``<UNK>=1``."""

    def __init__(self, stoi: Dict[str, int] | None = None) -> None:
        if stoi is None:
            self.stoi: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        else:
            self.stoi = dict(stoi)
        self.itos: List[str] = [""] * len(self.stoi)
        for ch, idx in self.stoi.items():
            if idx < len(self.itos):
                self.itos[idx] = ch

    def __len__(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        """Reserved index (``0``) for sequence padding."""
        return 0

    @property
    def unk_id(self) -> int:
        """Reserved index (``1``) for out-of-vocabulary characters."""
        return 1

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_size: int = 8000,
    ) -> "CharVocab":
        """Build a frequency-ranked vocabulary strictly from training texts."""
        counter: Counter[str] = Counter()
        for txt in texts:
            counter.update(txt)
        stoi: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for ch, freq in counter.most_common():
            if freq < min_freq or ch in stoi:
                continue
            if len(stoi) >= max_size:
                break
            stoi[ch] = len(stoi)
        return cls(stoi)

    def encode(
        self,
        text: str,
        max_len: int = 192,
        oov_map: Optional[Dict[str, int]] = None,
    ) -> List[int]:
        """Encode a string into token IDs, optionally resolving OOV chars via ``oov_map``."""
        ids: List[int] = []
        for ch in text[:max_len]:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            elif oov_map is not None and ch in oov_map:
                ids.append(oov_map[ch])
            else:
                ids.append(self.unk_id)
        return ids if ids else [self.unk_id]

    def decode(self, ids: Iterable[int]) -> str:
        """Decode a sequence of token IDs back to text, skipping ``<PAD>``."""
        return "".join(
            self.itos[i] if 0 <= i < len(self.itos) else UNK_TOKEN
            for i in ids
            if i != self.pad_id
        )

    def save(self, path: str | Path) -> None:
        """Persist the character-to-ID mapping to UTF-8 JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        """Load a persisted :class:`CharVocab` JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["stoi"])
