"""Adversarial perturbation engine for training augmentation and intensity stress tests."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from mvcsn.data.preprocessing import is_cjk

PHON_MAP: Dict[str, List[str]] = {
    "加": ["伽", "佳", "家", "迦"],
    "微": ["威", "维", "薇", "危"],
    "信": ["芯", "新", "欣"],
    "注": ["住", "驻", "柱"],
    "册": ["策", "测", "侧"],
    "充": ["茺", "冲", "崇"],
    "送": ["宋", "颂", "松"],
    "高": ["髙", "膏", "糕"],
    "赚": ["賺", "篆", "转"],
    "赔": ["賠", "陪", "培"],
    "领": ["領", "岭", "零"],
    "金": ["鑫", "斤", "今"],
    "钱": ["錢", "前", "浅"],
    "网": ["網", "往", "旺"],
    "彩": ["采", "菜", "才"],
    "博": ["搏", "播", "伯"],
    "贷": ["貸", "代", "带"],
    "款": ["宽", "歀"],
    "快": ["筷", "块"],
    "天": ["夭", "添"],
    "百": ["佰", "白"],
    "万": ["萬", "蔓"],
    "专": ["專", "砖"],
    "顶": ["頂", "鼎"],
    "赠": ["贈", "增"],
    "一": ["壹", "乙"],
    "发": ["發", "乏"],
    "现": ["現", "县"],
    "联": ["聯", "连"],
    "系": ["係", "细"],
    "扣": ["叩", "口"],
    "稳": ["穏", "问"],
}

GLYPH_MAP: Dict[str, List[str]] = {
    "元": ["え", "兀"],
    "高": ["髙"],
    "稳": ["穏"],
    "充": ["茺"],
    "天": ["夭"],
    "不": ["芣"],
    "到": ["箌"],
    "你": ["伱", "妳"],
    "百": ["佰"],
    "万": ["萬"],
    "赚": ["賺"],
    "赔": ["賠"],
    "顶": ["頂"],
    "专": ["專"],
    "赠": ["贈"],
    "Q": ["Ꝗ", "q"],
    "q": ["Ꝗ", "Q"],
    "o": ["0", "O"],
    "O": ["0", "o"],
    "0": ["o", "O", "〇"],
    "8": ["⑧", "B"],
    "3": ["③"],
    "6": ["⑥"],
    "1": ["l", "I"],
    "l": ["1", "I"],
    "v": ["V", "\\/"],
    "V": ["v", "\\/"],
}

SPLIT_MAP: Dict[str, str] = {
    "清": "氵青",
    "注": "氵主",
    "海": "氵每",
    "流": "氵充",
    "河": "氵可",
    "活": "氵舌",
    "信": "亻言",
    "你": "亻尔",
    "他": "亻也",
    "倍": "亻咅",
    "稳": "禾急",
    "私": "禾厶",
    "利": "禾刂",
    "财": "贝才",
    "账": "贝长",
    "赠": "贝曾",
    "赔": "贝咅",
    "请": "讠青",
    "话": "讠舌",
    "计": "讠十",
    "加": "力口",
    "v": "\\/",
    "Q": "Ꝗ",
}


class AdversarialPerturber:
    """Applies controlled character-level evasion perturbations and tracks binary obf masks."""

    def __init__(self, seed: int = 42, allow_glyph: bool = True) -> None:
        self.rng = random.Random(seed)
        self.allow_glyph = allow_glyph

    def perturb(
        self,
        text: str,
        rate: float = 0.25,
        modes: Tuple[str, ...] = ("PHON", "GLYPH", "SPLIT", "SPACE"),
        return_mask: bool = False,
    ) -> str | Tuple[str, List[int]]:
        if rate <= 0.0 or not text:
            return (text, [0] * len(text)) if return_mask else text

        allowed = [m for m in modes if self.allow_glyph or m != "GLYPH"]
        if not allowed:
            allowed = ["PHON"]

        chars: List[str] = []
        mask: List[int] = []
        for ch in text:
            if self.rng.random() < rate:
                m = self.rng.choice(allowed)
                if m == "GLYPH" and ch in GLYPH_MAP:
                    rep = self.rng.choice(GLYPH_MAP[ch])
                    chars.extend(list(rep))
                    mask.extend([1] * len(rep))
                    continue
                if m == "PHON" and ch in PHON_MAP:
                    rep = self.rng.choice(PHON_MAP[ch])
                    chars.append(rep)
                    mask.append(1)
                    continue
                if m == "SPLIT" and ch in SPLIT_MAP:
                    rep = SPLIT_MAP[ch]
                    chars.extend(list(rep))
                    mask.extend([1] * len(rep))
                    continue
                if m == "SPACE" and (is_cjk(ch) or ch.isalnum()):
                    noise = self.rng.choice([" ", "  ", "（", "）", "*"])
                    if noise in ("（", "）"):
                        chars.extend(["（", ch, "）"])
                        mask.extend([1, 0, 1])
                    else:
                        chars.extend([ch, noise])
                        mask.extend([0, 1])
                    continue
                if self.allow_glyph and ch in GLYPH_MAP:
                    rep = self.rng.choice(GLYPH_MAP[ch])
                    chars.extend(list(rep))
                    mask.extend([1] * len(rep))
                    continue
                if ch in PHON_MAP:
                    rep = self.rng.choice(PHON_MAP[ch])
                    chars.append(rep)
                    mask.append(1)
                    continue
            chars.append(ch)
            mask.append(0)
        res = "".join(chars)
        return (res, mask) if return_mask else res
