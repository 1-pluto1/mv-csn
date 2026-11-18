"""PyTorch Dataset and padding collator for spam detection & token localization."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from mvcsn.data.augmentation import AdversarialPerturber
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, compute_token_obfuscation_mask
from mvcsn.data.vocab import CharVocab


class SpamDataset(Dataset):
    def __init__(
        self,
        records: List[SpamRecord],
        vocab: CharVocab,
        max_len: int = 192,
        is_train: bool = False,
        aug_prob: float = 0.0,
        allow_glyph_aug: bool = True,
        preproc: TextPreprocessor | None = None,
        use_preprocessor: bool = True,
        oov_map: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.vocab = vocab
        self.max_len = max_len
        self.is_train = is_train
        self.aug_prob = aug_prob
        self.preproc = preproc
        self.use_preprocessor = use_preprocessor
        self.oov_map = oov_map
        self.rng = random.Random(seed)
        self.perturber = AdversarialPerturber(seed=seed, allow_glyph=allow_glyph_aug)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        text = rec.proc_text if self.use_preprocessor else rec.raw_text
        obf = rec.obf_mask if self.use_preprocessor else compute_token_obfuscation_mask(
            rec.raw_text, rec.clean_text, rec.label, max_len=self.max_len
        )

        if self.is_train and self.aug_prob > 0.0 and self.rng.random() < self.aug_prob:
            base = rec.clean_text if (rec.label == 1 and rec.clean_text) else text
            rate = self.rng.uniform(0.12, 0.35)
            pert, pert_mask = self.perturber.perturb(base, rate=rate, return_mask=True)
            if self.preproc is not None and self.use_preprocessor:
                text = self.preproc.process(pert, enabled=True)
                obf = compute_token_obfuscation_mask(text, base, 1, max_len=self.max_len)
            else:
                text = pert
                obf = pert_mask[: self.max_len]

        ids = self.vocab.encode(text, max_len=self.max_len, oov_map=self.oov_map)
        seq_len = len(ids)
        if len(obf) < seq_len:
            obf = list(obf) + [0] * (seq_len - len(obf))
        else:
            obf = list(obf[:seq_len])

        return {
            "ids": ids,
            "label": float(rec.label),
            "obf_mask": obf,
            "length": seq_len,
        }


def collate_spam_batch(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    max_len = max(int(item["length"]) for item in batch)
    bsz = len(batch)
    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    obf_labels = torch.zeros((bsz, max_len), dtype=torch.float32)
    labels = torch.zeros((bsz,), dtype=torch.float32)

    for i, item in enumerate(batch):
        L = int(item["length"])
        input_ids[i, :L] = torch.tensor(item["ids"], dtype=torch.long)
        attn_mask[i, :L] = True
        obf_labels[i, :L] = torch.tensor(item["obf_mask"], dtype=torch.float32)
        labels[i] = float(item["label"])

    return {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "obf_labels": obf_labels,
        "labels": labels,
    }
