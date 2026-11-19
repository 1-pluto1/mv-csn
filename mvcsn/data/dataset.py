"""PyTorch Dataset and padding collator for spam detection and token localization."""

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
    """Character-indexed dataset returning token IDs, binary labels, and obfuscation masks.

    Precomputes static encoded sequences once at initialization for fast epoch
    iteration while supporting optional on-the-fly adversarial perturbation
    during training when ``is_train=True`` and ``aug_prob > 0``.
    """

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

        # Precompute static IDs and token-level masks once for fast DataLoader indexing.
        self._static_ids: List[List[int]] = []
        self._static_obf: List[List[int]] = []
        for rec in records:
            txt = rec.proc_text if use_preprocessor else rec.raw_text
            ids = vocab.encode(txt, max_len=max_len, oov_map=oov_map)
            if use_preprocessor:
                obf = rec.obf_mask[: len(ids)]
            else:
                obf = compute_token_obfuscation_mask(
                    rec.raw_text, rec.clean_text, rec.label, max_len=max_len
                )[: len(ids)]
            if len(obf) < len(ids):
                obf = list(obf) + [0] * (len(ids) - len(obf))
            self._static_ids.append(ids)
            self._static_obf.append(obf)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        if self.is_train and self.aug_prob > 0.0 and self.rng.random() < self.aug_prob:
            base = (
                rec.clean_text
                if (rec.label == 1 and rec.clean_text)
                else (rec.proc_text if self.use_preprocessor else rec.raw_text)
            )
            rate = self.rng.uniform(0.12, 0.35)
            pert, pert_mask = self.perturber.perturb(base, rate=rate, return_mask=True)
            if self.preproc is not None and self.use_preprocessor:
                text, obf = self.preproc.process_with_mask(pert, pert_mask, enabled=True)
            else:
                text, obf = pert, pert_mask
            ids = self.vocab.encode(text, max_len=self.max_len, oov_map=self.oov_map)
            obf = obf[: len(ids)]
            if len(obf) < len(ids):
                obf = list(obf) + [0] * (len(ids) - len(obf))
        else:
            ids = self._static_ids[idx]
            obf = self._static_obf[idx]

        return {
            "ids": ids,
            "label": float(rec.label),
            "obf_mask": obf,
            "length": len(ids),
        }


def collate_spam_batch(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length character sequences into dense batch tensors.

    Returns:
        Dictionary with ``input_ids`` ``[B, L]``, ``attn_mask`` ``[B, L]``,
        ``obf_labels`` ``[B, L]``, and ``labels`` ``[B]``.
    """
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
