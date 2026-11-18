"""Pretrained Chinese RoBERTa (hfl/rbt3) baseline and RBT3 + MV-CSN fusion model."""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord
from mvcsn.data.vocab import CharVocab
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.losses import compute_mvcsn_loss
from mvcsn.training.metrics import compute_classification_metrics


class RBT3HybridDataset(Dataset):
    """Pre-tokenizes RBT3 inputs & character spans once for 15x faster GPU training."""

    def __init__(
        self,
        records: List[SpamRecord],
        vocab: CharVocab,
        tokenizer,
        preproc: TextPreprocessor | None = None,
        oov_map: Optional[Dict[str, int]] = None,
        max_len: int = 192,
    ) -> None:
        self.records = records
        raw_texts = [r.raw_text for r in records]
        proc_texts = [r.proc_text[:max_len] for r in records]

        enc_raw = tokenizer(raw_texts, padding=False, truncation=True, max_length=max_len)
        enc_proc = tokenizer(
            proc_texts,
            padding=False,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
        )

        self.items: List[Dict[str, object]] = []
        for i, rec in enumerate(records):
            ids = vocab.encode(proc_texts[i], max_len=max_len, oov_map=oov_map)
            obf = rec.obf_mask[: len(ids)]
            if len(obf) < len(ids):
                obf = list(obf) + [0] * (len(ids) - len(obf))
            self.items.append(
                {
                    "raw_ids": enc_raw["input_ids"][i],
                    "proc_ids": enc_proc["input_ids"][i],
                    "proc_offsets": enc_proc["offset_mapping"][i],
                    "char_ids": ids,
                    "obf_mask": obf,
                    "label": float(rec.label),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.items[idx]


def make_rbt3_collator(pad_token_id: int = 0):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        max_raw = max(len(b["raw_ids"]) for b in batch)
        max_proc = max(len(b["proc_ids"]) for b in batch)
        max_c = max(len(b["char_ids"]) for b in batch)

        raw_input_ids = torch.full((bsz, max_raw), pad_token_id, dtype=torch.long)
        raw_attention_mask = torch.zeros((bsz, max_raw), dtype=torch.long)
        proc_input_ids = torch.full((bsz, max_proc), pad_token_id, dtype=torch.long)
        proc_attention_mask = torch.zeros((bsz, max_proc), dtype=torch.long)
        span_align = torch.zeros((bsz, max_proc, max_c), dtype=torch.float32)

        char_ids = torch.zeros((bsz, max_c), dtype=torch.long)
        char_mask = torch.zeros((bsz, max_c), dtype=torch.bool)
        obf_labels = torch.zeros((bsz, max_c), dtype=torch.float32)
        labels = torch.zeros((bsz,), dtype=torch.float32)

        for i, b in enumerate(batch):
            Lr = len(b["raw_ids"])
            Lp = len(b["proc_ids"])
            Lc = len(b["char_ids"])
            raw_input_ids[i, :Lr] = torch.tensor(b["raw_ids"], dtype=torch.long)
            raw_attention_mask[i, :Lr] = 1
            proc_input_ids[i, :Lp] = torch.tensor(b["proc_ids"], dtype=torch.long)
            proc_attention_mask[i, :Lp] = 1
            char_ids[i, :Lc] = torch.tensor(b["char_ids"], dtype=torch.long)
            char_mask[i, :Lc] = True
            obf_labels[i, :Lc] = torch.tensor(b["obf_mask"], dtype=torch.float32)
            labels[i] = float(b["label"])

            for t_idx, (s_c, e_c) in enumerate(b["proc_offsets"]):
                if e_c > s_c and s_c < Lc:
                    ec = min(e_c, Lc)
                    span_align[i, t_idx, s_c:ec] = 1.0 / max(1, ec - s_c)

        return {
            "raw_input_ids": raw_input_ids,
            "raw_attention_mask": raw_attention_mask,
            "proc_input_ids": proc_input_ids,
            "proc_attention_mask": proc_attention_mask,
            "span_align": span_align,
            "char_ids": char_ids,
            "char_mask": char_mask,
            "obf_labels": obf_labels,
            "labels": labels,
        }

    return collate


class RBT3Classifier(nn.Module):
    def __init__(self, model_name: str = "hfl/rbt3", dropout: float = 0.15) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hdim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hdim, hdim // 2),
            nn.GELU(),
            nn.Linear(hdim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.backbone(input_ids=batch["raw_input_ids"], attention_mask=batch["raw_attention_mask"])
        cls = out.last_hidden_state[:, 0, :]
        return {"spam_logit": self.head(cls).squeeze(-1)}


class RBT3MVCSNFusion(nn.Module):
    """Fuses projected RBT3 contextual states (768->128) with aligned MV-CSN char-graph states."""

    def __init__(self, mvcsn_model: MVCSNModel, model_name: str = "hfl/rbt3", dropout: float = 0.15) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.mvcsn = mvcsn_model
        r_dim = self.backbone.config.hidden_size
        m_dim = mvcsn_model.char_emb.embedding_dim

        self.rbt_proj = nn.Linear(r_dim, m_dim)
        self.fuse_gate = nn.Sequential(
            nn.Linear(m_dim * 2, m_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(m_dim * 2, m_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(m_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rbt_out = self.backbone(
            input_ids=batch["proc_input_ids"],
            attention_mask=batch["proc_attention_mask"],
        ).last_hidden_state
        r_proj = self.rbt_proj(rbt_out)

        mv_out = self.mvcsn(batch["char_ids"], batch["char_mask"])
        h_char = mv_out["h_fused"]
        h_aligned = torch.bmm(batch["span_align"], h_char)

        g = self.fuse_gate(torch.cat([r_proj, h_aligned], dim=-1))
        tok_fused = g * r_proj + (1.0 - g) * h_aligned

        mask_t = batch["proc_attention_mask"].float()
        tok_pooled = (tok_fused * mask_t.unsqueeze(-1)).sum(dim=1) / torch.clamp(mask_t.sum(dim=1, keepdim=True), min=1.0)
        mv_pooled, _ = self.mvcsn.pool(h_char, batch["char_mask"])
        spam_logit = self.classifier(torch.cat([tok_pooled, mv_pooled], dim=-1)).squeeze(-1)

        return {
            "spam_logit": spam_logit,
            "obf_logits": mv_out["obf_logits"],
            "alpha": mv_out["alpha"],
        }


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


@torch.no_grad()
def evaluate_rbt3_family(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], List[float], List[int]]:
    model.eval()
    probs, targets = [], []
    for batch in loader:
        b = _to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(b)
        probs.extend(torch.sigmoid(out["spam_logit"].float()).cpu().tolist())
        targets.extend(batch["labels"].long().tolist())
    return compute_classification_metrics(targets, probs), probs, targets


def train_rbt3_family(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-5,
    is_fusion: bool = False,
    loc_loss_weight: float = 0.25,
) -> Dict[str, object]:
    model = model.to(device)
    if is_fusion:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": lr},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": 5e-4},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            b = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(b)
                loss = F.binary_cross_entropy_with_logits(out["spam_logit"], b["labels"])
                if is_fusion and "obf_logits" in out:
                    aux = compute_mvcsn_loss(
                        outputs=out,
                        labels=b["labels"],
                        obf_labels=b["obf_labels"],
                        attn_mask=b["char_mask"],
                        loc_loss_weight=loc_loss_weight,
                        gate_prior_weight=0.05,
                    )
                    loss = aux["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        val_m, _, _ = evaluate_rbt3_family(model, val_loader, device)
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {"best_val_f1": best_f1, "train_seconds": time.time() - t0}
