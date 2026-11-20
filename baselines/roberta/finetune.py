"""Pretrained Chinese RoBERTa/MacBERT baselines and Backbone + MV-CSN fusion models."""

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
from mvcsn.training.losses import (
    compute_consistency_loss,
    compute_mvcsn_loss,
    generate_gpu_graph_perturbation,
)
from mvcsn.training.metrics import compute_classification_metrics


class RBT3HybridDataset(Dataset):
    """Pre-tokenizes Transformer inputs & character spans once for fast GPU training."""

    def __init__(
        self,
        records: List[SpamRecord],
        vocab: CharVocab,
        tokenizer,
        preproc: TextPreprocessor | None = None,
        oov_map: Optional[Dict[str, int]] = None,
        max_len: int = 192,
        use_preprocessor: bool = True,
        soft_probs: Optional[List[float]] = None,
    ) -> None:
        self.records = records
        raw_texts = [r.raw_text for r in records]
        proc_texts = [(r.proc_text if use_preprocessor else r.raw_text)[:max_len] for r in records]

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
                    "soft_prob": float(soft_probs[i]) if soft_probs is not None else float(rec.label),
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
        soft_probs = torch.zeros((bsz,), dtype=torch.float32)

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
            soft_probs[i] = float(b["soft_prob"])

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
            "soft_probs": soft_probs,
        }

    return collate


class RBT3Classifier(nn.Module):
    def __init__(self, model_name: str = "hfl/rbt3", dropout: float = 0.15, use_proc: bool = True) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.use_proc = use_proc
        hdim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hdim, hdim // 2),
            nn.GELU(),
            nn.Linear(hdim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        in_key = "proc_input_ids" if self.use_proc else "raw_input_ids"
        m_key = "proc_attention_mask" if self.use_proc else "raw_attention_mask"
        out = self.backbone(input_ids=batch[in_key], attention_mask=batch[m_key])
        cls = out.last_hidden_state[:, 0, :]
        return {"spam_logit": self.head(cls).squeeze(-1), "backbone_feat": cls}


class RBT3MVCSNFusion(nn.Module):
    """Fuses projected Backbone contextual states with aligned MV-CSN char-graph states."""

    def __init__(
        self,
        mvcsn_model: MVCSNModel,
        model_name: str = "hfl/rbt3",
        dropout: float = 0.15,
        use_proc: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.mvcsn = mvcsn_model
        self.use_proc = use_proc
        r_dim = self.backbone.config.hidden_size
        m_dim = mvcsn_model.char_emb.embedding_dim

        self.rbt_proj = nn.Linear(r_dim, m_dim)
        self.fuse_gate = nn.Sequential(nn.Linear(m_dim * 2, m_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(m_dim * 2, m_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(m_dim, 1),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        override_char_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        in_key = "proc_input_ids" if self.use_proc else "raw_input_ids"
        m_key = "proc_attention_mask" if self.use_proc else "raw_attention_mask"
        rbt_out = self.backbone(input_ids=batch[in_key], attention_mask=batch[m_key]).last_hidden_state
        r_proj = self.rbt_proj(rbt_out)

        cids = override_char_ids if override_char_ids is not None else batch["char_ids"]
        mv_out = self.mvcsn(cids, batch["char_mask"])
        h_char = mv_out["h_fused"]

        if self.use_proc:
            h_aligned = torch.bmm(batch["span_align"], h_char)
            g = self.fuse_gate(torch.cat([r_proj, h_aligned], dim=-1))
            tok_fused = g * r_proj + (1.0 - g) * h_aligned
        else:
            tok_fused = r_proj

        mask_t = batch[m_key].float()
        tok_pooled = (tok_fused * mask_t.unsqueeze(-1)).sum(dim=1) / torch.clamp(mask_t.sum(dim=1, keepdim=True), min=1.0)
        mv_pooled, _ = self.mvcsn.pool(h_char, batch["char_mask"])
        pooled = torch.cat([tok_pooled, mv_pooled], dim=-1)
        spam_logit = self.classifier(pooled).squeeze(-1)

        bb_mean = (rbt_out * mask_t.unsqueeze(-1)).sum(dim=1) / torch.clamp(mask_t.sum(dim=1, keepdim=True), min=1.0)
        return {
            "spam_logit": spam_logit,
            "obf_logits": mv_out["obf_logits"],
            "alpha": mv_out["alpha"],
            "pooled": pooled,
            "backbone_feat": bb_mean,
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
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    for batch in loader:
        b = _to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(b)
        probs.extend(torch.sigmoid(out["spam_logit"].float()).cpu().tolist())
        targets.extend(batch["labels"].long().tolist())
    return compute_classification_metrics(targets, probs), probs, targets


@torch.no_grad()
def extract_backbone_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    logits_all: List[torch.Tensor] = []
    feats_all: List[torch.Tensor] = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    for batch in loader:
        b = _to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(b)
        logits_all.append(out["spam_logit"].float().cpu())
        feats_all.append(F.normalize(out["backbone_feat"].float(), p=2, dim=-1).cpu())
    return {
        "logits": torch.cat(logits_all, dim=0),
        "feats": torch.cat(feats_all, dim=0),
    }


def train_rbt3_family(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 4,
    lr: float = 3e-5,
    is_fusion: bool = False,
    loc_loss_weight: float = 0.25,
    use_consistency: bool = True,
    unlabeled_loader: Optional[DataLoader] = None,
    unlabeled_weight: float = 0.35,
    aug_train_loader: Optional[DataLoader] = None,
    aug_cons_weight: float = 0.35,
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
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = use_amp and not use_bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for _ in range(epochs):
        model.train()
        u_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None
        a_iter = iter(aug_train_loader) if aug_train_loader is not None else None

        for batch in train_loader:
            b = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(b)
                if not is_fusion:
                    loss = F.binary_cross_entropy_with_logits(out["spam_logit"], b["labels"])
                else:
                    aux = compute_mvcsn_loss(
                        outputs=out,
                        labels=b["labels"],
                        obf_labels=b["obf_labels"],
                        attn_mask=b["char_mask"],
                        loc_loss_weight=loc_loss_weight if b["obf_labels"].sum() > 0 else 0.0,
                        gate_prior_weight=0.05 if b["obf_labels"].sum() > 0 else 0.0,
                    )
                    loss = aux["total"]
                    if use_consistency:
                        pert_ids, pert_obf = generate_gpu_graph_perturbation(
                            input_ids=b["char_ids"],
                            attn_mask=b["char_mask"],
                            phon_nbr_idx=model.mvcsn.phon_nbr_idx,
                            phon_nbr_sim=model.mvcsn.phon_nbr_sim,
                            glyph_nbr_idx=model.mvcsn.glyph_nbr_idx,
                            glyph_nbr_sim=model.mvcsn.glyph_nbr_sim,
                            token_prob=0.20,
                        )
                        out_pert = model(b, override_char_ids=pert_ids)
                        aux_p = compute_mvcsn_loss(
                            outputs=out_pert,
                            labels=b["labels"],
                            obf_labels=torch.maximum(b["obf_labels"], pert_obf),
                            attn_mask=b["char_mask"],
                            loc_loss_weight=loc_loss_weight,
                            gate_prior_weight=0.05,
                        )
                        l_cons = compute_consistency_loss(out, out_pert)
                        loss = 0.5 * loss + 0.5 * aux_p["total"] + 0.35 * l_cons

                if a_iter is not None:
                    try:
                        ab = next(a_iter)
                    except StopIteration:
                        a_iter = iter(aug_train_loader)
                        ab = next(a_iter)
                    ab_dev = _to_device(ab, device)
                    out_aug = model(ab_dev)
                    l_aug = F.binary_cross_entropy_with_logits(out_aug["spam_logit"], ab_dev["labels"])
                    loss = 0.5 * loss + 0.5 * l_aug

                if u_iter is not None:
                    try:
                        ub = next(u_iter)
                    except StopIteration:
                        u_iter = iter(unlabeled_loader)
                        ub = next(u_iter)
                    ub_dev = _to_device(ub, device)
                    out_u = model(ub_dev)
                    l_u = F.binary_cross_entropy_with_logits(
                        out_u["spam_logit"], ub_dev["soft_probs"].clamp(1e-4, 1.0 - 1e-4)
                    )
                    loss = loss + unlabeled_weight * l_u

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        val_m, _, _ = evaluate_rbt3_family(model, val_loader, device)
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {"best_val_f1": best_f1, "train_seconds": time.time() - t0}
