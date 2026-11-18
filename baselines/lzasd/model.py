"""Faithful textbook LZASD baseline: Word2Vec + SSC static graph smoothing + Self-Attention + MLP."""

from __future__ import annotations

import copy
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from torch.utils.data import DataLoader

from baselines.lzasd.ssc import compute_ssc_similarity_matrix
from mvcsn.data.vocab import CharVocab
from mvcsn.training.metrics import compute_classification_metrics


def build_lzasd_smoothed_embeddings(
    train_texts: List[str],
    vocab: CharVocab,
    ssc_lookup_path: str = "data/resources/ssc_lookup.json.gz",
    emb_dim: int = 100,
    sim_threshold: float = 0.60,
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    sentences = [list(t) for t in train_texts if t]
    w2v = Word2Vec(sentences=sentences, vector_size=emb_dim, window=5, min_count=1, sg=0, seed=seed, workers=2)

    V = len(vocab)
    raw_vecs = torch.zeros((V, emb_dim), dtype=torch.float32)
    counts = torch.zeros((V,), dtype=torch.float32)
    char_cnt: Counter[str] = Counter()
    for t in train_texts:
        char_cnt.update(t)

    for idx, ch in enumerate(vocab.itos):
        if idx < 2:
            continue
        if ch in w2v.wv:
            raw_vecs[idx] = torch.from_numpy(np.array(w2v.wv[ch], copy=True))
        counts[idx] = float(char_cnt.get(ch, 0))

    sim_mat, valid_cjk = compute_ssc_similarity_matrix(vocab, ssc_lookup_path=ssc_lookup_path, device=device)
    adj = (sim_mat >= sim_threshold).float()
    weights = adj * counts.unsqueeze(0)
    denom = weights.sum(dim=-1, keepdim=True)
    smoothed = torch.where(denom > 0, (weights @ raw_vecs) / torch.clamp(denom, min=1.0), raw_vecs)

    final_emb = torch.where(valid_cjk.unsqueeze(-1), smoothed, raw_vecs * 0.0)
    final_emb[:2] = 0.0
    return final_emb


class LZASDModel(nn.Module):
    def __init__(self, pretrained_emb: torch.Tensor, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.register_buffer("char_table", pretrained_emb.clone())
        emb_dim = pretrained_emb.size(1)
        self.q_proj = nn.Linear(emb_dim, hidden_dim)
        self.k_proj = nn.Linear(emb_dim, hidden_dim)
        self.v_proj = nn.Linear(emb_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = F.embedding(input_ids, self.char_table)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~attn_mask.unsqueeze(1), -1e4)
        attn = F.softmax(scores, dim=-1) * attn_mask.unsqueeze(1).float()
        ctx = torch.bmm(attn, v)
        lengths = torch.clamp(attn_mask.float().sum(dim=1, keepdim=True), min=1.0)
        sent = (ctx * attn_mask.unsqueeze(-1).float()).sum(dim=1) / lengths
        return self.classifier(sent).squeeze(-1)


@torch.no_grad()
def evaluate_lzasd(model: LZASDModel, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], List[float], List[int]]:
    model.eval()
    probs, targets = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attn_mask"].to(device))
        probs.extend(torch.sigmoid(logits).cpu().tolist())
        targets.extend(batch["labels"].long().tolist())
    return compute_classification_metrics(targets, probs), probs, targets


def train_lzasd(
    model: LZASDModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 6,
) -> Dict[str, object]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_imp = 0
    t0 = time.time()

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            logits = model(batch["input_ids"].to(device), batch["attn_mask"].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_m, _, _ = evaluate_lzasd(model, val_loader, device)
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    model.load_state_dict(best_state)
    return {"best_val_f1": best_f1, "train_seconds": time.time() - t0}
