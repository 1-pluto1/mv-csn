"""Dual-Teacher Cross-View Knowledge Distillation & Pretrained Char Embedding Init for MV-CSN v3."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mvcsn.data.splits import SpamRecord
from mvcsn.data.vocab import CharVocab
from mvcsn.training.losses import (
    compute_consistency_loss,
    compute_mvcsn_loss,
    generate_gpu_graph_perturbation,
)
from mvcsn.training.trainer import evaluate_mvcsn
from mvcsn.utils.config import DistillConfig


def build_pretrained_char_init(
    vocab: CharVocab,
    backbone_name: str = "hfl/chinese-macbert-base",
    hidden_dim: int = 192,
    cache_path: Optional[str | Path] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if cache_path is not None and Path(cache_path).exists():
        cached = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
        if cached["init_matrix"].shape == (len(vocab), hidden_dim):
            return cached["init_matrix"], cached["stats"]

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(backbone_name)
    hf_model = AutoModel.from_pretrained(backbone_name)
    hf_vocab = tok.get_vocab()
    emb_w = hf_model.get_input_embeddings().weight.detach().float().cpu()

    matched_rows: List[int] = []
    matched_vecs: List[torch.Tensor] = []
    for idx, ch in enumerate(vocab.itos):
        if idx < 2 or not ch:
            continue
        hf_id = hf_vocab.get(ch)
        if hf_id is None and ch.isalpha():
            hf_id = hf_vocab.get(ch.lower())
        if hf_id is not None and hf_id != tok.unk_token_id:
            matched_rows.append(idx)
            matched_vecs.append(emb_w[hf_id])

    rng = torch.Generator().manual_seed(seed)
    init_mat = torch.randn((len(vocab), hidden_dim), generator=rng) * 0.05
    init_mat[0].zero_()

    if matched_vecs:
        M = torch.stack(matched_vecs, dim=0)
        M_centered = M - M.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(M_centered, full_matrices=False)
        k = min(hidden_dim, Vt.size(0))
        proj = M_centered @ Vt[:k].T
        std_val = float(proj.std().clamp(min=1e-6).item())
        proj = proj * (0.15 / std_val)
        if k < hidden_dim:
            pad_proj = torch.zeros((proj.size(0), hidden_dim))
            pad_proj[:, :k] = proj
            proj = pad_proj
        idx_t = torch.tensor(matched_rows, dtype=torch.long)
        init_mat[idx_t] = proj

    stats = {
        "backbone": backbone_name,
        "vocab_size": len(vocab),
        "hidden_dim": hidden_dim,
        "matched_tokens": len(matched_rows),
        "coverage": float(len(matched_rows) / max(1, len(vocab) - 2)),
    }
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"init_matrix": init_mat, "stats": stats}, cp)

    del hf_model
    return init_mat, stats


class DistillDataset(Dataset):
    """Lightweight dataset returning pre-encoded IDs and integer sample_idx for zero-IPC GPU teacher slicing."""

    def __init__(
        self,
        records: Sequence[SpamRecord],
        vocab: CharVocab,
        max_len: int = 192,
        oov_map: Optional[Dict[str, int]] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_feats: Optional[torch.Tensor] = None,
        unlabeled_logits: Optional[torch.Tensor] = None,
    ) -> None:
        self.records = list(records)
        self.ids_list: List[List[int]] = []
        self.obf_list: List[List[int]] = []
        self.labels_list: List[float] = []
        for rec in self.records:
            ids = vocab.encode(rec.proc_text, max_len=max_len, oov_map=oov_map)
            obf = list(rec.obf_mask[: len(ids)])
            if len(obf) < len(ids):
                obf.extend([0] * (len(ids) - len(obf)))
            self.ids_list.append(ids)
            self.obf_list.append(obf)
            self.labels_list.append(float(rec.label))

        self.teacher_logits = teacher_logits.float().cpu() if teacher_logits is not None else None
        self.teacher_feats = teacher_feats.float().cpu() if teacher_feats is not None else None
        self.unlabeled_logits = unlabeled_logits.float().cpu() if unlabeled_logits is not None else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "sample_idx": idx,
            "ids": self.ids_list[idx],
            "obf_mask": self.obf_list[idx],
            "label": self.labels_list[idx],
            "length": len(self.ids_list[idx]),
        }


def collate_distill_batch(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    max_len = max(int(item["length"]) for item in batch)
    bsz = len(batch)
    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    obf_labels = torch.zeros((bsz, max_len), dtype=torch.float32)
    labels = torch.zeros((bsz,), dtype=torch.float32)
    sample_idx = torch.zeros((bsz,), dtype=torch.long)

    for i, item in enumerate(batch):
        L = int(item["length"])
        input_ids[i, :L] = torch.tensor(item["ids"], dtype=torch.long)
        attn_mask[i, :L] = True
        obf_labels[i, :L] = torch.tensor(item["obf_mask"], dtype=torch.float32)
        labels[i] = float(item["label"])
        sample_idx[i] = int(item["sample_idx"])

    return {
        "sample_idx": sample_idx,
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "obf_labels": obf_labels,
        "labels": labels,
    }


def inject_space_noise_gpu(
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    obf_labels: torch.Tensor,
    space_id: int,
    row_prob: float = 0.25,
    space_prob: float = 0.25,
    max_len: int = 192,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if space_id < 2 or row_prob <= 0.0:
        return input_ids, attn_mask, obf_labels

    B, L = input_ids.shape
    dev = input_ids.device
    inject_rows = torch.rand((B,), device=dev) < row_prob
    if not inject_rows.any():
        return input_ids, attn_mask, obf_labels

    insert_after = (
        (torch.rand((B, L), device=dev) < space_prob)
        & attn_mask
        & inject_rows.unsqueeze(1)
    )
    insert_after[:, -1] = False

    n_ins = insert_after.long().sum(dim=1)
    orig_len = attn_mask.long().sum(dim=1)
    new_len = torch.clamp(orig_len + n_ins, max=max_len)
    L_out = max(L, int(new_len.max().item()))

    shift = torch.cumsum(insert_after.long(), dim=1) - insert_after.long()
    new_pos = torch.arange(L, device=dev).unsqueeze(0) + shift

    out_mask = torch.arange(L_out, device=dev).unsqueeze(0) < new_len.unsqueeze(1)
    out_ids = torch.where(
        out_mask,
        torch.full((B, L_out), space_id, dtype=torch.long, device=dev),
        torch.zeros((B, L_out), dtype=torch.long, device=dev),
    )
    out_obf = torch.zeros((B, L_out), dtype=torch.float32, device=dev)

    valid_keep = attn_mask & (new_pos < L_out)
    b_idx, c_idx = valid_keep.nonzero(as_tuple=True)
    p_idx = new_pos[b_idx, c_idx]
    out_ids[b_idx, p_idx] = input_ids[b_idx, c_idx]
    out_obf[b_idx, p_idx] = obf_labels[b_idx, c_idx]
    return out_ids, out_mask, out_obf


def generate_hybrid_adv_batch(
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    obf_labels: torch.Tensor,
    phon_nbr_idx: torch.Tensor,
    phon_nbr_sim: torch.Tensor,
    glyph_nbr_idx: torch.Tensor,
    glyph_nbr_sim: torch.Tensor,
    space_id: int = 1,
    token_prob: float = 0.20,
    unk_prob: float = 0.25,
    space_inject_prob: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pert_ids, pert_obf = generate_gpu_graph_perturbation(
        input_ids=input_ids,
        attn_mask=attn_mask,
        phon_nbr_idx=phon_nbr_idx,
        phon_nbr_sim=phon_nbr_sim,
        glyph_nbr_idx=glyph_nbr_idx,
        glyph_nbr_sim=glyph_nbr_sim,
        token_prob=token_prob,
        unk_prob=unk_prob,
    )
    comb_obf = torch.maximum(obf_labels, pert_obf)
    if space_inject_prob > 0.0 and space_id >= 2:
        return inject_space_noise_gpu(
            pert_ids, attn_mask, comb_obf, space_id=space_id, row_prob=space_inject_prob
        )
    return pert_ids, attn_mask, comb_obf


def compute_distill_batch_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: DistillConfig,
    space_id: int,
    t_logits: Optional[torch.Tensor] = None,
    t_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attn_mask = batch["attn_mask"]
    obf_labels = batch["obf_labels"]
    labels = batch["labels"]
    use_loc = getattr(model, "use_loc_head", True)
    use_gate = getattr(model, "use_gate", True)

    out_clean = model(input_ids, attn_mask)
    clean_non_unk = attn_mask & (input_ids != 1)
    sup_clean = compute_mvcsn_loss(
        outputs=out_clean,
        labels=labels,
        obf_labels=obf_labels,
        attn_mask=clean_non_unk,
        loc_loss_weight=cfg.loc_loss_weight if obf_labels.sum() > 0 else 0.0,
        gate_prior_weight=0.08,
        use_loc_head=use_loc,
        use_gate=use_gate,
    )

    adv_ids, adv_mask, adv_obf = generate_hybrid_adv_batch(
        input_ids=input_ids,
        attn_mask=attn_mask,
        obf_labels=obf_labels,
        phon_nbr_idx=model.phon_nbr_idx,
        phon_nbr_sim=model.phon_nbr_sim,
        glyph_nbr_idx=model.glyph_nbr_idx,
        glyph_nbr_sim=model.glyph_nbr_sim,
        space_id=space_id,
        token_prob=cfg.pert_token_prob,
        unk_prob=cfg.unk_prob,
        space_inject_prob=cfg.space_inject_prob if cfg.use_space_aug else 0.0,
    )
    adv_obf = torch.maximum(adv_obf, (adv_ids == 1).float() * adv_mask.float())
    out_adv = model(adv_ids, adv_mask)
    sup_adv = compute_mvcsn_loss(
        outputs=out_adv,
        labels=labels,
        obf_labels=adv_obf,
        attn_mask=adv_mask,
        loc_loss_weight=cfg.loc_loss_weight,
        gate_prior_weight=cfg.gate_prior_weight,
        use_loc_head=use_loc,
        use_gate=use_gate,
        unk_mask=(adv_ids == 1),
    )

    total = 0.5 * sup_clean["total"] + 0.5 * sup_adv["total"]

    if cfg.use_consistency:
        total = total + cfg.cons_weight * compute_consistency_loss(out_clean, out_adv)

    if cfg.use_logit_kd and t_logits is not None:
        T = cfg.temp_domain
        t_prob = torch.sigmoid(t_logits.float() / T).clamp(1e-5, 1.0 - 1e-5)
        l_clean = F.binary_cross_entropy_with_logits(out_clean["spam_logit"].float() / T, t_prob)
        if cfg.use_cross_view_kd:
            l_adv = F.binary_cross_entropy_with_logits(out_adv["spam_logit"].float() / T, t_prob)
            l_kd = 0.5 * (l_clean + l_adv) * T
        else:
            l_kd = l_clean * T
        total = total + cfg.lambda_logit * l_kd

    if cfg.use_feat_kd and t_feats is not None and "kd_feat" in out_clean:
        h_t = F.normalize(t_feats.float(), p=2, dim=-1)
        cos_c = (1.0 - F.cosine_similarity(out_clean["kd_feat"].float(), h_t, dim=-1)).mean()
        if cfg.use_cross_view_kd:
            cos_a = (1.0 - F.cosine_similarity(out_adv["kd_feat"].float(), h_t, dim=-1)).mean()
            total = total + cfg.lambda_feat * (0.5 * (cos_c + cos_a))
        else:
            total = total + cfg.lambda_feat * cos_c

    return total


def _compute_unlabeled_kd_loss(
    model: torch.nn.Module,
    u_batch: Dict[str, torch.Tensor],
    u_logits: torch.Tensor,
    u_feats: Optional[torch.Tensor],
    cfg: DistillConfig,
    space_id: int,
) -> torch.Tensor:
    u_ids = u_batch["input_ids"]
    u_mask = u_batch["attn_mask"]
    Tu = cfg.temp_unlabeled
    p_u = torch.sigmoid(u_logits.float() / Tu).clamp(1e-5, 1.0 - 1e-5)

    out_u_clean = model(u_ids, u_mask)
    loss_uc = F.binary_cross_entropy_with_logits(out_u_clean["spam_logit"].float() / Tu, p_u) * Tu

    adv_u_ids, adv_u_mask, adv_u_obf = generate_hybrid_adv_batch(
        input_ids=u_ids,
        attn_mask=u_mask,
        obf_labels=torch.zeros_like(u_ids, dtype=torch.float32),
        phon_nbr_idx=model.phon_nbr_idx,
        phon_nbr_sim=model.phon_nbr_sim,
        glyph_nbr_idx=model.glyph_nbr_idx,
        glyph_nbr_sim=model.glyph_nbr_sim,
        space_id=space_id,
        token_prob=cfg.pert_token_prob,
        unk_prob=cfg.unk_prob,
        space_inject_prob=cfg.space_inject_prob if cfg.use_space_aug else 0.0,
    )
    adv_u_obf = torch.maximum(adv_u_obf, (adv_u_ids == 1).float() * adv_u_mask.float())
    out_u_adv = model(adv_u_ids, adv_u_mask)

    use_loc = getattr(model, "use_loc_head", True)
    use_gate = getattr(model, "use_gate", True)
    aux_adv = compute_mvcsn_loss(
        outputs=out_u_adv,
        labels=torch.zeros_like(p_u),
        obf_labels=adv_u_obf,
        attn_mask=adv_u_mask,
        loc_loss_weight=cfg.loc_loss_weight,
        gate_prior_weight=cfg.gate_prior_weight,
        use_loc_head=use_loc,
        use_gate=use_gate,
        unk_mask=(adv_u_ids == 1),
    )
    reg_u = aux_adv["loc"] * cfg.loc_loss_weight + aux_adv["gate"] * cfg.gate_prior_weight

    if cfg.use_cross_view_kd:
        loss_ua = F.binary_cross_entropy_with_logits(out_u_adv["spam_logit"].float() / Tu, p_u) * Tu
        kd_u = 0.5 * (loss_uc + loss_ua)
    else:
        kd_u = loss_uc

    if cfg.use_feat_kd and u_feats is not None and "kd_feat" in out_u_clean:
        h_ut = F.normalize(u_feats.float(), p=2, dim=-1)
        cos_uc = (1.0 - F.cosine_similarity(out_u_clean["kd_feat"].float(), h_ut, dim=-1)).mean()
        if cfg.use_cross_view_kd:
            cos_ua = (1.0 - F.cosine_similarity(out_u_adv["kd_feat"].float(), h_ut, dim=-1)).mean()
            kd_u = kd_u + cfg.lambda_feat * 0.5 * (cos_uc + cos_ua)
        else:
            kd_u = kd_u + cfg.lambda_feat * cos_uc

    if cfg.use_consistency:
        kd_u = kd_u + 0.5 * cfg.cons_weight * compute_consistency_loss(out_u_clean, out_u_adv)

    return kd_u + reg_u


def train_mvcsn_distilled(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    vocab: CharVocab,
    config: Optional[DistillConfig] = None,
    unlabeled_loader: Optional[DataLoader] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    teacher_feats: Optional[torch.Tensor] = None,
    unlabeled_logits: Optional[torch.Tensor] = None,
    unlabeled_feats: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Train MV-CSN v3 with GPU-resident teacher tables & zero-IPC DataLoader slicing."""
    cfg = config or DistillConfig()
    space_id = vocab.stoi.get(" ", vocab.unk_id)

    ds_tr = getattr(train_loader, "dataset", None)
    if teacher_logits is None and hasattr(ds_tr, "teacher_logits"):
        teacher_logits = ds_tr.teacher_logits
    if teacher_feats is None and hasattr(ds_tr, "teacher_feats"):
        teacher_feats = ds_tr.teacher_feats
    if unlabeled_loader is not None:
        ds_u = getattr(unlabeled_loader, "dataset", None)
        if unlabeled_logits is None and hasattr(ds_u, "unlabeled_logits"):
            unlabeled_logits = ds_u.unlabeled_logits

    t_log_gpu = teacher_logits.to(device) if teacher_logits is not None else None
    t_feat_gpu = teacher_feats.to(device) if teacher_feats is not None else None
    u_log_gpu = unlabeled_logits.to(device) if unlabeled_logits is not None else None
    u_feat_gpu = unlabeled_feats.to(device) if unlabeled_feats is not None else None

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))

    use_amp = device.type == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = use_amp and not use_bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_imp = 0
    history: List[Dict[str, float]] = []
    t0 = time.time()

    active_unlabeled = (
        unlabeled_loader
        if (cfg.use_unlabeled_kd and unlabeled_loader is not None and u_log_gpu is not None)
        else None
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        u_iter = iter(active_unlabeled) if active_unlabeled is not None else None
        tot_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            s_idx = b["sample_idx"]
            bt_log = t_log_gpu[s_idx] if t_log_gpu is not None else None
            bt_feat = t_feat_gpu[s_idx] if t_feat_gpu is not None else None

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = compute_distill_batch_loss(
                    model=model,
                    batch=b,
                    cfg=cfg,
                    space_id=space_id,
                    t_logits=bt_log,
                    t_feats=bt_feat,
                )
                if u_iter is not None:
                    try:
                        ub = next(u_iter)
                    except StopIteration:
                        u_iter = iter(active_unlabeled)
                        ub = next(u_iter)
                    ub_dev = {k: v.to(device, non_blocking=True) for k, v in ub.items() if torch.is_tensor(v)}
                    u_idx = ub_dev["sample_idx"]
                    bu_log = u_log_gpu[u_idx]
                    bu_feat = u_feat_gpu[u_idx] if u_feat_gpu is not None else None
                    loss_u = _compute_unlabeled_kd_loss(
                        model=model,
                        u_batch=ub_dev,
                        u_logits=bu_log,
                        u_feats=bu_feat,
                        cfg=cfg,
                        space_id=space_id,
                    )
                    loss = loss + cfg.lambda_unlabeled * loss_u

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

            tot_loss += float(loss.item())
            n_steps += 1

        sched.step()
        val_m, _, _ = evaluate_mvcsn(model, val_loader, device)
        history.append({"epoch": epoch, "loss": tot_loss / max(1, n_steps), "val_f1": val_m["f1"]})
        if val_m["f1"] > best_f1 + 1e-5:
            best_f1 = val_m["f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return {
        "best_val_f1": best_f1,
        "epochs_run": len(history),
        "train_seconds": time.time() - t0,
        "history": history,
    }
