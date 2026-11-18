"""Training & evaluation loop for MV-CSN v2 with RoCBert consistency & AMP."""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from mvcsn.training.losses import (
    compute_consistency_loss,
    compute_mvcsn_loss,
    generate_gpu_graph_perturbation,
)
from mvcsn.training.metrics import compute_classification_metrics


@torch.no_grad()
def evaluate_mvcsn(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], List[float], List[int]]:
    model.eval()
    probs: List[float] = []
    targets: List[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attn_mask"].to(device, non_blocking=True)
        out = model(input_ids, attn_mask)
        p = torch.sigmoid(out["spam_logit"]).detach().cpu().tolist()
        probs.extend(p)
        targets.extend(batch["labels"].long().tolist())
    return compute_classification_metrics(targets, probs), probs, targets


def train_mvcsn(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    patience: int = 6,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    loc_loss_weight: float = 0.40,
    gate_prior_weight: float = 0.10,
    use_consistency: bool = True,
    cons_weight: float = 0.50,
) -> Dict[str, object]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history: List[Dict[str, float]] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)
            obf_labels = batch["obf_labels"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out_orig = model(input_ids, attn_mask)
                losses_orig = compute_mvcsn_loss(
                    outputs=out_orig,
                    labels=labels,
                    obf_labels=obf_labels,
                    attn_mask=attn_mask,
                    loc_loss_weight=loc_loss_weight if obf_labels.sum() > 0 else 0.0,
                    gate_prior_weight=gate_prior_weight,
                    use_loc_head=getattr(model, "use_loc_head", True),
                    use_gate=getattr(model, "use_gate", True),
                )
                total = losses_orig["total"]

                if use_consistency:
                    pert_ids, pert_obf = generate_gpu_graph_perturbation(
                        input_ids=input_ids,
                        attn_mask=attn_mask,
                        phon_nbr_idx=model.phon_nbr_idx,
                        phon_nbr_sim=model.phon_nbr_sim,
                        glyph_nbr_idx=model.glyph_nbr_idx,
                        glyph_nbr_sim=model.glyph_nbr_sim,
                        token_prob=0.20,
                    )
                    combined_obf = torch.maximum(obf_labels, pert_obf)
                    out_pert = model(pert_ids, attn_mask)
                    losses_pert = compute_mvcsn_loss(
                        outputs=out_pert,
                        labels=labels,
                        obf_labels=combined_obf,
                        attn_mask=attn_mask,
                        loc_loss_weight=loc_loss_weight,
                        gate_prior_weight=gate_prior_weight,
                        use_loc_head=getattr(model, "use_loc_head", True),
                        use_gate=getattr(model, "use_gate", True),
                    )
                    l_cons = compute_consistency_loss(out_orig, out_pert)
                    total = 0.5 * total + 0.5 * losses_pert["total"] + cons_weight * l_cons

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tot_loss += float(total.item())
            n_batches += 1

        scheduler.step()
        val_metrics, _, _ = evaluate_mvcsn(model, val_loader, device)
        history.append({"epoch": epoch, "loss": tot_loss / max(1, n_batches), "val_f1": val_metrics["f1"]})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "best_val_f1": best_f1,
        "epochs_run": len(history),
        "train_seconds": time.time() - t0,
        "history": history,
    }
