"""Training and evaluation loop for MV-CSN v2 with RoCBert consistency & AMP."""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from mvcsn.training.losses import (
    compute_consistency_loss,
    compute_mvcsn_loss,
    generate_gpu_graph_perturbation,
)
from mvcsn.training.metrics import compute_classification_metrics
from mvcsn.utils.config import TrainConfig


@torch.no_grad()
def evaluate_mvcsn(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], List[float], List[int]]:
    """Evaluate an MV-CSN model on a DataLoader and return metrics, probs, and labels."""
    model.eval()
    probs: List[float] = []
    targets: List[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attn_mask"].to(device, non_blocking=True)
        out = model(input_ids, attn_mask)
        probs.extend(torch.sigmoid(out["spam_logit"]).detach().cpu().tolist())
        targets.extend(batch["labels"].long().tolist())
    return compute_classification_metrics(targets, probs), probs, targets


def _compute_batch_objective(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Run clean + optional RoCBert-perturbed forward passes and return scalar batch loss."""
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attn_mask = batch["attn_mask"].to(device, non_blocking=True)
    obf_labels = batch["obf_labels"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    use_loc = getattr(model, "use_loc_head", True)
    use_gate = getattr(model, "use_gate", True)

    out_orig = model(input_ids, attn_mask)
    losses_orig = compute_mvcsn_loss(
        outputs=out_orig,
        labels=labels,
        obf_labels=obf_labels,
        attn_mask=attn_mask,
        loc_loss_weight=cfg.loc_loss_weight if obf_labels.sum() > 0 else 0.0,
        gate_prior_weight=cfg.gate_prior_weight,
        use_loc_head=use_loc,
        use_gate=use_gate,
    )
    total = losses_orig["total"]

    if cfg.use_consistency:
        pert_ids, pert_obf = generate_gpu_graph_perturbation(
            input_ids=input_ids,
            attn_mask=attn_mask,
            phon_nbr_idx=model.phon_nbr_idx,
            phon_nbr_sim=model.phon_nbr_sim,
            glyph_nbr_idx=model.glyph_nbr_idx,
            glyph_nbr_sim=model.glyph_nbr_sim,
            token_prob=cfg.pert_token_prob,
        )
        combined_obf = torch.maximum(obf_labels, pert_obf)
        out_pert = model(pert_ids, attn_mask)
        losses_pert = compute_mvcsn_loss(
            outputs=out_pert,
            labels=labels,
            obf_labels=combined_obf,
            attn_mask=attn_mask,
            loc_loss_weight=cfg.loc_loss_weight,
            gate_prior_weight=cfg.gate_prior_weight,
            use_loc_head=use_loc,
            use_gate=use_gate,
        )
        l_cons = compute_consistency_loss(out_orig, out_pert)
        total = 0.5 * total + 0.5 * losses_pert["total"] + cfg.cons_weight * l_cons

    return total


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
    config: Optional[TrainConfig] = None,
) -> Dict[str, object]:
    """Train MV-CSN v2 with AdamW, cosine LR decay, AMP, and val-F1 early stopping.

    Accepts either an explicit :class:`TrainConfig` via ``config`` or legacy
    keyword arguments for backward compatibility with existing scripts.

    Returns:
        Training summary dictionary with ``best_val_f1``, ``epochs_run``,
        ``train_seconds``, and per-epoch ``history``.
    """
    cfg = config or TrainConfig(
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        loc_loss_weight=loc_loss_weight,
        gate_prior_weight=gate_prior_weight,
        use_consistency=use_consistency,
        cons_weight=cons_weight,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history: List[Dict[str, float]] = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                total = _compute_batch_objective(model, batch, device, cfg)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tot_loss += float(total.item())
            n_batches += 1

        scheduler.step()
        val_metrics, _, _ = evaluate_mvcsn(model, val_loader, device)
        history.append(
            {"epoch": epoch, "loss": tot_loss / max(1, n_batches), "val_f1": val_metrics["f1"]}
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return {
        "best_val_f1": best_f1,
        "epochs_run": len(history),
        "train_seconds": time.time() - t0,
        "history": history,
    }
