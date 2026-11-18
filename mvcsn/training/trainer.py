"""Training and evaluation loop for MV-CSN with AMP and early stopping."""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from mvcsn.training.losses import compute_mvcsn_loss
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
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attn_mask"].to(device)
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
    epochs: int = 22,
    patience: int = 6,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    loc_loss_weight: float = 0.40,
    gate_prior_weight: float = 0.10,
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
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            obf_labels = batch["obf_labels"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(input_ids, attn_mask)
                losses = compute_mvcsn_loss(
                    outputs=out,
                    labels=labels,
                    obf_labels=obf_labels,
                    attn_mask=attn_mask,
                    loc_loss_weight=loc_loss_weight,
                    gate_prior_weight=gate_prior_weight,
                    use_loc_head=getattr(model, "use_loc_head", True),
                    use_gate=getattr(model, "use_gate", True),
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tot_loss += float(losses["total"].item())
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
