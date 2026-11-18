"""Multi-task training objective: Spam BCE + Obfuscation Localization + Gate Prior."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn.functional as F


def compute_mvcsn_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    obf_labels: torch.Tensor,
    attn_mask: torch.Tensor,
    loc_loss_weight: float = 0.40,
    gate_prior_weight: float = 0.10,
    use_loc_head: bool = True,
    use_gate: bool = True,
) -> Dict[str, torch.Tensor]:
    spam_loss = F.binary_cross_entropy_with_logits(outputs["spam_logit"], labels)
    total = spam_loss
    loc_loss = torch.zeros((), device=labels.device)
    gate_loss = torch.zeros((), device=labels.device)

    valid = attn_mask.float()
    denom = torch.clamp(valid.sum(), min=1.0)

    if use_loc_head and loc_loss_weight > 0.0:
        pos_w = torch.where(obf_labels > 0.5, torch.full_like(obf_labels, 4.0), torch.ones_like(obf_labels))
        bce_tok = F.binary_cross_entropy_with_logits(outputs["obf_logits"], obf_labels, reduction="none")
        loc_loss = (bce_tok * pos_w * valid).sum() / denom
        total = total + loc_loss_weight * loc_loss

    if use_gate and gate_prior_weight > 0.0:
        alpha = outputs["alpha"]
        alpha_ctx = torch.clamp(alpha[..., 0], min=1e-6, max=1.0 - 1e-6)
        alpha_graph = torch.clamp(alpha[..., 1] + alpha[..., 2], min=1e-6, max=1.0 - 1e-6)
        is_obf = (obf_labels > 0.5).float()
        is_clean = 1.0 - is_obf
        g_clean = -torch.log(alpha_ctx) * is_clean
        g_obf = -torch.log(alpha_graph) * is_obf
        gate_loss = ((g_clean + 2.0 * g_obf) * valid).sum() / denom
        total = total + gate_prior_weight * gate_loss

    return {"total": total, "spam": spam_loss, "loc": loc_loss, "gate": gate_loss}
