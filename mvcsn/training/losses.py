"""Multi-task objective: Spam BCE + Token Localization + Gate Prior + RoCBert Consistency."""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_gpu_graph_perturbation(
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    phon_nbr_idx: torch.Tensor,
    phon_nbr_sim: torch.Tensor,
    glyph_nbr_idx: torch.Tensor,
    glyph_nbr_sim: torch.Tensor,
    token_prob: float = 0.20,
    min_sim: float = 0.60,
    unk_prob: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast GPU-vectorized RoCBert-style multimodal graph perturbation."""
    B, L = input_ids.shape
    dev = input_ids.device

    # Combine phonetic + glyph neighbor candidates: [B, L, 2*K]
    cand_idx = torch.cat([phon_nbr_idx[input_ids], glyph_nbr_idx[input_ids]], dim=-1)
    cand_sim = torch.cat([phon_nbr_sim[input_ids], glyph_nbr_sim[input_ids]], dim=-1)
    valid_cand = cand_sim >= min_sim

    # Sample random valid neighbor per position via Gumbel/uniform scoring
    rand_scores = torch.rand_like(cand_sim)
    rand_scores = torch.where(valid_cand, rand_scores, torch.full_like(rand_scores, -1.0))
    best_pos = rand_scores.argmax(dim=-1, keepdim=True)
    sampled_nbr = cand_idx.gather(-1, best_pos).squeeze(-1)
    has_cand = valid_cand.any(dim=-1)

    drop_mask = (torch.rand((B, L), device=dev) < token_prob) & attn_mask & (input_ids >= 2)
    use_unk = torch.rand((B, L), device=dev) < unk_prob
    rep_ids = torch.where(has_cand & (~use_unk), sampled_nbr, torch.ones_like(input_ids))

    pert_ids = torch.where(drop_mask, rep_ids, input_ids)
    pert_obf = drop_mask.float()
    return pert_ids, pert_obf


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


def compute_consistency_loss(
    out_orig: Dict[str, torch.Tensor],
    out_pert: Dict[str, torch.Tensor],
) -> torch.Tensor:
    p1 = torch.sigmoid(out_orig["spam_logit"])
    p2 = torch.sigmoid(out_pert["spam_logit"])
    prob_mse = F.mse_loss(p1, p2)
    repr_cos = (1.0 - F.cosine_similarity(out_orig["pooled"], out_pert["pooled"], dim=-1)).mean()
    return prob_mse + 0.15 * repr_cos
