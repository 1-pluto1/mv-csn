"""Evaluation metrics, per-attack breakdown, and exact McNemar significance test."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ATTACK_CATEGORIES = ("PHON", "GLYPH", "SPLIT", "SPACE", "MIXED", "CLEAN")


def compute_classification_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary spam-detection metrics including clean-ham accuracy and ROC-AUC.

    Args:
        y_true: Ground-truth binary labels (``0`` = ham, ``1`` = spam).
        y_prob: Predicted spam probabilities in ``[0, 1]``.
        threshold: Classification decision threshold (default ``0.5``).

    Returns:
        Dictionary with ``precision``, ``recall``, ``f1``, ``macro_f1``,
        ``accuracy``, ``auc``, ``clean_ham_acc``, and ``spam_recall``.
    """
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_prob), dtype=np.float64)
    yhat = (yp >= threshold).astype(np.int64)

    ham_mask = yt == 0
    adv_mask = yt == 1
    ham_acc = float((yhat[ham_mask] == 0).mean()) if ham_mask.any() else 0.0
    adv_rec = float((yhat[adv_mask] == 1).mean()) if adv_mask.any() else 0.0
    try:
        auc = float(roc_auc_score(yt, yp))
    except ValueError:
        auc = 0.5

    return {
        "precision": float(precision_score(yt, yhat, zero_division=0)),
        "recall": float(recall_score(yt, yhat, zero_division=0)),
        "f1": float(f1_score(yt, yhat, zero_division=0)),
        "macro_f1": float(f1_score(yt, yhat, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(yt, yhat)),
        "auc": auc,
        "clean_ham_acc": ham_acc,
        "spam_recall": adv_rec,
    }


def compute_per_attack_f1(
    y_true: List[int],
    y_prob: List[float],
    attack_types: List[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate per-attack F1 and recall against the shared clean-ham pool.

    For each evasion category (``PHON``, ``GLYPH``, ``SPLIT``, ``SPACE``,
    ``MIXED``), pairs spam messages of that category with all clean-ham test
    samples so precision reflects false positives realistically.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float64)
    yhat = (yp >= threshold).astype(np.int64)
    atks = np.asarray(attack_types)

    ham_idx = yt == 0
    res: Dict[str, float] = {}
    for tag in ATTACK_CATEGORIES:
        if tag == "CLEAN":
            sel = atks == "CLEAN"
            res[tag] = float(f1_score(yt[sel], yhat[sel], zero_division=0)) if sel.any() else 0.0
        else:
            sel_atk = (atks == tag) & (yt == 1)
            if not sel_atk.any():
                res[tag] = 0.0
                continue
            sel = sel_atk | ham_idx
            res[tag] = float(f1_score(yt[sel], yhat[sel], zero_division=0))
            res[f"{tag}_recall"] = float((yhat[sel_atk] == 1).mean())
    return res


def mcnemar_test(
    y_true: Iterable[int],
    pred_a: Iterable[int],
    pred_b: Iterable[int],
) -> Dict[str, float]:
    """Run exact two-sided McNemar binomial test comparing paired model predictions.

    Args:
        y_true: Ground-truth labels.
        pred_a: Binary predictions of model A.
        pred_b: Binary predictions of model B.

    Returns:
        Dictionary ``{"b": A-only correct, "c": B-only correct, "p_value": exact p}``.
    """
    yt = np.asarray(list(y_true), dtype=np.int64)
    pa = np.asarray(list(pred_a), dtype=np.int64)
    pb = np.asarray(list(pred_b), dtype=np.int64)
    ca = pa == yt
    cb = pb == yt
    b = int((ca & (~cb)).sum())
    c = int(((~ca) & cb).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    p_val = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return {"b": b, "c": c, "p_value": p_val}
