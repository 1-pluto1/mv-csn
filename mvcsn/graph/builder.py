"""Build and serialize dense top-K multi-view character similarity neighbor tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from mvcsn.data.vocab import CharVocab
from mvcsn.graph.glyph import compute_glyph_similarity_matrix
from mvcsn.graph.phonetic import compute_phonetic_similarity_matrix


def extract_topk_neighbors(
    sim_matrix: torch.Tensor,
    top_k: int = 16,
    min_sim: float = 0.35,
) -> Dict[str, torch.Tensor]:
    """Convert dense [V, V] similarity matrix into [V, K] index & weight tensors."""
    V = sim_matrix.shape[0]
    mat = sim_matrix.clone()
    mat.fill_diagonal_(0.0)
    mat[:2, :] = 0.0
    mat[:, :2] = 0.0
    mat[mat < min_sim] = 0.0

    k = min(top_k, V - 1)
    vals, idx = torch.topk(mat, k=k, dim=-1)
    idx = torch.where(vals > 0.0, idx, torch.zeros_like(idx))
    return {"nbr_idx": idx.long(), "nbr_sim": vals.float()}


def build_multiview_graphs(
    vocab: CharVocab,
    chaizi_path: str | Path = "data/resources/chaizi_map.json.gz",
    font_dir: str | Path = "/content/fonts",
    top_k: int = 16,
    phon_min_sim: float = 0.35,
    glyph_min_sim: float = 0.50,
    phonetic_radical_bonus: float = 0.08,
    glyph_size: int = 32,
    glyph_emb_dim: int = 64,
    glyph_ae_epochs: int = 40,
    glyph_ae_lr: float = 2e-3,
    glyph_cnn_weight: float = 0.70,
    glyph_pixel_weight: float = 0.30,
    seed: int = 42,
    device: torch.device | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Build View 1 (Soft-Phonetic) and View 2 (Rendered-Glyph CNN) dense top-K graphs."""
    phon_sim = compute_phonetic_similarity_matrix(
        vocab=vocab,
        chaizi_path=chaizi_path,
        radical_bonus=phonetic_radical_bonus,
        device=device,
    )
    glyph_sim, glyph_meta = compute_glyph_similarity_matrix(
        vocab=vocab,
        font_dir=font_dir,
        glyph_size=glyph_size,
        emb_dim=glyph_emb_dim,
        ae_epochs=glyph_ae_epochs,
        ae_lr=glyph_ae_lr,
        cnn_weight=glyph_cnn_weight,
        pixel_weight=glyph_pixel_weight,
        seed=seed,
        device=device,
    )
    phon_top = extract_topk_neighbors(phon_sim, top_k=top_k, min_sim=phon_min_sim)
    glyph_top = extract_topk_neighbors(glyph_sim, top_k=top_k, min_sim=glyph_min_sim)

    graphs = {
        "phon_nbr_idx": phon_top["nbr_idx"],
        "phon_nbr_sim": phon_top["nbr_sim"],
        "glyph_nbr_idx": glyph_top["nbr_idx"],
        "glyph_nbr_sim": glyph_top["nbr_sim"],
    }
    return graphs, glyph_meta


def save_graphs(
    graphs: Dict[str, torch.Tensor],
    glyph_meta: Dict[str, object],
    graph_path: str | Path = "artifacts/graphs_top16.pt",
    glyph_meta_path: str | Path = "artifacts/glyph_meta.pt",
) -> None:
    gp = Path(graph_path)
    gp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, gp)
    mp = Path(glyph_meta_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(glyph_meta, mp)


def load_graphs(path: str | Path = "artifacts/graphs_top16.pt") -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_glyph_meta(path: str | Path = "artifacts/glyph_meta.pt") -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)
