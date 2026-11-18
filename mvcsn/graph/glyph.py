"""Script-agnostic 32x32 Rendered-Glyph Visual Similarity Graph (View 2) & OOV Resolver."""

from __future__ import annotations

import os
import random
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from mvcsn.data.vocab import CharVocab

FONT_SPECS = [
    (
        "NotoSansCJKsc-Regular.otf",
        "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    ),
    (
        "unifont.otf",
        "https://unifoundry.com/pub/unifont/unifont-15.1.05/font-builds/unifont-15.1.05.otf",
    ),
]


def ensure_fonts(font_dir: str | Path = "/content/fonts") -> Tuple[Path, Path]:
    """Ensure CJK + Pan-Unicode fonts exist locally without blocking apt-get."""
    fdir = Path(font_dir)
    if not (fdir / "NotoSansCJKsc-Regular.otf").exists():
        alt_dir = Path("assets/fonts")
        if (alt_dir / "NotoSansCJKsc-Regular.otf").exists():
            fdir = alt_dir
    fdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fname, url in FONT_SPECS:
        dest = fdir / fname
        if not dest.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
        paths.append(dest)
    return paths[0], paths[1]


class GlyphRenderer:
    """Renders any Unicode character into a tight-cropped, centered 32x32 grayscale bitmap."""

    def __init__(self, font_dir: str | Path = "/content/fonts", size: int = 32) -> None:
        self.size = size
        noto_path, uni_path = ensure_fonts(font_dir)
        self.noto_cmap = set(TTFont(str(noto_path)).getBestCmap().keys())
        self.uni_cmap = set(TTFont(str(uni_path)).getBestCmap().keys())
        self.noto_font = ImageFont.truetype(str(noto_path), 48)
        self.uni_font = ImageFont.truetype(str(uni_path), 32)

    def render(self, ch: str) -> Tuple[np.ndarray, bool]:
        if not ch or len(ch) != 1 or ch.isspace():
            return np.zeros((self.size, self.size), dtype=np.float32), False
        cp = ord(ch)
        if cp in self.noto_cmap:
            font = self.noto_font
        elif cp in self.uni_cmap:
            font = self.uni_font
        else:
            return np.zeros((self.size, self.size), dtype=np.float32), False

        canvas = Image.new("L", (96, 96), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 16), ch, fill=255, font=font)
        bbox = canvas.getbbox()
        if bbox is None:
            return np.zeros((self.size, self.size), dtype=np.float32), False

        crop = canvas.crop(bbox)
        w, h = crop.size
        side = max(w, h)
        sq = Image.new("L", (side, side), 0)
        sq.paste(crop, ((side - w) // 2, (side - h) // 2))
        margin = 3
        inner = self.size - 2 * margin
        resized = sq.resize((inner, inner), Image.Resampling.BILINEAR)
        out = Image.new("L", (self.size, self.size), 0)
        out.paste(resized, (margin, margin))
        arr = np.asarray(out, dtype=np.float32) / 255.0
        return arr, bool(arr.max() > 0.05)

    def render_vocab(self, chars: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bmps: List[np.ndarray] = []
        valid: List[bool] = []
        for idx, ch in enumerate(chars):
            if idx < 2 and ch in ("<PAD>", "<UNK>"):
                bmps.append(np.zeros((self.size, self.size), dtype=np.float32))
                valid.append(False)
            else:
                arr, ok = self.render(ch)
                bmps.append(arr)
                valid.append(ok)
        return torch.from_numpy(np.stack(bmps, axis=0)), torch.tensor(valid, dtype=torch.bool)


class GlyphCAE(nn.Module):
    """Self-supervised Convolutional Autoencoder learning shift-tolerant 64-d glyph embeddings."""

    def __init__(self, emb_dim: int = 64) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, emb_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(emb_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.enc(x), p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        return self.dec(z), F.normalize(z, p=2, dim=-1)


def _blurred_pixel_features(bitmaps: torch.Tensor) -> torch.Tensor:
    x = bitmaps.unsqueeze(1)
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1)), kernel_size=3, stride=1)
    return F.normalize(blur.flatten(1), p=2, dim=-1)


def train_glyph_cae(
    bitmaps: torch.Tensor,
    valid_mask: torch.Tensor,
    emb_dim: int = 64,
    epochs: int = 40,
    lr: float = 2e-3,
    seed: int = 42,
    device: torch.device | None = None,
) -> GlyphCAE:
    """Train GlyphCAE strictly unsupervised on rendered train-vocab character bitmaps."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = GlyphCAE(emb_dim=emb_dim).to(device)
    X = bitmaps[valid_mask].unsqueeze(1).to(device)
    if X.size(0) == 0:
        return model.cpu()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bsz = 256
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(X.size(0), device=device)
        for s in range(0, X.size(0), bsz):
            xb = X[perm[s : s + bsz]]
            dx = rng.randint(-1, 1)
            dy = rng.randint(-1, 1)
            x_aug = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
            recon, _ = model(x_aug)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


def compute_glyph_similarity_matrix(
    vocab: CharVocab,
    font_dir: str | Path = "/content/fonts",
    glyph_size: int = 32,
    emb_dim: int = 64,
    ae_epochs: int = 40,
    ae_lr: float = 2e-3,
    cnn_weight: float = 0.70,
    pixel_weight: float = 0.30,
    seed: int = 42,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Render vocab bitmaps, train unsupervised GlyphCAE, and return [V, V] visual similarity."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    renderer = GlyphRenderer(font_dir=font_dir, size=glyph_size)
    bitmaps, valid_mask = renderer.render_vocab(vocab.itos)

    cae = train_glyph_cae(
        bitmaps=bitmaps,
        valid_mask=valid_mask,
        emb_dim=emb_dim,
        epochs=ae_epochs,
        lr=ae_lr,
        seed=seed,
        device=device,
    )
    cae.eval()
    with torch.no_grad():
        x_all = bitmaps.unsqueeze(1).to(device)
        z = cae.encode(x_all)
        pflat = _blurred_pixel_features(bitmaps.to(device))

        sim = cnn_weight * (z @ z.T) + pixel_weight * (pflat @ pflat.T)
        both_valid = valid_mask.to(device)[:, None] & valid_mask.to(device)[None, :]
        sim = torch.where(both_valid, sim, torch.zeros_like(sim)).cpu()

    sim.fill_diagonal_(1.0)
    sim[:2, :] = 0.0
    sim[:, :2] = 0.0

    meta = {
        "bitmaps": bitmaps.cpu(),
        "valid_mask": valid_mask.cpu(),
        "glyph_z": z.cpu(),
        "glyph_pix": pflat.cpu(),
        "cae_state": {k: v.cpu() for k, v in cae.state_dict().items()},
        "emb_dim": emb_dim,
        "glyph_size": glyph_size,
        "cnn_weight": cnn_weight,
        "pixel_weight": pixel_weight,
    }
    return sim, meta


class OOVGlyphResolver:
    """Resolves unseen Unicode characters at inference time via frozen GlyphCAE visual matching."""

    def __init__(
        self,
        vocab: CharVocab,
        glyph_meta: Dict[str, object],
        font_dir: str | Path = "/content/fonts",
        min_sim: float = 0.72,
    ) -> None:
        self.vocab = vocab
        self.min_sim = min_sim
        self.renderer = GlyphRenderer(font_dir=font_dir, size=int(glyph_meta["glyph_size"]))
        self.cae = GlyphCAE(emb_dim=int(glyph_meta["emb_dim"]))
        self.cae.load_state_dict(glyph_meta["cae_state"])
        self.cae.eval()
        self.valid_mask: torch.Tensor = glyph_meta["valid_mask"].bool()
        self.vocab_z: torch.Tensor = glyph_meta["glyph_z"].float()
        self.vocab_pix: torch.Tensor = glyph_meta["glyph_pix"].float()
        self.cnn_weight = float(glyph_meta["cnn_weight"])
        self.pixel_weight = float(glyph_meta["pixel_weight"])
        self._cache: Dict[str, int] = {}

    def resolve_char(self, ch: str) -> Tuple[int, float]:
        if ch in self.vocab.stoi:
            return self.vocab.stoi[ch], 1.0
        if ch in self._cache:
            return self._cache[ch], 1.0
        arr, ok = self.renderer.render(ch)
        if not ok:
            self._cache[ch] = 1
            return 1, 0.0
        with torch.no_grad():
            bmp = torch.from_numpy(arr).unsqueeze(0)
            z = self.cae.encode(bmp.unsqueeze(1))
            p = _blurred_pixel_features(bmp)
            sims = self.cnn_weight * (z @ self.vocab_z.T).squeeze(0) + self.pixel_weight * (
                p @ self.vocab_pix.T
            ).squeeze(0)
            sims = torch.where(self.valid_mask, sims, torch.full_like(sims, -1.0))
            best_sim, best_idx = torch.max(sims, dim=0)
            score = float(best_sim.item())
            idx = int(best_idx.item()) if score >= self.min_sim else 1
        self._cache[ch] = idx
        return idx, score

    def build_oov_map(self, texts: Iterable[str]) -> Dict[str, int]:
        oov_map: Dict[str, int] = {}
        seen = set()
        for t in texts:
            for ch in t:
                if ch not in self.vocab.stoi and ch not in seen:
                    seen.add(ch)
                    idx, _ = self.resolve_char(ch)
                    if idx != 1:
                        oov_map[ch] = idx
        return oov_map
