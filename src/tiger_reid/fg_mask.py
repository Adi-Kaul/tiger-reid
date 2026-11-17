"""Tiger foreground masks from pretrained DINOv2 patch features (docs/decisions.md D27).

DINOv2's first PCA component over patch tokens separates the main object from
the background with no labels and no training. We fit the PCA once on random
train-split images, then a patch is "tiger" if its projection is > 0. The mask
is a 32x32 patch grid; callers resize it to whatever resolution they need.

Masks are cached in checkpoints/emb/fg_masks.pkl (gitignored).
"""

import pickle

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from tiger_reid.data import REPO

BACKBONE = "vit_small_patch14_dinov2.lvd142m"
SIZE = 448                      # -> 32 x 32 patches of 14 px
CACHE = REPO / "checkpoints" / "emb" / "fg_masks.pkl"


def clean_mask(raw: np.ndarray, dilate: int = 1) -> np.ndarray:
    """3x3 closing, hole filling, then `dilate` patches of dilation. bool in, bool out."""
    st = np.ones((3, 3), bool)
    m = ndimage.binary_closing(raw, structure=st, border_value=0)
    m = ndimage.binary_fill_holes(m | raw)
    if dilate:
        m = ndimage.binary_dilation(m, structure=st, iterations=dilate)
    return m


def resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a bool grid to (h, w) -> uint8 0/255 (OpenCV mask format)."""
    h, w = hw
    rows = np.minimum((np.arange(h) * mask.shape[0]) // h, mask.shape[0] - 1)
    cols = np.minimum((np.arange(w) * mask.shape[1]) // w, mask.shape[1] - 1)
    return (mask[rows][:, cols] * 255).astype(np.uint8)


class FgMasker:
    def __init__(self, fit_paths, device=None, n_fit: int = 200, seed: int = 0):
        import timm
        from torchvision import transforms as T
        self.device = device or torch.device("cpu")
        self.model = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=SIZE).to(self.device).eval()
        self.tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.store = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
        fit_paths = sorted(map(str, fit_paths))
        key = ("pca", BACKBONE, SIZE, n_fit, seed, len(fit_paths))
        if key not in self.store:
            pick = np.random.default_rng(seed).choice(fit_paths, min(n_fit, len(fit_paths)), replace=False)
            F = np.stack([self._tokens(p) for p in pick])
            X = F.reshape(-1, F.shape[-1])
            mu = X.mean(0)
            v = np.linalg.svd(X - mu, full_matrices=False)[2][0]
            g = int(np.sqrt(F.shape[1]))
            proj = ((F - mu) @ v).reshape(len(F), g, g)
            c = slice(g // 3, g - g // 3)
            if proj[:, c, c].mean() < 0:     # sign: the centre of a crop is tiger
                v = -v
            self.store[key] = (mu, v)
            self._save()
        self.mu, self.v = self.store[key]

    def _save(self):
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.store, open(CACHE, "wb"))

    def _tokens(self, path) -> np.ndarray:
        x = self.tf(Image.open(path).convert("RGB"))[None].to(self.device)
        with torch.inference_mode():
            t = self.model.forward_features(x)[:, self.model.num_prefix_tokens:]
        return t[0].float().cpu().numpy()

    def mask(self, path) -> np.ndarray:
        """(32, 32) bool patch grid, True = tiger."""
        p = str(path)
        if p not in self.store:
            proj = (self._tokens(p) - self.mu) @ self.v
            g = int(np.sqrt(len(proj)))
            self.store[p] = clean_mask(proj.reshape(g, g) > 0)
        return self.store[p]

    def masks(self, paths) -> list[np.ndarray]:
        new = [p for p in map(str, paths) if p not in self.store]
        if new:
            print(f"  foreground masks for {len(new)} images", flush=True)
        out = [self.mask(p) for p in paths]
        if new:
            self._save()
        return out
