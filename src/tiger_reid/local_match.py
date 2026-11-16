"""Learned local matching for the open-set re-check (docs/decisions.md D25).

DISK keypoints + LightGlue matches (kornia pretrained weights), then RANSAC
homography inliers, the same geometric check as the SIFT baseline. Score for a
(query, gallery) pair = inlier count; more inliers = more likely the same stripes.

Features are cached per image in checkpoints/emb/disk/ (gitignored).
"""

import hashlib
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from tiger_reid.data import REPO

CACHE = REPO / "checkpoints" / "emb" / "disk"


def ransac_inliers(src: np.ndarray, dst: np.ndarray, reproj: float = 8.0) -> int:
    """Matched point pairs that agree on one homography (0 if fewer than 4 matches)."""
    if len(src) < 4:
        return 0
    _, mask = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, reproj)
    return 0 if mask is None else int(mask.sum())


class DiskLightGlue:
    def __init__(self, device=None, n_kpts: int = 1024, max_side: int = 512):
        import kornia.feature as KF
        warnings.filterwarnings("ignore", category=FutureWarning, module="kornia")
        self.KF, self.n, self.max_side = KF, n_kpts, max_side
        self.device = device or torch.device("cpu")
        self.disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
        self.lg = KF.LightGlueMatcher("disk").to(self.device).eval()
        self._mem: dict[str, tuple] = {}
        CACHE.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, path) -> Path:
        st = Path(path).stat()
        key = f"{Path(path).resolve()}|{st.st_size}|{int(st.st_mtime)}|{self.n}|{self.max_side}"
        return CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".npz")

    def features(self, path) -> tuple:
        """(keypoints (K, 2) float32, descriptors (K, 128) float16, (h, w))."""
        path = str(path)
        if path in self._mem:
            return self._mem[path]
        f = self._cache_file(path)
        if f.exists():
            z = np.load(f)
            out = (z["k"], z["d"], tuple(z["hw"]))
        else:
            im = Image.open(path).convert("RGB")
            im.thumbnail((self.max_side, self.max_side))
            x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float()[None] / 255
            with torch.inference_mode():
                feat = self.disk(x.to(self.device), n=self.n, pad_if_not_divisible=True)[0]
            out = (feat.keypoints.cpu().numpy().astype(np.float32),
                   feat.descriptors.cpu().numpy().astype(np.float16), (im.height, im.width))
            np.savez(f, k=out[0], d=out[1], hw=np.array(out[2]))
        self._mem[path] = out
        return out

    def _t(self, a):
        return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.device)

    def inliers(self, fa, fb) -> int:
        (ka, da, ha), (kb, db, hb) = fa, fb
        if len(ka) < 4 or len(kb) < 4:
            return 0
        KF = self.KF
        lafa = KF.laf_from_center_scale_ori(self._t(ka)[None], torch.ones(1, len(ka), 1, 1, device=self.device))
        lafb = KF.laf_from_center_scale_ori(self._t(kb)[None], torch.ones(1, len(kb), 1, 1, device=self.device))
        with torch.inference_mode():
            _, idx = self.lg(self._t(da), self._t(db), lafa, lafb,
                             hw1=torch.tensor(ha, device=self.device), hw2=torch.tensor(hb, device=self.device))
        idx = idx.cpu().numpy()
        return ransac_inliers(ka[idx[:, 0]], kb[idx[:, 1]]) if len(idx) else 0

    def max_inliers_candidates(self, q_paths, g_paths, candidates, log_every: int = 50) -> np.ndarray:
        """For each query, max inliers over its candidate gallery indices."""
        out = np.zeros(len(q_paths))
        for i, (qp, cand) in enumerate(zip(q_paths, candidates)):
            fq = self.features(qp)
            out[i] = max((self.inliers(fq, self.features(g_paths[j])) for j in cand), default=0)
            if log_every and (i + 1) % log_every == 0:
                print(f"    LightGlue {i + 1}/{len(q_paths)}", flush=True)
        return out
