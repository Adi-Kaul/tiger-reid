"""Frozen train/val/test/open-set splits (see docs/decisions.md D4-D6).

Rules:
- 20% of entities -> open-set (never trained on), half `openset_val`, half `openset_test`.
- Known entities are split by pseudo-clip:
    The largest clip always stays in train. From the other clips:
    >= 3 clips: least similar -> test, next least similar -> val, rest -> train
       2 clips: the smaller clip -> test
       1 clip : everything -> train
    Ties in similarity go to the smaller clip (keeps more images for training).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tiger_reid.data import REPO

SPLITS_DIR = REPO / "splits"
SPLIT_NAMES = ["train", "val", "test", "openset_val", "openset_test"]


def clip_distinctness(sim: np.ndarray, clip_ids: np.ndarray) -> dict:
    """For each clip: its highest similarity to any image in another clip of the
    same entity. Lower = more visually distinct = safer to put in test."""
    out = {}
    for c in np.unique(clip_ids):
        inside = clip_ids == c
        out[c] = float(sim[np.ix_(inside, ~inside)].max()) if (~inside).any() else -np.inf
    return out


def make_splits(clipped: pd.DataFrame, features: np.ndarray, openset_frac: float = 0.2,
                seed: int = 0) -> pd.DataFrame:
    """Add a `split` column to a frame with entity_id / clip_id (rows aligned with
    `features`, as returned by assign_pseudo_clips)."""
    from tiger_reid.pseudo_clips import cosine_similarity_matrix

    rng = np.random.default_rng(seed)
    entities = np.sort(clipped["entity_id"].unique())
    n_open = round(openset_frac * len(entities))
    open_ents = rng.permutation(entities)[:n_open]
    open_val = set(open_ents[: n_open // 2].tolist())
    open_test = set(open_ents[n_open // 2:].tolist())

    split = np.array(["train"] * len(clipped), dtype=object)
    for ent, idx in clipped.groupby("entity_id").indices.items():
        if ent in open_val:
            split[idx] = "openset_val"
            continue
        if ent in open_test:
            split[idx] = "openset_test"
            continue
        clips = clipped["clip_id"].values[idx]
        if len(np.unique(clips)) < 2:
            continue
        sim = cosine_similarity_matrix(features[idx])
        dist = clip_distinctness(sim, clips)
        sizes = {c: int((clips == c).sum()) for c in dist}
        largest = max(sizes, key=lambda c: (sizes[c], -dist[c], str(c)))
        candidates = [c for c in dist if c != largest]
        ranked = sorted(candidates, key=lambda c: (dist[c], sizes[c], str(c)))
        split[idx[clips == ranked[0]]] = "test"
        if len(ranked) >= 2:
            split[idx[clips == ranked[1]]] = "val"
    out = clipped.copy()
    out["split"] = split
    return out


def load_splits(version: str = "v1", data_root: Path | None = None) -> pd.DataFrame:
    """Read the frozen split file and attach image paths."""
    from tiger_reid.data import ATRW
    df = pd.read_csv(SPLITS_DIR / f"splits_{version}.csv")
    root = Path(data_root) if data_root else ATRW
    df["path"] = [root / "atrw_reid_train" / "train" / f for f in df["filename"]]
    return df
