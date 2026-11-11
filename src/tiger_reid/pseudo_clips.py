"""Reconstruct pseudo-clips: groups of near-duplicate images within one entity.

Why: ATRW lost its video/clip info, and frames from one video are near-duplicates.
Splitting by pseudo-clip (not by frame) keeps near-duplicates on the same side of
the train/test line. See docs/learning/01-pseudo-clips.md.

Pipeline:
    features = image_features(paths)              # one vector per image
    sim      = cosine_similarity_matrix(features)  # N x N, within ONE entity
    clip_ids = group_by_threshold(sim, threshold)  # e.g. [0, 0, 1, 0, 2, ...]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def image_features(paths: list[Path], size: int = 32, center: bool = True) -> np.ndarray:
    """Return an (N, size*size*3) array: one tiny-thumbnail vector per image.

    A 32x32 thumbnail throws away fine detail (stripes) and keeps the scene
    layout: pose, background, lighting. That is what "same video" looks like.

    center=True subtracts each image's mean. Raw pixels are all >= 0, so every
    pair has cosine ~0.75-0.97 and same/different scenes overlap (checked Nov 10:
    within-entity median 0.885 vs between 0.873). Centered: 0.38 vs 0.09.
    """
    feats = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        feats.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
    f = np.stack(feats)
    if center:
        f = f - f.mean(axis=1, keepdims=True)
    return f


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """(N, N) matrix of cosine similarities between rows."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    unit = features / np.clip(norms, 1e-12, None)
    return unit @ unit.T


def group_by_threshold(sim: np.ndarray, threshold: float) -> np.ndarray:
    """(N,) clip ids: i and j share a clip if linked by a chain of pairs with
    sim > threshold (connected components). Chaining means A~B~C is one clip
    even when A and C differ, e.g. a tiger walking across the frame."""
    adjacency = csr_matrix(sim > threshold)
    _, labels = connected_components(adjacency, directed=False)
    return labels


def assign_pseudo_clips(df, threshold: float, features: np.ndarray | None = None):
    """Add a global `clip_id` column ("<entity>_<k>") to a load_labeled() frame.
    Grouping is done within each entity only."""
    if features is None:
        features = image_features(list(df["path"]))
    clip = np.empty(len(df), dtype=object)
    for ent, idx in df.groupby("entity_id").indices.items():
        labels = group_by_threshold(cosine_similarity_matrix(features[idx]), threshold)
        clip[idx] = [f"{ent}_{k}" for k in labels]
    out = df.copy()
    out["clip_id"] = clip
    return out


# ---------------------------------------------------------------- plumbing


def show_clips(paths: list[Path], clip_ids: np.ndarray, max_per_clip: int = 8,
               title: str = "") -> plt.Figure:
    """One row per pseudo-clip, up to `max_per_clip` thumbnails each."""
    clips = sorted(set(clip_ids.tolist()), key=lambda c: -(clip_ids == c).sum())
    fig, axes = plt.subplots(len(clips), max_per_clip,
                             figsize=(max_per_clip * 1.6, len(clips) * 1.4),
                             squeeze=False)
    for r, c in enumerate(clips):
        members = [p for p, k in zip(paths, clip_ids) if k == c]
        for col in range(max_per_clip):
            ax = axes[r, col]
            ax.axis("off")
            if col < len(members):
                ax.imshow(Image.open(members[col]).convert("RGB"))
        axes[r, 0].set_title(f"clip {c} ({len(members)} imgs)", fontsize=8, loc="left")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
