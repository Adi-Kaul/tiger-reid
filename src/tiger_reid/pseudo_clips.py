"""Reconstruct pseudo-clips: groups of near-duplicate images within one entity.

Why: ATRW lost its video/clip info, and frames from one video are near-duplicates.
Splitting by pseudo-clip (not by frame) keeps near-duplicates on the same side of
the train/test line. See docs/learning/01-pseudo-clips.md.

YOUR PART: the three TODO functions. The plotting helper at the bottom is done.

Pipeline:
    features = image_features(paths)              # one vector per image
    sim      = cosine_similarity_matrix(features)  # N x N, within ONE entity
    clip_ids = group_by_threshold(sim, threshold)  # e.g. [0, 0, 1, 0, 2, ...]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def image_features(paths: list[Path], size: int = 32) -> np.ndarray:
    """Return an (N, D) float array: one feature vector per image.

    Start with the dumbest thing that could work: a tiny thumbnail.
      - open each image, convert to RGB
      - resize to (size, size)  -> throws away fine detail (stripes) and keeps
        the overall layout: pose, background, lighting. That is exactly
        what "same video" looks like.
      - flatten to a vector, scale to [0, 1]

    Hint: np.asarray(img, dtype=np.float32) gives (H, W, 3).
    Later (optional) compare with a pretrained CNN's features and see which one
    groups better. Don't start there.
    """
    # TODO
    raise NotImplementedError


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Return an (N, N) matrix where [i, j] = cosine similarity of rows i and j.

    Hint: L2-normalize each row (divide by its norm), then one matrix multiply.
    Sanity checks: diagonal is all 1.0, matrix is symmetric.
    Question for you: for thumbnail pixels (all >= 0), what's the lowest
    value this can be? Will that make a threshold easy or hard to pick?
    """
    # TODO
    raise NotImplementedError


def group_by_threshold(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Return an (N,) int array of clip ids: images i, j share a clip if they are
    connected by a chain of pairs with sim > threshold.

    "Chain" matters: if A~B and B~C, then A, B, C are one clip even if A and C
    are not similar enough on their own (the tiger walked across the frame).

    This is "connected components" of a graph where edges = (sim > threshold).
    Hint: scipy.sparse.csgraph.connected_components does it in one call, but
    try writing it yourself first (BFS/DFS over the adjacency matrix, ~10 lines).
    """
    # TODO
    raise NotImplementedError


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
