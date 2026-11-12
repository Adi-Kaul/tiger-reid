"""Classic stripe-matching baseline: SIFT keypoints + ratio test + RANSAC.

Score for a (query, gallery) pair = number of RANSAC inliers: matched keypoints
that agree on one geometric transform. Distance = -score, so it plugs into
metrics.evaluate(). Standard settings, no tuning (Lowe's ratio 0.75).
"""

from multiprocessing import Pool

import cv2
import numpy as np

_GALLERY = None   # set in worker processes


def sift_features(path, max_side: int = 400, nfeatures: int = 500):
    """(keypoint xy (K, 2) float32, descriptors (K, 128) float32) for one image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    scale = max_side / max(img.shape)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kps, desc = cv2.SIFT_create(nfeatures=nfeatures).detectAndCompute(img, None)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([k.pt for k in kps]), desc.astype(np.float32)


def match_score(f1, f2, ratio: float = 0.75, reproj: float = 8.0) -> float:
    """RANSAC inlier count, plus a tiny tie-breaker from the ratio-test match count."""
    (p1, d1), (p2, d2) = f1, f2
    if len(d1) < 2 or len(d2) < 2:
        return 0.0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 4:
        return len(good) * 1e-3
    src = p1[[m.queryIdx for m in good]]
    dst = p2[[m.trainIdx for m in good]]
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj)
    inliers = 0 if mask is None else int(mask.sum())
    return inliers + len(good) * 1e-3


def _init(gallery):
    global _GALLERY
    _GALLERY = gallery


def _row(qf):
    return [match_score(qf, gf) for gf in _GALLERY]


def distance_matrix(q_feats, g_feats, n_jobs: int = 8) -> np.ndarray:
    """(Nq, Ng) distance = -score, computed in parallel over queries."""
    with Pool(n_jobs, initializer=_init, initargs=(g_feats,)) as pool:
        scores = pool.map(_row, q_feats, chunksize=1)
    return -np.asarray(scores, dtype=np.float64)


def _max_row(args):
    qf, cand = args
    return max((match_score(qf, _GALLERY[j]) for j in cand), default=0.0)


def max_inliers_candidates(q_feats, g_feats, candidates, n_jobs: int = 8) -> np.ndarray:
    """For each query, the best RANSAC inlier score over its candidate gallery indices
    (D18 SIFT fusion: candidates = gallery images of the top-k entities by embedding)."""
    with Pool(n_jobs, initializer=_init, initargs=(g_feats,)) as pool:
        out = pool.map(_max_row, list(zip(q_feats, candidates)), chunksize=4)
    return np.asarray(out, dtype=np.float64)


def candidates_from_entities(top_ents, g_labels) -> list[np.ndarray]:
    """Gallery indices belonging to each query's top-k entities."""
    g_labels = np.asarray(g_labels)
    return [np.flatnonzero(np.isin(g_labels, row)) for row in top_ents]
