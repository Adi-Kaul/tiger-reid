"""Open-set scoring: how "new" does a query look? (docs/decisions.md D18)

Every score is "higher = more likely a new tiger", so it goes straight into
metrics.auroc_from_scores. Inputs are distance matrices (query x gallery,
lower = more similar) or embeddings, plus the gallery's entity labels.

S1 nearest  : distance to the single nearest gallery image (the original score).
S2 centroid : distance to the nearest entity *mean* embedding.
S3 margin   : -(d2 - d1), d1 <= d2 = nearest-image distance to the best and
              second-best entities. Known tigers have one clear winner.
"""

import numpy as np


def entity_min_dist(dist: np.ndarray, g_labels) -> tuple[np.ndarray, np.ndarray]:
    """(Nq, E) distance from each query to each entity's nearest gallery image, + entity ids."""
    g_labels = np.asarray(g_labels)
    ents = np.unique(g_labels)
    out = np.empty((dist.shape[0], len(ents)))
    for j, e in enumerate(ents):
        out[:, j] = dist[:, g_labels == e].min(axis=1)
    return out, ents


def nn_score(dist: np.ndarray) -> np.ndarray:
    """S1: distance to the nearest gallery image."""
    return dist.min(axis=1)


def _unit(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def centroid_dist(q_emb, g_emb, g_labels) -> tuple[np.ndarray, np.ndarray]:
    """(Nq, E) cosine distance to each entity's mean (re-normalized) embedding, + entity ids."""
    g_labels = np.asarray(g_labels)
    ents = np.unique(g_labels)
    g = _unit(np.asarray(g_emb, dtype=np.float64))
    cent = _unit(np.stack([g[g_labels == e].mean(axis=0) for e in ents]))
    return 1.0 - _unit(np.asarray(q_emb, dtype=np.float64)) @ cent.T, ents


def centroid_score(q_emb, g_emb, g_labels) -> np.ndarray:
    """S2: distance to the nearest entity centroid."""
    return centroid_dist(q_emb, g_emb, g_labels)[0].min(axis=1)


def margin_score(ent_dist: np.ndarray) -> np.ndarray:
    """S3: -(second-best entity distance - best entity distance). Ambiguous = high."""
    two = np.sort(ent_dist, axis=1)[:, :2]
    return -(two[:, 1] - two[:, 0])


def topk_entities(ent_dist: np.ndarray, ents: np.ndarray, k: int = 10) -> np.ndarray:
    """(Nq, k) entity ids of the k nearest entities per query."""
    return ents[np.argsort(ent_dist, axis=1, kind="stable")[:, :k]]


def zscore_params(*arrays) -> tuple[float, float]:
    """Mean / std over a reference pool (e.g. val known + unknown scores)."""
    x = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays])
    return float(x.mean()), float(max(x.std(), 1e-12))


def zscore(x, params) -> np.ndarray:
    mu, sd = params
    return (np.asarray(x, dtype=np.float64) - mu) / sd


def known_threshold(s_known, keep: float = 0.95) -> float:
    """Score threshold that still accepts `keep` of the known queries as known
    (score <= threshold -> "known tiger"). Fixed on val knowns, never on unknowns."""
    return float(np.quantile(np.asarray(s_known, dtype=np.float64), keep))


def false_known_rate(s_new, threshold: float) -> float:
    """Share of truly-new tigers wrongly accepted as known (score <= threshold)."""
    return float(np.mean(np.asarray(s_new, dtype=np.float64) <= threshold))


def fuse_scores(base, spec: dict, zp: dict, nearest=None, log_inl=None, log_lg=None) -> np.ndarray:
    """Final novelty score (D18/D24) from its parts.

    base    : the spec's embedding score (nearest / centroid / margin), raw.
    nearest : nearest-image distance, used when spec["near_a"] is set (D24 M+N).
    log_inl : log1p(max SIFT inliers in the top-10 entities), used when spec["sift_w"] is set.
    log_lg  : log1p(max DISK+LightGlue inliers, D25), used when spec["lg_w"] is set.
    zp      : z-score params {"emb", "near", "sift", "lg"} fitted on val known + unknown.
    With neither extra part, the raw base score is returned (original D18 behaviour).
    """
    a, w, lw = spec.get("near_a"), spec.get("sift_w"), spec.get("lg_w")
    if a is None and w is None and lw is None:
        return np.asarray(base, dtype=np.float64)
    s = zscore(base, zp["emb"])
    if a is not None:
        s = s + a * zscore(nearest, zp["near"])
    if w is not None:
        s = s - w * zscore(log_inl, zp["sift"])
    if lw is not None:
        s = s - lw * zscore(log_lg, zp["lg"])
    return s


def top_images_per_entity(img_dist: np.ndarray, g_labels, top_ents: np.ndarray, n: int = 5) -> list[np.ndarray]:
    """D25 candidates: for each query, the n gallery images (nearest by embedding)
    of each of its top entities. -> list of gallery index arrays."""
    g_labels = np.asarray(g_labels)
    out = []
    for i, ents in enumerate(top_ents):
        idx = []
        for e in ents:
            members = np.flatnonzero(g_labels == e)
            idx.extend(members[np.argsort(img_dist[i, members], kind="stable")[:n]])
        out.append(np.asarray(idx, dtype=int))
    return out
