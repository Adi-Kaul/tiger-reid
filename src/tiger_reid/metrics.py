"""Re-ID metrics. Every method (SIFT, pretrained, trained) is scored with these.

Everything works from a distance matrix, so methods that don't produce
embeddings (SIFT: inlier counts) just pass their own distances (e.g. -inliers).

    dist[i, j] = distance from query i to gallery image j   (lower = more similar)
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def cosine_distance(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """1 - cosine similarity, (Nq, Ng). Inputs needn't be normalized."""
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    g = g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - q @ g.T


def _sorted_matches(dist, q_labels, g_labels):
    """(Nq, Ng) bool: is the gallery item at each rank the right identity?
    Stable sort so ties are broken by gallery order (deterministic)."""
    order = np.argsort(dist, axis=1, kind="stable")
    return np.asarray(g_labels)[order] == np.asarray(q_labels)[:, None]


def rank_k(dist, q_labels, g_labels, ks=(1, 5)) -> dict:
    """Fraction of queries with a correct match among the k nearest gallery items."""
    matches = _sorted_matches(dist, q_labels, g_labels)
    return {f"rank{k}": float(matches[:, :k].any(axis=1).mean()) for k in ks}


def mean_ap(dist, q_labels, g_labels) -> float:
    """Mean average precision. For one query, AP = mean of precision@r over the
    ranks r where a correct gallery item appears. Rewards ranking *all* of a
    tiger's gallery images high, not just the first."""
    matches = _sorted_matches(dist, q_labels, g_labels)
    aps = []
    for row in matches:
        hits = np.flatnonzero(row)
        if len(hits) == 0:
            raise ValueError("query identity missing from gallery (closed-set needs it)")
        precision_at_hits = np.arange(1, len(hits) + 1) / (hits + 1)
        aps.append(precision_at_hits.mean())
    return float(np.mean(aps))


def open_set_auroc(dist_known, dist_unknown) -> float:
    """How well "distance to nearest gallery image" separates new tigers from known
    ones. 1.0 = every new tiger is farther than every known one; 0.5 = coin flip.
    Threshold-free: it's the chance a random unknown query is farther than a
    random known one."""
    s_known = np.min(dist_known, axis=1)
    s_unknown = np.min(dist_unknown, axis=1)
    y = np.r_[np.zeros(len(s_known)), np.ones(len(s_unknown))]
    return float(roc_auc_score(y, np.r_[s_known, s_unknown]))


def bootstrap_ci(dist_known, q_labels, g_labels, dist_unknown=None, n_boot: int = 1000,
                 seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap CIs for rank-1 and AUROC: resample queries (and
    unknowns) with replacement. Shows how much the small test set wobbles."""
    rng = np.random.default_rng(seed)
    nn_correct = _sorted_matches(dist_known, q_labels, g_labels)[:, 0]
    s_known = np.min(dist_known, axis=1)
    s_unk = None if dist_unknown is None else np.min(dist_unknown, axis=1)
    r1, au = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(s_known), len(s_known))
        r1.append(nn_correct[i].mean())
        if s_unk is not None:
            j = rng.integers(0, len(s_unk), len(s_unk))
            y = np.r_[np.zeros(len(i)), np.ones(len(j))]
            au.append(roc_auc_score(y, np.r_[s_known[i], s_unk[j]]))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = {"rank1_lo": float(np.percentile(r1, lo)), "rank1_hi": float(np.percentile(r1, hi))}
    if au:
        out |= {"auroc_lo": float(np.percentile(au, lo)), "auroc_hi": float(np.percentile(au, hi))}
    return out


def evaluate(dist_known, q_labels, g_labels, dist_unknown=None, dist_unknown_all=None,
             ci: bool = True) -> dict:
    """All headline metrics from precomputed distances.
    dist_known: known queries x gallery
    dist_unknown: matched open-set queries x gallery (primary AUROC, D10)
    dist_unknown_all: all open-set images x gallery (secondary `auroc_all`)"""
    out = {**rank_k(dist_known, q_labels, g_labels), "mAP": mean_ap(dist_known, q_labels, g_labels)}
    if dist_unknown is not None:
        out["auroc"] = open_set_auroc(dist_known, dist_unknown)
    if dist_unknown_all is not None:
        out["auroc_all"] = open_set_auroc(dist_known, dist_unknown_all)
    if ci:
        out |= bootstrap_ci(dist_known, q_labels, g_labels, dist_unknown)
    out["n_query"] = len(q_labels)
    out["n_gallery"] = len(g_labels)
    out["n_unknown"] = 0 if dist_unknown is None else len(dist_unknown)
    return out


def evaluate_embeddings(q_emb, q_labels, g_emb, g_labels, unk_emb=None, unk_all_emb=None,
                        ci: bool = True) -> dict:
    """evaluate() for embedding methods, using cosine distance."""
    d = lambda x: None if x is None else cosine_distance(x, g_emb)
    return evaluate(cosine_distance(q_emb, g_emb), q_labels, g_labels, d(unk_emb),
                    d(unk_all_emb), ci)


def evaluate_sets(emb: dict, sets: dict, ci: bool = True) -> dict:
    """evaluate_embeddings() on the dicts from splits.eval_sets() + matching embeddings."""
    return evaluate_embeddings(emb["query"], sets["query"].entity_id.values, emb["gallery"],
                               sets["gallery"].entity_id.values, emb.get("unknown"),
                               emb.get("unknown_all"), ci)
