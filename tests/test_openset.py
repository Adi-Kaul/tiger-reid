"""Open-set scoring (D18) on toy data where the right answer is obvious."""

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms as T

from tiger_reid.dataset import tta_transforms
from tiger_reid.metrics import auroc_from_scores, cosine_distance, open_set_auroc
from tiger_reid.openset import (centroid_dist, centroid_score, entity_min_dist, margin_score,
                                nn_score, topk_entities, zscore, zscore_params)
from tiger_reid.sift_baseline import candidates_from_entities


@pytest.fixture
def toy():
    """3 gallery entities on separate axes; known queries sit on an axis, unknowns between."""
    rng = np.random.default_rng(0)
    axes = np.eye(3, 8)
    g = np.repeat(axes, 4, axis=0) + 0.05 * rng.standard_normal((12, 8))
    g_lab = np.repeat([10, 20, 30], 4)
    known = axes + 0.05 * rng.standard_normal((3, 8))
    unk = np.stack([axes[0] + axes[1], axes[1] + axes[2], np.eye(8)[5]]) + 0.05 * rng.standard_normal((3, 8))
    return g, g_lab, known, unk


def test_auroc_from_scores_matches_old(toy):
    g, g_lab, known, unk = toy
    dk, du = cosine_distance(known, g), cosine_distance(unk, g)
    assert auroc_from_scores(nn_score(dk), nn_score(du)) == open_set_auroc(dk, du)


def test_all_scores_separate_obvious_case(toy):
    g, g_lab, known, unk = toy
    dk, du = cosine_distance(known, g), cosine_distance(unk, g)
    ek, ents = entity_min_dist(dk, g_lab)
    eu, _ = entity_min_dist(du, g_lab)
    assert list(ents) == [10, 20, 30]
    assert auroc_from_scores(nn_score(dk), nn_score(du)) == 1.0
    assert auroc_from_scores(centroid_score(known, g, g_lab), centroid_score(unk, g, g_lab)) == 1.0
    # margin: queries between two tigers are ambiguous; the off-axis one is far from all,
    # so margin alone can miss it -> just check the two "between" unknowns beat every known
    mk, mu = margin_score(ek), margin_score(eu)
    assert mu[:2].min() > mk.max()


def test_entity_min_dist_and_centroid_shapes(toy):
    g, g_lab, known, _ = toy
    d = cosine_distance(known, g)
    ed, ents = entity_min_dist(d, g_lab)
    assert ed.shape == (3, 3)
    assert np.allclose(ed.min(axis=1), d.min(axis=1))
    cd, _ = centroid_dist(known, g, g_lab)
    assert cd.shape == (3, 3) and (cd.argmin(axis=1) == [0, 1, 2]).all()


def test_margin_sign():
    ed = np.array([[0.1, 0.9, 0.95], [0.5, 0.52, 0.9]])   # clear winner vs ambiguous
    m = margin_score(ed)
    assert m[1] > m[0]


def test_topk_candidates():
    ed = np.array([[0.3, 0.1, 0.2]])
    top = topk_entities(ed, np.array([10, 20, 30]), k=2)
    assert list(top[0]) == [20, 30]
    cand = candidates_from_entities(top, np.array([10, 10, 20, 30, 30]))
    assert list(cand[0]) == [2, 3, 4]


def test_zscore():
    p = zscore_params([1, 2, 3], [4, 5])
    z = zscore([1, 2, 3, 4, 5], p)
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9


def test_tta_views_no_flips_and_shapes():
    views = tta_transforms((56, 84))
    assert len(views) == 5
    img = Image.new("RGB", (120, 80), (10, 20, 30))
    for tf in views:
        assert not any(isinstance(t, (T.RandomHorizontalFlip, T.RandomVerticalFlip))
                       for t in tf.transforms)
        assert tuple(tf(img).shape) == (3, 56, 84)


def test_known_threshold_and_false_known_rate():
    from tiger_reid.openset import false_known_rate, known_threshold
    known = np.arange(100, dtype=float)          # scores 0..99
    thr = known_threshold(known, 0.95)
    assert np.mean(known <= thr) == pytest.approx(0.95, abs=0.01)
    assert false_known_rate(np.array([200.0, 300.0]), thr) == 0.0   # clearly new -> rejected
    assert false_known_rate(np.array([1.0, 200.0]), thr) == 0.5     # one looks known


def test_fuse_scores_parts():
    from tiger_reid.openset import fuse_scores
    base, near, inl = np.array([0.0, 2.0]), np.array([1.0, 3.0]), np.array([4.0, 0.0])
    zp = {"emb": (1.0, 1.0), "near": (2.0, 1.0), "sift": (2.0, 2.0)}
    plain = {"score": "margin", "sift_w": None}
    assert np.allclose(fuse_scores(base, plain, zp), base)                     # raw, unchanged
    s = fuse_scores(base, {"sift_w": 1.0}, zp, log_inl=inl)
    assert np.allclose(s, [-1 - 1, 1 + 1])                                     # many inliers -> known
    s = fuse_scores(base, {"near_a": 2.0, "sift_w": 0.5}, zp, nearest=near, log_inl=inl)
    assert np.allclose(s, [-1 + 2 * -1 - 0.5 * 1, 1 + 2 * 1 - 0.5 * -1])       # far from gallery -> new


def test_top_images_per_entity_and_lg_term():
    from tiger_reid.openset import fuse_scores, top_images_per_entity
    g_lab = np.array([1, 1, 1, 2, 2, 3])
    dist = np.array([[0.9, 0.1, 0.5, 0.3, 0.2, 0.0]])
    cand = top_images_per_entity(dist, g_lab, np.array([[2, 1]]), n=2)
    assert list(cand[0]) == [4, 3, 1, 2]        # entity 2 nearest-first, then entity 1's best two
    zp = {"emb": (0.0, 1.0), "lg": (1.0, 1.0)}
    s = fuse_scores(np.array([0.0, 0.0]), {"lg_w": 2.0}, zp, log_lg=np.array([3.0, 0.0]))
    assert np.allclose(s, [-4.0, 2.0])          # many LightGlue inliers -> known


def test_ransac_inliers_on_exact_homography():
    from tiger_reid.local_match import ransac_inliers
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (30, 2))
    dst = src * 1.5 + 7                          # pure scale + shift: all inliers
    dst_bad = dst.copy(); dst_bad[:10] = rng.uniform(0, 150, (10, 2))
    assert ransac_inliers(src, dst) == 30
    assert 18 <= ransac_inliers(src, dst_bad) <= 21
    assert ransac_inliers(src[:3], dst[:3]) == 0
