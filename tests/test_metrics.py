"""Metrics checked against small examples worked out by hand (docs/learning/02-metrics.md)."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from tiger_reid.metrics import (cosine_distance, evaluate, evaluate_embeddings, mean_ap,
                                open_set_auroc, rank_k)

# Gallery: A, A, B.  Query 0 is A, query 1 is B.
G = np.array(["A", "A", "B"])
Q = np.array(["A", "B"])
D = np.array([[0.1, 0.5, 0.3],    # ranking: A(hit) B A(hit)  -> AP = (1/1 + 2/3)/2 = 0.8333
              [0.2, 0.3, 0.9]])   # ranking: A A B(hit)       -> AP = 1/3


def test_rank_k_hand_example():
    r = rank_k(D, Q, G, ks=(1, 2, 3))
    assert r == {"rank1": 0.5, "rank2": 0.5, "rank3": 1.0}


def test_rank_k_larger_than_gallery():
    assert rank_k(D, Q, G, ks=(5,))["rank5"] == 1.0


def test_mean_ap_hand_example():
    assert mean_ap(D, Q, G) == pytest.approx((0.8333333 + 0.3333333) / 2)


def test_mean_ap_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 5, 40)
    q = rng.integers(0, 5, 15)
    d = rng.random((15, 40))
    ref = np.mean([average_precision_score(g == q[i], -d[i]) for i in range(15)])
    assert mean_ap(d, q, g) == pytest.approx(ref)


def test_perfect_ranking():
    d = np.array([[0.0, 0.1, 0.9], [0.9, 0.8, 0.0]])
    assert mean_ap(d, Q, G) == 1.0
    assert rank_k(d, Q, G)["rank1"] == 1.0


def test_missing_identity_raises():
    with pytest.raises(ValueError):
        mean_ap(D, np.array(["A", "C"]), G)


def test_auroc_hand_example():
    # nearest-gallery distance: known 0.1, 0.2; unknown 0.3, 0.05
    # unknown farther than known in 2 of 4 pairs -> 0.5
    known = np.array([[0.1, 0.4], [0.2, 0.9]])
    unknown = np.array([[0.3, 0.7], [0.05, 0.6]])
    assert open_set_auroc(known, unknown) == 0.5


def test_auroc_perfect_and_inverted():
    known = np.array([[0.1], [0.2]])
    far = np.array([[0.5], [0.6]])
    assert open_set_auroc(known, far) == 1.0
    assert open_set_auroc(far, known) == 0.0


def test_auroc_random_is_about_half():
    rng = np.random.default_rng(0)
    assert open_set_auroc(rng.random((2000, 5)), rng.random((2000, 5))) == pytest.approx(0.5, abs=0.03)


def test_cosine_distance():
    q = np.array([[1.0, 0.0], [0.0, 2.0]])
    g = np.array([[3.0, 0.0], [-1.0, 0.0]])
    assert np.allclose(cosine_distance(q, g), [[0.0, 2.0], [1.0, 1.0]])


def test_evaluate_embeddings_end_to_end():
    # Two identities on orthogonal axes; queries near their own; unknown on a third axis.
    g = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=float)
    q = np.array([[1, 0.05, 0], [0.05, 1, 0]], dtype=float)
    unk = np.array([[0, 0, 1]], dtype=float)
    m = evaluate_embeddings(q, Q, g, G, unk, unk)
    assert m["rank1"] == 1.0 and m["mAP"] == 1.0 and m["auroc"] == 1.0 == m["auroc_all"]
    assert m["rank1_lo"] == m["rank1_hi"] == 1.0
    assert (m["n_query"], m["n_gallery"], m["n_unknown"]) == (2, 3, 1)
    assert "auroc" not in evaluate(D, Q, G)


def test_eval_sets_on_real_splits():
    from tiger_reid.splits import SPLITS_DIR, eval_sets
    import pandas as pd
    s = pd.read_csv(SPLITS_DIR / "splits_v1.csv")
    for stage in ["val", "test"]:
        e = eval_sets(s, stage)
        assert set(e["query"].entity_id) <= set(e["gallery"].entity_id)          # closed set
        assert not set(e["unknown"].entity_id) & set(e["gallery"].entity_id)     # open set
        assert not set(e["query"].clip_id) & set(e["gallery"].clip_id)           # no clip leak
        assert set(e["unknown"].filename) < set(e["unknown_all"].filename)
        # matched unknowns never come from an entity's largest clip
        sizes = e["unknown_all"].groupby("clip_id").size()
        for ent, g in e["unknown"].groupby("entity_id"):
            all_clips = e["unknown_all"][e["unknown_all"].entity_id == ent].clip_id.unique()
            assert sizes[g.clip_id.iloc[0]] <= sizes[all_clips].max()
            assert sizes[all_clips].idxmax() not in set(g.clip_id) or (sizes[all_clips] == sizes[all_clips].max()).sum() > 1


def test_log_result(tmp_path):
    import pandas as pd
    from tiger_reid.results import log_result
    p = tmp_path / "r.csv"
    log_result("a", "val", {"rank1": 0.123456}, {"lr": 1e-3}, seed=0, path=p)
    log_result("b", "test", {"rank1": 0.5, "mAP": 0.4}, path=p)
    df = pd.read_csv(p)
    assert list(df.method) == ["a", "b"] and df.rank1[0] == 0.1235


def test_bootstrap_ci_brackets_point_estimate():
    from tiger_reid.metrics import bootstrap_ci
    rng = np.random.default_rng(1)
    g = rng.integers(0, 5, 50); q = rng.integers(0, 5, 40)
    d, du = rng.random((40, 50)), rng.random((30, 50)) + 0.1
    ci = bootstrap_ci(d, q, g, du, n_boot=300)
    r1 = rank_k(d, q, g)["rank1"]; au = open_set_auroc(d, du)
    assert ci["rank1_lo"] <= r1 <= ci["rank1_hi"]
    assert ci["auroc_lo"] <= au <= ci["auroc_hi"]
    assert ci["auroc_hi"] - ci["auroc_lo"] > 0.05      # small sets -> wide interval


def test_largest_clip_mask():
    import pandas as pd
    from tiger_reid.splits import largest_clip_mask
    df = pd.DataFrame({"entity_id": [1, 1, 1, 2, 2], "clip_id": ["1_0", "1_1", "1_1", "2_0", "2_1"]})
    assert list(largest_clip_mask(df)) == [False, True, True, True, False]
