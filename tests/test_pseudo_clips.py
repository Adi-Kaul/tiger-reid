import numpy as np
from PIL import Image

from tiger_reid.pseudo_clips import (cosine_similarity_matrix, group_by_threshold,
                                     image_features)


def test_cosine_matrix_basic():
    f = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    s = cosine_similarity_matrix(f)
    assert np.allclose(np.diag(s), 1.0)
    assert np.allclose(s, s.T)
    assert np.isclose(s[0, 1], 1.0)   # same direction, different length
    assert np.isclose(s[0, 2], 0.0)   # orthogonal


def test_grouping_chains_through_middle_image():
    # A~B and B~C above threshold, A~C below: all three must be one clip.
    sim = np.array([[1.0, 0.9, 0.5, 0.0],
                    [0.9, 1.0, 0.9, 0.0],
                    [0.5, 0.9, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    labels = group_by_threshold(sim, 0.8)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_threshold_extremes():
    sim = np.array([[1.0, 0.6], [0.6, 1.0]])
    assert len(set(group_by_threshold(sim, 0.5))) == 1
    assert len(set(group_by_threshold(sim, 0.99))) == 2


def test_image_features_shape(tmp_path):
    paths = []
    for i, size in enumerate([(50, 80), (300, 120)]):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", size, (i * 100, 50, 50)).save(p)
        paths.append(p)
    f = image_features(paths, size=16, center=False)
    assert f.shape == (2, 16 * 16 * 3)
    assert f.min() >= 0 and f.max() <= 1
    fc = image_features(paths, size=16)          # centered: each row has mean 0
    assert np.allclose(fc.mean(axis=1), 0, atol=1e-6)


def test_assign_pseudo_clips_stays_within_entity():
    import pandas as pd
    from tiger_reid.pseudo_clips import assign_pseudo_clips
    df = pd.DataFrame({"entity_id": [1, 1, 2, 2], "path": ["a", "b", "c", "d"]})
    feats = np.array([[1.0, 0], [1.0, 0], [1.0, 0], [0, 1.0]])  # img 0,1,2 identical
    out = assign_pseudo_clips(df, 0.5, features=feats)
    assert out.clip_id[0] == out.clip_id[1]
    assert out.clip_id[2] != out.clip_id[0]      # identical image, other entity
    assert out.clip_id[2] != out.clip_id[3]
