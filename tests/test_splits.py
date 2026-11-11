"""Leakage guards on the real frozen split file, plus a toy test of the rules."""

import numpy as np
import pandas as pd
import pytest

from tiger_reid.splits import SPLITS_DIR, make_splits

SPLIT_FILE = SPLITS_DIR / "splits_v1.csv"


@pytest.fixture(scope="module")
def splits():
    if not SPLIT_FILE.exists():
        pytest.skip("splits_v1.csv not built yet")
    return pd.read_csv(SPLIT_FILE)


def test_every_labeled_image_once(splits):
    assert len(splits) == 1887
    assert splits["filename"].is_unique
    assert splits["entity_id"].nunique() == 107


def test_no_clip_in_two_splits(splits):
    assert (splits.groupby("clip_id")["split"].nunique() == 1).all()


def test_openset_entities_never_in_known_splits(splits):
    open_ents = set(splits.loc[splits.split.str.startswith("openset"), "entity_id"])
    known_ents = set(splits.loc[~splits.split.str.startswith("openset"), "entity_id"])
    assert open_ents and not (open_ents & known_ents)
    ov = set(splits.loc[splits.split == "openset_val", "entity_id"])
    ot = set(splits.loc[splits.split == "openset_test", "entity_id"])
    assert ov and ot and not (ov & ot)


def test_test_and_val_entities_have_training_images(splits):
    train_ents = set(splits.loc[splits.split == "train", "entity_id"])
    for s in ["val", "test"]:
        assert set(splits.loc[splits.split == s, "entity_id"]) <= train_ents


def test_test_images_not_near_duplicates_of_train(splits):
    """Recompute features: no test image may be > threshold-similar to a train
    image of the same entity (that's what makes them different pseudo-clips)."""
    from tiger_reid.data import TRAIN_IMG_DIR
    from tiger_reid.pseudo_clips import cosine_similarity_matrix, image_features
    if not TRAIN_IMG_DIR.exists():
        pytest.skip("ATRW images not downloaded")
    for ent in splits.loc[splits.split == "test", "entity_id"].unique():
        sub = splits[(splits.entity_id == ent) & splits.split.isin(["train", "test"])]
        sim = cosine_similarity_matrix(image_features([TRAIN_IMG_DIR / f for f in sub.filename]))
        is_test = (sub.split == "test").values
        assert sim[np.ix_(is_test, ~is_test)].max() <= 0.4


def test_rules_on_toy_data():
    # entity 0: 3 clips, entity 1: 2 clips, entity 2: 1 clip, entities 3-9 filler
    rows, feats = [], []
    def add(ent, clip, vec):
        rows.append({"entity_id": ent, "clip_id": f"{ent}_{clip}"}); feats.append(vec)
    add(0, 0, [1, 0, 0]); add(0, 0, [1, 0, 0])
    add(0, 1, [0.9, 0.1, 0])       # close to clip 0
    add(0, 2, [0, 0, 1])           # most distinct -> test
    add(1, 0, [1, 0, 0]); add(1, 1, [0, 1, 0])
    add(2, 0, [1, 0, 0])
    for e in range(3, 10):
        add(e, 0, [1, 1, 1])
    df = pd.DataFrame(rows)
    # openset_frac=0 so the rules on entities 0-2 are tested deterministically
    out = make_splits(df, np.array(feats, dtype=float), openset_frac=0.0)
    s = dict(zip(df.index, out.split))
    assert s[3] == "test" and s[2] == "val" and s[0] == s[1] == "train"
    assert sorted([s[4], s[5]]) == ["test", "train"]
    assert s[6] == "train"
