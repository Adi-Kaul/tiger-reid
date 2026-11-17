"""D27: foreground-mask post-processing and tiger-only SIFT keypoints."""

import cv2
import numpy as np

from tiger_reid.fg_mask import clean_mask, resize_mask
from tiger_reid.openset import known_accept_rate, threshold_for_false_yes
from tiger_reid.sift_baseline import sift_features


def test_clean_mask_fills_holes_and_dilates():
    raw = np.zeros((12, 12), bool)
    raw[3:9, 3:9] = True
    raw[5, 5] = False                       # hole inside the tiger
    m = clean_mask(raw, dilate=0)
    assert m[5, 5] and m[3:9, 3:9].all() and not m[0, 0]
    d = clean_mask(raw, dilate=1)
    assert d[2, 2] and not d[0, 0]          # grew by one patch


def test_resize_mask_nearest():
    g = np.zeros((4, 4), bool)
    g[:, :2] = True
    m = resize_mask(g, (40, 80))
    assert m.shape == (40, 80) and m.dtype == np.uint8
    assert (m[:, :40] == 255).all() and (m[:, 40:] == 0).all()


def test_sift_keypoints_stay_inside_mask(tmp_path):
    rng = np.random.default_rng(0)
    img = (rng.random((200, 300)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    p = tmp_path / "noise.png"
    cv2.imwrite(str(p), img)
    full, _ = sift_features(p)
    grid = np.zeros((32, 32), bool)
    grid[:, :16] = True                     # left half is "tiger"
    xy, desc = sift_features(p, mask=grid)
    assert len(xy) > 20 and (full[:, 0] >= 150).any()
    assert (xy[:, 0] < 150 + 2).all() and len(desc) == len(xy)


def test_false_yes_operating_point():
    s_new = np.arange(100, dtype=float)        # new tigers: scores 0..99
    s_known = np.arange(-50, 50, dtype=float)  # knowns score lower (look more known)
    thr = threshold_for_false_yes(s_new, 0.10)
    assert np.mean(s_new <= thr) == 0.10
    assert known_accept_rate(s_known, thr) == 0.60
