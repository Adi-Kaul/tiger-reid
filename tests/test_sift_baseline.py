import cv2
import numpy as np

from tiger_reid.sift_baseline import distance_matrix, match_score, sift_features


def _texture(seed, size=300):
    """Blurred noise: lots of SIFT keypoints, unique per seed (a fake stripe pattern)."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (size, size)).astype(np.uint8), (5, 5), 2)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def _save(tmp_path, name, img):
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


def test_same_pattern_scores_higher_than_different(tmp_path):
    a = _texture(0)
    M = cv2.getRotationMatrix2D((150, 150), 8, 1.1)                    # rotate 8°, zoom 10%
    a_moved = cv2.warpAffine(a, M, (300, 300), borderMode=cv2.BORDER_REFLECT)
    fa, fa2, fb = (sift_features(_save(tmp_path, n, i)) for n, i in
                   [("a.png", a), ("a2.png", a_moved), ("b.png", _texture(1))])
    same, diff = match_score(fa, fa2), match_score(fa, fb)
    assert same > 20 and diff < 10 and same > 3 * diff


def test_resize_cap_and_empty_image(tmp_path):
    big = sift_features(_save(tmp_path, "big.png", cv2.resize(_texture(2), (1200, 900))))
    assert big[0].max() <= 400                                          # coordinates in resized image
    blank = sift_features(_save(tmp_path, "blank.png", np.full((100, 100), 128, np.uint8)))
    assert blank[1].shape == (0, 128)
    assert match_score(blank, big) == 0.0


def test_distance_matrix_parallel(tmp_path):
    feats = [sift_features(_save(tmp_path, f"{s}.png", _texture(s))) for s in range(3)]
    d = distance_matrix(feats[:2], feats, n_jobs=2)
    assert d.shape == (2, 3)
    assert d[0].argmin() == 0 and d[1].argmin() == 1                   # self is nearest
