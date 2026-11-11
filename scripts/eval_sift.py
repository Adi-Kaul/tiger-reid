"""SIFT + RANSAC baseline through the shared evaluator.

    python scripts/eval_sift.py --stage val
    python scripts/eval_sift.py --stage test
    python scripts/eval_sift.py --stage val --limit 10    # timing check
"""

import argparse
import time

import numpy as np

from tiger_reid.metrics import evaluate
from tiger_reid.results import log_result
from tiger_reid.sift_baseline import distance_matrix, sift_features
from tiger_reid.splits import eval_sets, load_splits

CONFIG = {"max_side": 400, "nfeatures": 500, "ratio": 0.75, "ransac": "homography, 8px",
          "score": "RANSAC inliers"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=None, help="only first N queries (timing)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    sets = eval_sets(load_splits(), args.stage)
    t0 = time.time()
    feats = {k: [sift_features(p) for p in v["path"]] for k, v in sets.items() if k != "unknown"}
    print(f"features: {time.time() - t0:.0f}s, median keypoints "
          f"{np.median([len(f[0]) for f in feats['gallery']]):.0f}")

    q, unk_all = feats["query"], feats["unknown_all"]
    if args.limit:
        q, unk_all = q[: args.limit], unk_all[: args.limit]
    t0 = time.time()
    d_q = distance_matrix(q, feats["gallery"], args.jobs)
    d_all = distance_matrix(unk_all, feats["gallery"], args.jobs)
    print(f"matching {len(q) + len(unk_all)} x {len(feats['gallery'])}: {time.time() - t0:.0f}s")
    if args.limit:
        return

    # matched unknowns are a subset of unknown_all: pick their rows
    pos = {f: i for i, f in enumerate(sets["unknown_all"]["filename"])}
    d_unk = d_all[[pos[f] for f in sets["unknown"]["filename"]]]
    m = evaluate(d_q, sets["query"].entity_id.values, sets["gallery"].entity_id.values, d_unk, d_all)
    log_result("SIFT + RANSAC", args.stage, m, CONFIG, seed=0)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    main()
