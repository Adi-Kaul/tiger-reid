"""Build and freeze splits/splits_v1.csv (+ a JSON of the settings and counts).

    python scripts/make_splits.py            # refuses to overwrite an existing version
"""

import argparse
import json

from tiger_reid.data import load_labeled
from tiger_reid.pseudo_clips import assign_pseudo_clips, image_features
from tiger_reid.splits import SPLIT_NAMES, SPLITS_DIR, make_splits

THRESHOLD = 0.4   # docs/decisions.md D4
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_csv = SPLITS_DIR / f"splits_{args.version}.csv"
    if out_csv.exists() and not args.force:
        raise SystemExit(f"{out_csv} exists; splits are frozen. Use a new --version.")

    df = load_labeled()
    feats = image_features(list(df["path"]))
    split = make_splits(assign_pseudo_clips(df, THRESHOLD, feats), feats, seed=SEED)

    SPLITS_DIR.mkdir(exist_ok=True)
    split[["filename", "entity_id", "clip_id", "split"]].to_csv(out_csv, index=False)

    counts = {s: {"images": int((split.split == s).sum()),
                  "entities": int(split.loc[split.split == s, "entity_id"].nunique())}
              for s in SPLIT_NAMES}
    meta = {"version": args.version, "pseudo_clip_threshold": THRESHOLD, "seed": SEED,
            "features": "32x32 mean-centered thumbnails, cosine", "openset_frac": 0.2,
            "test_clip_rule": "least similar clip (decisions D5)", "counts": counts}
    (SPLITS_DIR / f"splits_{args.version}.json").write_text(json.dumps(meta, indent=2))
    for s, c in counts.items():
        print(f"{s:13s} {c['images']:5d} images  {c['entities']:3d} entities")


if __name__ == "__main__":
    main()
