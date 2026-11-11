"""Sweep pseudo-clip thresholds and save grids to choose one by eye.

    python scripts/make_pseudo_clips.py                 # sweep + grids
    python scripts/make_pseudo_clips.py --threshold 0.6  # also print chosen stats

Writes results/pseudo_clips/sweep.csv and grid PNGs for a few sample entities.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tiger_reid.data import REPO, load_labeled
from tiger_reid.pseudo_clips import assign_pseudo_clips, image_features, show_clips

OUT = REPO / "results" / "pseudo_clips"
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
SAMPLE_ENTITIES = 4


def clip_stats(clipped: pd.DataFrame) -> dict:
    per_ent = clipped.groupby("entity_id")["clip_id"].nunique()
    biggest = (clipped.groupby(["entity_id", "clip_id"]).size()
               .groupby("entity_id").max() / clipped.groupby("entity_id").size())
    return {
        "clips_total": clipped["clip_id"].nunique(),
        "clips_per_entity_median": per_ent.median(),
        "entities_1_clip": int((per_ent == 1).sum()),
        "entities_2_clips": int((per_ent == 2).sum()),
        "entities_3plus_clips": int((per_ent >= 3).sum()),
        "singleton_clips": int((clipped["clip_id"].value_counts() == 1).sum()),
        "biggest_clip_frac_median": round(biggest.median(), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    ap.add_argument("--grid-thresholds", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_labeled()
    feats = image_features(list(df["path"]))

    rows = []
    for t in args.thresholds:
        rows.append({"threshold": t, **clip_stats(assign_pseudo_clips(df, t, feats))})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "sweep.csv", index=False)
    print(sweep.to_string(index=False))

    # Sample entities: two typical-size, the largest, and one random.
    sizes = df["entity_id"].value_counts()
    sample = list(sizes.index[:1]) + list(
        sizes[sizes <= sizes.median()].sample(SAMPLE_ENTITIES - 1, random_state=args.seed).index)
    for t in args.grid_thresholds:
        clipped = assign_pseudo_clips(df, t, feats)
        for ent in sample:
            sub = clipped[clipped["entity_id"] == ent]
            codes = pd.factorize(sub["clip_id"])[0]
            fig = show_clips(list(sub["path"]), codes, max_per_clip=8,
                             title=f"entity {ent}, threshold {t}: {codes.max() + 1} clips")
            fig.savefig(OUT / f"entity{ent:03d}_t{t}.png", dpi=70)
            plt.close(fig)
    print(f"grids for entities {sample} at thresholds {args.grid_thresholds} -> {OUT}")


if __name__ == "__main__":
    main()
