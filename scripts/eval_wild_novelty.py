"""Wild novelty check: every Rajaji camera-trap tiger is new to the zoo gallery.

    python scripts/eval_wild_novelty.py --dino checkpoints/dinov2b_s0.pt [--d25]

The wild images (data/wild_rajaji/crops, gitignored) are wild Bengal tigers;
the ATRW gallery is zoo Amur tigers, so no wild image can be a known tiger.
Scores each one against the *val* gallery (test stays untouched) with the same
code as eval_openset.py (z-score params fitted on val known + unknown only) and reports:
  - AUROC val known queries vs each "new" set (higher = new tigers look more "new"),
  - false-"known" rate at thresholds keeping 95% / 90% of val known queries recognized,
for: the saved choice, plain nearest distance, and (--d25) every D25 candidate.
Wild images are never used to choose anything. Writes results/wild_novelty.csv
(aggregates only; no coordinates, no per-image rows).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_openset as eo  # noqa: E402

from tiger_reid.data import REPO  # noqa: E402
from tiger_reid.dataset import pick_device  # noqa: E402
from tiger_reid.metrics import auroc_from_scores  # noqa: E402
from tiger_reid.openset import false_known_rate, known_threshold  # noqa: E402

WILD = REPO / "data" / "wild_rajaji"
OUT = REPO / "results" / "wild_novelty.csv"
NEW_SETS = ["unknown", "unknown_all", "wild_all", "wild_good+ok", "wild_good"]


def wild_df() -> pd.DataFrame:
    notes = pd.read_csv(WILD / "claude_notes.csv", dtype={"gbif_key": str})
    notes["path"] = [str(WILD / "crops" / f"{k}.jpg") for k in notes.gbif_key]
    notes = notes[[Path(p).exists() for p in notes.path]].reset_index(drop=True)
    # ReidDataset requires an entity_id column; embedding extraction never reads it.
    # Per-image key so nothing groups wild images (we don't know which are the same tiger).
    notes["entity_id"] = "wild_" + notes.gbif_key
    return notes


def bootstrap_auroc(sk, su, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = [auroc_from_scores(rng.choice(sk, len(sk)), rng.choice(su, len(su))) for _ in range(n)]
    return np.percentile(a, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default="checkpoints/dinov2b_s0.pt")
    ap.add_argument("--d25", action="store_true", help="also every D25 candidate")
    args = ap.parse_args()
    choice = json.loads(eo.CHOICE_JSON.read_text())

    ctx = eo.Ctx("val", {"dino": REPO / args.dino}, pick_device())
    wild = wild_df()
    ctx.add_set("wild_all", wild)
    masks = {"wild_good+ok": wild.quality.isin(["g", "o"]).values, "wild_good": (wild.quality == "g").values}

    specs = [(f"chosen ({choice.get('rule', 'D18')})", choice["spec"])]
    if "previous" in choice:
        specs.append(("previous", choice["previous"]["spec"]))
    specs.append(("nearest plain", {"score": "nearest", "tta": True, "ens_w": None, "sift_w": None}))
    if args.d25:
        specs += [(n, sp) for n, sp in eo.d25_candidates() if sp != choice["spec"]]

    rows = []
    for name, spec in specs:
        _, _, s = eo.scores(ctx, spec)            # z-score params fitted on val known + unknown only
        for k, m in masks.items():
            s[k] = s["wild_all"][m]
        thr = {q: known_threshold(s["query"], q) for q in (0.95, 0.90)}
        for k in NEW_SETS:
            lo, hi = bootstrap_auroc(s["query"], s[k])
            rows.append({"name": name, "pipeline": eo.describe(spec), "new_set": k, "n_new": len(s[k]),
                         "auroc": round(auroc_from_scores(s["query"], s[k]), 4),
                         "auroc_lo": round(lo, 3), "auroc_hi": round(hi, 3),
                         "false_known@keep95": round(false_known_rate(s[k], thr[0.95]), 4),
                         "false_known@keep90": round(false_known_rate(s[k], thr[0.90]), 4),
                         "n_known": len(s["query"]), "dino": args.dino})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["dino", "n_known", "name"]).to_string(index=False))
    print(f"-> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
