"""Wild novelty check: every Rajaji camera-trap tiger is new to the zoo gallery.

    python scripts/eval_wild_novelty.py --dino checkpoints/dinov2b_s0.pt

The wild images (data/wild_rajaji/crops, gitignored) are wild Bengal tigers;
the ATRW gallery is zoo Amur tigers, so no wild image can be a known tiger.
Scores each one with the chosen open-set pipeline (results/openset_choice.json,
z-score params from val) against the *val* gallery (test stays untouched) and reports:
  - AUROC val known queries vs wild images (higher = wild looks more "new"),
  - false-"known" rate on wild at thresholds that keep 95% / 90% of val known
    queries recognized (thresholds fixed on val knowns only),
  - the same for val unknowns (openset_val) and for plain nearest distance, for comparison.
Writes results/wild_novelty.csv (aggregates only; no coordinates, no per-image rows).
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
from tiger_reid.dataset import extract_embeddings_tta, pick_device  # noqa: E402
from tiger_reid.metrics import auroc_from_scores, cosine_distance  # noqa: E402
from tiger_reid.openset import (entity_min_dist, false_known_rate, known_threshold,  # noqa: E402
                                margin_score, topk_entities, zscore)
from tiger_reid.sift_baseline import candidates_from_entities, max_inliers_candidates  # noqa: E402
from tiger_reid.training import load_model  # noqa: E402

WILD = REPO / "data" / "wild_rajaji"
OUT = REPO / "results" / "wild_novelty.csv"


def wild_sets() -> dict:
    notes = pd.read_csv(WILD / "claude_notes.csv", dtype={"gbif_key": str})
    notes["path"] = [str(WILD / "crops" / f"{k}.jpg") for k in notes.gbif_key]
    notes = notes[[Path(p).exists() for p in notes.path]].reset_index(drop=True)
    # ReidDataset requires an entity_id column; embedding extraction never reads it.
    # Per-image key so nothing groups wild images (we don't know which are the same tiger).
    notes["entity_id"] = "wild_" + notes.gbif_key
    notes["entity_id"] = "wild"   # dummy: ReidDataset needs it; never used as a label
    return {"wild_all": notes,
            "wild_good+ok": notes[notes.quality.isin(["g", "o"])].reset_index(drop=True),
            "wild_good": notes[notes.quality == "g"].reset_index(drop=True)}


def wild_embeddings(ckpt: Path, df: pd.DataFrame, device) -> np.ndarray:
    st = ckpt.stat()
    cache = eo.EMB_DIR / f"wild_{ckpt.stem}_{st.st_size}_{int(st.st_mtime)}_{len(df)}_tta.npy"
    if cache.exists():
        return np.load(cache)
    model, cfg = load_model(ckpt, device)
    print(f"  embed wild ({len(df)}) tta", flush=True)
    e = extract_embeddings_tta(model, df, tuple(cfg["size"]), device)
    np.save(cache, e)
    return e


def bootstrap_auroc(sk, su, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = [auroc_from_scores(rng.choice(sk, len(sk)), rng.choice(su, len(su))) for _ in range(n)]
    return np.percentile(a, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default="checkpoints/dinov2b_s0.pt")
    args = ap.parse_args()
    choice = json.loads(eo.CHOICE_JSON.read_text())
    spec, zp = choice["spec"], choice["zscore_params"]
    assert spec["score"] == "margin" and spec["tta"] and spec["ens_w"] is None, spec
    ckpt = REPO / args.dino
    device = pick_device()

    ctx = eo.Ctx("val", {"dino": ckpt}, device)
    e = ctx.emb("dino", True)
    g_lab, g_paths = ctx.g_lab, ctx.sets["gallery"].path.values
    g_sift = eo.sift_feats(g_paths)

    wild = wild_sets()
    all_df = wild["wild_all"]
    w_emb = wild_embeddings(ckpt, all_df, device)
    emb = {"val_known": e["query"], "val_unknown": e["unknown"], "wild_all": w_emb}
    paths = {"val_known": ctx.sets["query"].path.values, "val_unknown": ctx.sets["unknown"].path.values,
             "wild_all": all_df.path.values}

    fused, nearest = {}, {}
    for k in emb:
        d = cosine_distance(emb[k], e["gallery"])
        ent, ents = entity_min_dist(d, g_lab)
        print(f"  SIFT top-10 re-check: {k} ({len(d)})", flush=True)
        inl = max_inliers_candidates(eo.sift_feats(paths[k]), g_sift,
                                     candidates_from_entities(topk_entities(ent, ents, 10), g_lab))
        fused[k] = zscore(margin_score(ent), zp["emb"]) - spec["sift_w"] * zscore(np.log1p(inl), zp["sift"])
        nearest[k] = d.min(axis=1)
    for name in ["wild_good+ok", "wild_good"]:
        mask = all_df.gbif_key.isin(wild[name].gbif_key).values
        fused[name], nearest[name] = fused["wild_all"][mask], nearest["wild_all"][mask]

    rows = []
    for score_name, s in [(eo.describe(spec), fused), ("nearest (plain, TTA)", nearest)]:
        thr = {keep: known_threshold(s["val_known"], keep) for keep in (0.95, 0.90)}
        for k in ["val_unknown", "wild_all", "wild_good+ok", "wild_good"]:
            lo, hi = bootstrap_auroc(s["val_known"], s[k])
            rows.append({"score": score_name, "new_set": k, "n_new": len(s[k]), "n_known": len(s["val_known"]),
                         "auroc": round(auroc_from_scores(s["val_known"], s[k]), 4),
                         "auroc_lo": round(lo, 3), "auroc_hi": round(hi, 3),
                         "false_known@keep95": round(false_known_rate(s[k], thr[0.95]), 4),
                         "false_known@keep90": round(false_known_rate(s[k], thr[0.90]), 4),
                         "dino": args.dino})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.drop(columns=["dino", "n_known"]).to_string(index=False))
    print(f"-> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
