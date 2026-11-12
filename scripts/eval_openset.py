"""Open-set scoring on saved checkpoints (docs/decisions.md D18).

    python scripts/eval_openset.py --stage val            # all candidates + selection
    python scripts/eval_openset.py --stage test --final   # chosen pipeline, ONCE, for reporting

Val run: evaluates the pre-declared candidates step by step (S1-S3, then +TTA,
+ENS, +SIFT; keep a step only if AUROC gains >= 0.01 and val rank-1/mAP don't
drop), writes results/openset_val_grid.csv and results/openset_choice.json, and
logs the chosen pipeline's val row to results.csv.

Embeddings are cached in checkpoints/emb/ (gitignored).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tiger_reid.data import REPO
from tiger_reid.dataset import eval_transform, extract_embeddings, extract_embeddings_tta, pick_device
from tiger_reid.metrics import auroc_from_scores, bootstrap_ci, cosine_distance, mean_ap, rank_k
from tiger_reid.openset import (centroid_dist, entity_min_dist, margin_score, topk_entities, zscore,
                                zscore_params)
from tiger_reid.results import log_result
from tiger_reid.sift_baseline import candidates_from_entities, max_inliers_candidates, sift_features
from tiger_reid.splits import eval_sets, load_splits
from tiger_reid.training import load_model

EMB_DIR = REPO / "checkpoints" / "emb"
GRID_CSV = REPO / "results" / "openset_val_grid.csv"
CHOICE_JSON = REPO / "results" / "openset_choice.json"
SETS = ["gallery", "query", "unknown", "unknown_all"]
MIN_GAIN = 0.01          # D18
TOL = 1e-9               # "don't drop" = no decrease at all


# ---------- embeddings / SIFT features (cached) ----------

def embeddings(ckpt: Path, stage: str, tta: bool, sets: dict, device) -> dict:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    st = ckpt.stat()   # key on the exact file, so a replaced checkpoint never reuses old embeddings
    cache = EMB_DIR / f"{ckpt.stem}_{st.st_size}_{int(st.st_mtime)}_{stage}_{'tta' if tta else 'plain'}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in SETS}
    model, cfg = load_model(ckpt, device)
    size = tuple(cfg["size"])
    out = {}
    for k in SETS:
        print(f"  embed {ckpt.stem} {stage} {k} ({len(sets[k])}){' tta' if tta else ''}", flush=True)
        out[k] = (extract_embeddings_tta(model, sets[k], size, device) if tta
                  else extract_embeddings(model, sets[k], eval_transform(size), device))
    np.savez(cache, **out)
    return out


def sift_feats(paths) -> list:
    cache = EMB_DIR / "sift_feats.pkl"
    store = pickle.load(open(cache, "rb")) if cache.exists() else {}
    missing = [p for p in map(str, paths) if p not in store]
    if missing:
        print(f"  SIFT features for {len(missing)} images", flush=True)
        for p in missing:
            store[p] = sift_features(p)
        pickle.dump(store, open(cache, "wb"))
    return [store[str(p)] for p in paths]


# ---------- one pipeline ----------

class Ctx:
    """Everything a pipeline needs for one stage: sets, embeddings, SIFT features."""

    def __init__(self, stage, ckpts: dict, device):
        self.stage = stage
        self.sets = eval_sets(load_splits(), stage)
        self.g_lab = self.sets["gallery"].entity_id.values
        self.q_lab = self.sets["query"].entity_id.values
        self.ckpts, self.device = ckpts, device
        self._emb, self._inl = {}, {}

    def emb(self, model: str, tta: bool) -> dict:
        key = (model, tta)
        if key not in self._emb:
            self._emb[key] = embeddings(self.ckpts[model], self.stage, tta, self.sets, self.device)
        return self._emb[key]

    def dists(self, spec) -> dict:
        """Image-level and centroid-level distances for query / unknown / unknown_all."""
        models = [("dino", 1.0)] if spec["ens_w"] is None else [("dino", spec["ens_w"]),
                                                                ("resnet", 1 - spec["ens_w"])]
        out = {}
        for k in SETS[1:]:
            img, cen = 0.0, 0.0
            for m, w in models:
                e = self.emb(m, spec["tta"])
                img = img + w * cosine_distance(e[k], e["gallery"])
                if spec["score"] == "centroid":
                    cen = cen + w * centroid_dist(e[k], e["gallery"], self.g_lab)[0]
            out[k] = (img, cen)
        return out

    def inliers(self, spec, k, top_ents) -> np.ndarray:
        key = (spec["score"], spec["tta"], spec["ens_w"], k)
        if key not in self._inl:
            print(f"  SIFT top-10 re-check: {k} ({len(top_ents)} queries)", flush=True)
            q = sift_feats(self.sets[k].path.values)
            g = sift_feats(self.sets["gallery"].path.values)
            cand = candidates_from_entities(top_ents, self.g_lab)
            self._inl[key] = max_inliers_candidates(q, g, cand)
        return self._inl[key]


def scores(ctx: Ctx, spec, zp=None):
    """-> metrics dict, raw embedding scores per set (for z-score fitting)."""
    d = ctx.dists(spec)
    emb_s, top = {}, {}
    for k, (img, cen) in d.items():
        ent, ents = entity_min_dist(img, ctx.g_lab)
        top[k] = topk_entities(ent, ents, 10)
        emb_s[k] = {"nearest": img.min(axis=1), "centroid": cen.min(axis=1) if np.ndim(cen) else None,
                    "margin": margin_score(ent)}[spec["score"]]
    s = dict(emb_s)
    if spec["sift_w"] is not None:
        zp = zp or {"emb": zscore_params(emb_s["query"], emb_s["unknown"])}
        inl = {k: np.log1p(ctx.inliers(spec, k, top[k])) for k in SETS[1:]}
        zp.setdefault("sift", zscore_params(inl["query"], inl["unknown"]))
        s = {k: zscore(emb_s[k], zp["emb"]) - spec["sift_w"] * zscore(inl[k], zp["sift"])
             for k in SETS[1:]}
    dq = d["query"][0]
    m = {**rank_k(dq, ctx.q_lab, ctx.g_lab), "mAP": mean_ap(dq, ctx.q_lab, ctx.g_lab),
         "auroc": auroc_from_scores(s["query"], s["unknown"]),
         "auroc_all": auroc_from_scores(s["query"], s["unknown_all"])}
    m |= bootstrap_ci(dq, ctx.q_lab, ctx.g_lab, s_known=s["query"], s_unknown=s["unknown"])
    m |= {"n_query": len(ctx.q_lab), "n_gallery": len(ctx.g_lab), "n_unknown": len(s["unknown"])}
    return m, zp


def describe(spec) -> str:
    parts = [spec["score"]]
    if spec["tta"]:
        parts.append("TTA")
    if spec["ens_w"] is not None:
        parts.append(f"ENS w={spec['ens_w']}")
    if spec["sift_w"] is not None:
        parts.append(f"SIFT w={spec['sift_w']}")
    return " + ".join(parts)


# ---------- selection on val (D18) ----------

def select(ctx: Ctx):
    rows = []

    def run(step, spec):
        m, zp = scores(ctx, spec)
        rows.append({"step": step, "pipeline": describe(spec), **{k: round(v, 4) if isinstance(v, float)
                                                                  else v for k, v in m.items()}})
        print(f"  {step:5s} {describe(spec):32s} AUROC {m['auroc']:.3f} "
              f"[{m['auroc_lo']:.2f}-{m['auroc_hi']:.2f}]  all {m['auroc_all']:.3f}  "
              f"r1 {m['rank1']:.3f}  mAP {m['mAP']:.3f}", flush=True)
        return m, zp

    base = {"score": "nearest", "tta": False, "ens_w": None, "sift_w": None}
    print("Step 1: S1-S3")
    results = {sc: run("S", {**base, "score": sc}) for sc in ["nearest", "centroid", "margin"]}
    best_sc = max(results, key=lambda sc: results[sc][0]["auroc"])
    spec, (best, zp) = {**base, "score": best_sc}, results[best_sc]

    def better(m):
        return (m["auroc"] >= best["auroc"] + MIN_GAIN and m["rank1"] >= best["rank1"] - TOL
                and m["mAP"] >= best["mAP"] - TOL)

    for step, key, grid in [("+TTA", "tta", [True]), ("+ENS", "ens_w", [0.25, 0.5, 0.75]),
                            ("+SIFT", "sift_w", [0.25, 0.5, 1.0])]:
        print(f"Step: {step}")
        tried = [({**spec, key: v}, run(step, {**spec, key: v})) for v in grid]
        cand_spec, (cand_m, cand_zp) = max(tried, key=lambda t: t[1][0]["auroc"])
        if better(cand_m):
            spec, best, zp = cand_spec, cand_m, cand_zp
            print(f"  -> keep {describe(spec)}")
        else:
            print(f"  -> drop (needs AUROC +{MIN_GAIN} without losing rank-1/mAP)")

    df = pd.DataFrame(rows)
    df["chosen"] = df.pipeline == describe(spec)
    GRID_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(GRID_CSV, index=False)
    return spec, best, zp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="evaluate the chosen pipeline (test: once)")
    ap.add_argument("--dino", default="checkpoints/dinov2s_arcface_s0.pt")
    ap.add_argument("--resnet", default="checkpoints/resnet50_arcface_s0.pt")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    ckpts = {"dino": REPO / args.dino, "resnet": REPO / args.resnet}
    for k, p in ckpts.items():
        if not p.exists():
            raise SystemExit(f"missing {p}: download it from Drive tiger-reid-runs/ into checkpoints/")
    if args.stage == "test" and not args.final:
        raise SystemExit("test is for the final pipeline only: use --final")
    ctx = Ctx(args.stage, ckpts, pick_device())

    if args.stage == "val" and not args.final:
        spec, m, zp = select(ctx)
        CHOICE_JSON.write_text(json.dumps({"spec": spec, "zscore_params": zp,
                                           "ckpts": {k: args.__dict__[k] for k in ["dino", "resnet"]}},
                                          indent=2))
        print(f"\nchosen: {describe(spec)}  val AUROC {m['auroc']:.3f} "
              f"(optimistic: chosen on openset_val)  -> {CHOICE_JSON.name}, {GRID_CSV.name}")
    else:
        choice = json.loads(CHOICE_JSON.read_text())
        spec = choice["spec"]
        m, _ = scores(ctx, spec, choice["zscore_params"])
        print(f"{args.stage}: {describe(spec)}", {k: round(v, 3) if isinstance(v, float) else v
                                                   for k, v in m.items()})
    if not args.no_log:
        log_result(f"open-set pipeline: {describe(spec)}", args.stage, m,
                   {"spec": spec, "dino": args.dino, "resnet": args.resnet, "rule": "D18"}, seed=0)


if __name__ == "__main__":
    main()
