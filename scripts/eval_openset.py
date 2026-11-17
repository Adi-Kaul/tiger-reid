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
from tiger_reid.openset import (centroid_dist, entity_min_dist, false_known_rate, fuse_scores,
                                known_accept_rate, known_threshold, margin_score, threshold_for_false_yes,
                                top_images_per_entity, topk_entities, zscore_params)
from tiger_reid.results import log_result
from tiger_reid.sift_baseline import candidates_from_entities, max_inliers_candidates, sift_features
from tiger_reid.splits import eval_sets, load_splits
from tiger_reid.training import load_model

EMB_DIR = REPO / "checkpoints" / "emb"
GRID_CSV = REPO / "results" / "openset_val_grid.csv"
GRID_D24_CSV = REPO / "results" / "openset_val_grid_d24.csv"
GRID_D25_CSV = REPO / "results" / "openset_val_grid_d25.csv"
GRID_D27_CSV = REPO / "results" / "openset_val_grid_d27.csv"
LG_PER_ENTITY = 5        # D25: nearest gallery images per top-10 entity for LightGlue
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


def extra_embeddings(ckpt: Path, name: str, df, tta: bool, device) -> np.ndarray:
    st = ckpt.stat()
    cache = EMB_DIR / f"{name}_{ckpt.stem}_{st.st_size}_{int(st.st_mtime)}_{len(df)}_{'tta' if tta else 'plain'}.npy"
    if cache.exists():
        return np.load(cache)
    model, cfg = load_model(ckpt, device)
    print(f"  embed {ckpt.stem} {name} ({len(df)}){' tta' if tta else ''}", flush=True)
    e = (extract_embeddings_tta(model, df, tuple(cfg["size"]), device) if tta
         else extract_embeddings(model, df, eval_transform(tuple(cfg["size"])), device))
    np.save(cache, e)
    return e


def sift_feats(paths, masker=None) -> list:
    """SIFT features per image; with a masker (D27), keypoints only on the tiger."""
    cache = EMB_DIR / ("sift_feats_masked.pkl" if masker else "sift_feats.pkl")
    store = pickle.load(open(cache, "rb")) if cache.exists() else {}
    missing = [p for p in map(str, paths) if p not in store]
    if missing:
        print(f"  SIFT features for {len(missing)} images{' (tiger-only)' if masker else ''}", flush=True)
        masks = masker.masks(missing) if masker else [None] * len(missing)
        for p, mk in zip(missing, masks):
            store[p] = sift_features(p, mask=mk)
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
        self._emb, self._inl, self._lg = {}, {}, {}
        self.score_sets = list(SETS[1:])     # sets that get novelty scores
        self._matcher = None
        self._masker = None

    def masker(self):
        """D27 foreground masker; PCA fit on train-split images only."""
        if self._masker is None:
            from tiger_reid.fg_mask import FgMasker
            sp = load_splits()
            self._masker = FgMasker(sp.loc[sp.split == "train", "path"].values, self.device)
        return self._masker

    def add_set(self, name: str, df):
        """Extra query set scored against the same gallery (e.g. wild images). No labels used."""
        self.sets[name] = df.reset_index(drop=True)
        self.score_sets.append(name)
        self._emb = {}

    def emb(self, model: str, tta: bool) -> dict:
        key = (model, tta)
        if key not in self._emb:
            e = embeddings(self.ckpts[model], self.stage, tta, self.sets, self.device)
            for name in self.score_sets[len(SETS) - 1:]:
                e[name] = extra_embeddings(self.ckpts[model], name, self.sets[name], tta, self.device)
            self._emb[key] = e
        return self._emb[key]

    def dists(self, spec) -> dict:
        """Image-level and centroid-level distances for query / unknown / unknown_all."""
        models = [("dino", 1.0)] if spec["ens_w"] is None else [("dino", spec["ens_w"]),
                                                                ("resnet", 1 - spec["ens_w"])]
        out = {}
        for k in self.score_sets:
            img, cen = 0.0, 0.0
            for m, w in models:
                e = self.emb(m, spec["tta"])
                img = img + w * cosine_distance(e[k], e["gallery"])
                if spec["score"] == "centroid":
                    cen = cen + w * centroid_dist(e[k], e["gallery"], self.g_lab)[0]
            out[k] = (img, cen)
        return out

    def inliers(self, spec, k, top_ents) -> np.ndarray:
        mk = bool(spec.get("sift_mask"))
        key = (spec["tta"], spec["ens_w"], k, mk)
        if key not in self._inl:
            print(f"  SIFT{' (tiger-only)' if mk else ''} top-10 re-check: {k} ({len(top_ents)} queries)", flush=True)
            masker = self.masker() if mk else None
            q = sift_feats(self.sets[k].path.values, masker)
            g = sift_feats(self.sets["gallery"].path.values, masker)
            cand = candidates_from_entities(top_ents, self.g_lab)
            self._inl[key] = max_inliers_candidates(q, g, cand)
        return self._inl[key]

    def lg_inliers(self, spec, k, cand) -> np.ndarray:
        key = (spec["tta"], spec["ens_w"], k)
        if key not in self._lg:
            if self._matcher is None:
                from tiger_reid.local_match import DiskLightGlue
                self._matcher = DiskLightGlue(self.device)
            print(f"  LightGlue re-check: {k} ({len(cand)} queries x {len(cand[0])} images)", flush=True)
            self._lg[key] = self._matcher.max_inliers_candidates(
                self.sets[k].path.values, self.sets["gallery"].path.values, cand)
        return self._lg[key]


def scores(ctx: Ctx, spec, zp=None):
    """-> metrics dict, z-score params, final novelty score per set.

    zp=None fits z-score params on this stage's known + unknown pool (val only);
    on test, pass the params saved from val."""
    d = ctx.dists(spec)
    emb_s, near, top, img_d = {}, {}, {}, {}
    for k, (img, cen) in d.items():
        ent, ents = entity_min_dist(img, ctx.g_lab)
        top[k], img_d[k] = topk_entities(ent, ents, 10), img
        near[k] = img.min(axis=1)
        emb_s[k] = {"nearest": near[k], "centroid": cen.min(axis=1) if np.ndim(cen) else None,
                    "margin": margin_score(ent)}[spec["score"]]
    a, w, lw = spec.get("near_a"), spec["sift_w"], spec.get("lg_w")
    zp = dict(zp) if zp else {}
    need = (["emb"] if a is not None or w is not None or lw is not None else []) \
        + (["near"] if a is not None else []) + (["sift"] if w is not None else []) + (["lg"] if lw is not None else [])
    if ctx.stage == "test" and any(n not in zp for n in need):
        raise SystemExit(f"test needs val z-score params {need}, got {list(zp)}")
    inl = {k: np.log1p(ctx.inliers(spec, k, top[k])) for k in d} if w is not None else {}
    lg = {k: np.log1p(ctx.lg_inliers(spec, k, top_images_per_entity(img_d[k], ctx.g_lab, top[k], LG_PER_ENTITY)))
          for k in d} if lw is not None else {}
    if "emb" in need:
        zp.setdefault("emb", zscore_params(emb_s["query"], emb_s["unknown"]))
    if "near" in need:
        zp.setdefault("near", zscore_params(near["query"], near["unknown"]))
    if "sift" in need:
        zp.setdefault("sift", zscore_params(inl["query"], inl["unknown"]))
    if "lg" in need:
        zp.setdefault("lg", zscore_params(lg["query"], lg["unknown"]))
    s = {k: fuse_scores(emb_s[k], spec, zp, near[k], inl.get(k), lg.get(k)) for k in d}
    dq = d["query"][0]
    m = {**rank_k(dq, ctx.q_lab, ctx.g_lab), "mAP": mean_ap(dq, ctx.q_lab, ctx.g_lab),
         "auroc": auroc_from_scores(s["query"], s["unknown"]),
         "auroc_all": auroc_from_scores(s["query"], s["unknown_all"])}
    m |= bootstrap_ci(dq, ctx.q_lab, ctx.g_lab, s_known=s["query"], s_unknown=s["unknown"])
    m |= {"n_query": len(ctx.q_lab), "n_gallery": len(ctx.g_lab), "n_unknown": len(s["unknown"])}
    return m, zp, s


def describe(spec) -> str:
    parts = [spec["score"]]
    if spec["tta"]:
        parts.append("TTA")
    if spec.get("near_a") is not None:
        parts.append(f"{spec['near_a']}*nearest")
    if spec.get("lg_w") is not None:
        parts.append(f"LightGlue w={spec['lg_w']}")
    if spec["ens_w"] is not None:
        parts.append(f"ENS w={spec['ens_w']}")
    if spec["sift_w"] is not None:
        parts.append(f"SIFT{' tiger-only' if spec.get('sift_mask') else ''} w={spec['sift_w']}")
    return " + ".join(parts)


# ---------- selection on val (D18) ----------

def select(ctx: Ctx):
    rows = []

    def run(step, spec):
        m, zp, _ = scores(ctx, spec)
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


def false_known(s) -> dict:
    """False-"known" rate on unknowns at thresholds keeping 95% / 90% of known queries,
    and (D26) the share of knowns auto-recognized at 5% / 10% false yes on val unknowns."""
    out = {f"false_known@keep{int(q * 100)}": round(false_known_rate(s["unknown"], known_threshold(s["query"], q)), 4)
           for q in (0.95, 0.90)}
    for r in (0.05, 0.10):
        thr = threshold_for_false_yes(s["unknown"], r)
        out[f"known_auto@fy{int(r * 100)}"] = round(known_accept_rate(s["query"], thr), 4)
        out[f"fy_all@fy{int(r * 100)}"] = round(false_known_rate(s["unknown_all"], thr), 4)
    return out


def select_extra(ctx: Ctx, choice: dict, cands: list, grid_csv: Path, tag: str, controls=()):
    """D24/D25/D27: pre-declared extra candidates; replace the current choice only if AUROC +0.01.
    `controls` are reported but can't be chosen (D27: unmasked SIFT at the new weight)."""
    cur = choice["spec"]
    rows = []

    def run(name, spec):
        m, zp, s = scores(ctx, spec)
        fk = false_known(s)
        rows.append({"candidate": name, "pipeline": describe(spec), "spec": json.dumps(spec),
                     "auroc_unknown_all": round(m["auroc_all"], 4),
                     "auroc": round(m["auroc"], 4), "auroc_lo": round(m["auroc_lo"], 3),
                     "auroc_hi": round(m["auroc_hi"], 3), "auroc_all": round(m["auroc_all"], 4), **fk,
                     "rank1": round(m["rank1"], 4), "mAP": round(m["mAP"], 4)})
        print(f"  {name:6s} {describe(spec):40s} AUROC {m['auroc']:.3f} [{m['auroc_lo']:.2f}-{m['auroc_hi']:.2f}]"
              f"  all {m['auroc_all']:.3f}  false-known 95/90: {fk['false_known@keep95']:.2f}"
              f" / {fk['false_known@keep90']:.2f}  auto@fy5/10: {fk['known_auto@fy5']:.2f} / {fk['known_auto@fy10']:.2f}",
              flush=True)
        return m, zp

    m_cur, _ = run("current", cur)
    for name, spec in controls:
        run(name, spec)
    tried = [(spec, run(name, spec)) for name, spec in cands]
    spec, (m, zp) = max(tried, key=lambda t: t[1][0]["auroc"])
    adopt = m["auroc"] >= m_cur["auroc"] + MIN_GAIN - TOL
    df = pd.DataFrame(rows)
    df["chosen"] = (df.pipeline == describe(spec if adopt else cur)) & ~df.candidate.isin([n for n, _ in controls])
    df.to_csv(grid_csv, index=False)
    print(f"\nbest {tag} candidate: {describe(spec)} AUROC {m['auroc']:.3f} vs current {m_cur['auroc']:.3f} -> "
          + ("ADOPT" if adopt else "keep current (needs +0.01)"))
    return (spec, m, zp) if adopt else None


def d24_candidates():
    base = {"tta": True, "ens_w": None}
    c = [("N+S", {**base, "score": "nearest", "near_a": None, "sift_w": w}) for w in (0.25, 0.5, 1.0)]
    return c + [("M+N+S", {**base, "score": "margin", "near_a": a, "sift_w": w}) for a in (0.5, 1.0, 2.0)
                for w in (0.5, 1.0)]


def d25_candidates():
    base = {"score": "margin", "tta": True, "ens_w": None, "near_a": None}
    c = [("LG", {**base, "sift_w": None, "lg_w": w}) for w in (0.5, 1.0, 2.0)]
    return c + [("S+LG", {**base, "sift_w": 1.0, "lg_w": w}) for w in (0.5, 1.0)]


def d27_candidates():
    base = {"score": "margin", "tta": True, "ens_w": None, "near_a": None}
    return [("MS", {**base, "sift_w": w, "sift_mask": True}) for w in (0.5, 1.0, 2.0)]


def d27_controls():
    return [("ctrl", {"score": "margin", "tta": True, "ens_w": None, "near_a": None, "sift_w": 2.0})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="evaluate the chosen pipeline (test: once)")
    ap.add_argument("--dino", default="checkpoints/dinov2s_arcface_s0.pt")
    ap.add_argument("--resnet", default="checkpoints/resnet50_arcface_s0.pt")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--d24", action="store_true", help="D24 extra candidates on val vs the saved choice")
    ap.add_argument("--d25", action="store_true", help="D25 LightGlue candidates on val vs the saved choice")
    ap.add_argument("--d27", action="store_true", help="D27 tiger-only SIFT candidates on val vs the saved choice")
    args = ap.parse_args()

    ckpts = {"dino": REPO / args.dino, "resnet": REPO / args.resnet}
    for k, p in ckpts.items():
        if not p.exists():
            raise SystemExit(f"missing {p}: download it from Drive tiger-reid-runs/ into checkpoints/")
    if args.stage == "test" and not args.final:
        raise SystemExit("test is for the final pipeline only: use --final")
    ctx = Ctx(args.stage, ckpts, pick_device())

    tag = "D24" if args.d24 else "D25" if args.d25 else "D27" if args.d27 else None
    if tag:
        if args.stage != "val":
            raise SystemExit("--d24/--d25/--d27 are val-only selection steps")
        choice = json.loads(CHOICE_JSON.read_text())
        cands, grid, ctrl = {"D24": (d24_candidates(), GRID_D24_CSV, ()),
                             "D25": (d25_candidates(), GRID_D25_CSV, ()),
                             "D27": (d27_candidates(), GRID_D27_CSV, d27_controls())}[tag]
        res = select_extra(ctx, choice, cands, grid, tag, ctrl)
        if res is None:
            return
        spec, m, zp = res
        CHOICE_JSON.write_text(json.dumps({"spec": spec, "zscore_params": zp, "rule": tag,
                                           "ckpts": {k: args.__dict__[k] for k in ["dino", "resnet"]},
                                           "previous": choice}, indent=2))
    elif args.stage == "val" and not args.final:
        spec, m, zp = select(ctx)
        CHOICE_JSON.write_text(json.dumps({"spec": spec, "zscore_params": zp,
                                           "ckpts": {k: args.__dict__[k] for k in ["dino", "resnet"]}},
                                          indent=2))
        print(f"\nchosen: {describe(spec)}  val AUROC {m['auroc']:.3f} "
              f"(optimistic: chosen on openset_val)  -> {CHOICE_JSON.name}, {GRID_CSV.name}")
    else:
        choice = json.loads(CHOICE_JSON.read_text())
        spec = choice["spec"]
        m, _, _ = scores(ctx, spec, choice["zscore_params"])
        print(f"{args.stage}: {describe(spec)}", {k: round(v, 3) if isinstance(v, float) else v
                                                   for k, v in m.items()})
    if not args.no_log:
        log_result(f"open-set pipeline: {describe(spec)}", args.stage, m,
                   {"spec": spec, "dino": args.dino, "resnet": args.resnet, "rule": tag or "D18"}, seed=0)


if __name__ == "__main__":
    main()
