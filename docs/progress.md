# Progress log

One entry per working day, newest first. Each entry: what got done, findings, problems and how they were solved, what's left for me, and what's next.

---

## ▶ START HERE (handoff for the next session, updated Nov 16)

**Where we are:** Phase 4, final steps. Base model is **`dinov2b`** (DINOv2 ViT-B/14 + ArcFace, 224 px; D22): 4-run val mAP 0.78 ± 0.04, rank-1 81%. Open-set pipeline chosen on val (D18): **margin score + 5-crop TTA + SIFT re-check of the top-10 entities (w=1)** → val AUROC **0.883** [0.82–0.94], rank-1 82%, mAP 0.82 (seed 0). **Test set still untouched.**

**⏸ Open decision (user asked to hold it): D26, how to pick the open-set threshold.** At a threshold that keeps 95% of known tigers recognized, ~half of new tigers are wrongly accepted as known (val 48%, wild 49%). Cause: the hardest ~5–10% of known queries show a view whose side isn't in the gallery (rear, frontal, crouched, behind a fence), so no matcher can line up the stripes (D25 result). Proposal: set the threshold for a **false-yes target** on val and send everything above it to human review. D18 pipeline on val: 5% false yes → 69% of knowns auto-recognized; 10% → 79%. Options: 5%, 10%, or report both. See `docs/learning/05-open-set.md`.

**What was tried for the false yes's (all on val, all pre-declared, test untouched):**
- D24 add absolute nearest distance → worse (best 0.864). It only helps on wild images (domain shift).
- D25 DISK + LightGlue re-check (`src/tiger_reid/local_match.py`) → 0.890, under the +0.01 bar. Much better on typical images, no help on unmatched views.
- Wild check (`scripts/eval_wild_novelty.py`): all 269 Rajaji images are new tigers by construction; D18 AUROC 0.885 known-vs-wild.

**Next, in order:**
1. User decides D26 (threshold rule). Then log it in `docs/decisions.md` and implement the threshold in `eval_openset.py --final` (false-yes rate + share of knowns auto-recognized).
2. Download `dinov2b` seed 1 (17:58 run) and seed 2 checkpoints from the adikaul07 Drive `tiger-reid-runs/` into `checkpoints/`. (`dinov2b_s0.pt` is in place.)
3. Before the test run, fix in `eval_openset.py`: (a) `log_result` hard-codes seed=0; (b) z-score params (and the D26 threshold) must be refit on val **per seed**, not reused from seed 0.
4. Test once (D21 C) on seeds 0–2 → mean ± sd → `scripts/summarize_results.py`. That closes Phase 4; check with the user before Phase 5.

**Known data issues:** val unknowns = 46 images of only **7 tigers**; openset_val entity 250 looks like a mislabeled image of known 247 (kept in, D25 note). Wild Rajaji labeling (D23) is **parked**: proposed groups weren't well sorted.

**Mode:** rapid development (`CLAUDE.md`): Claude writes code, explains briefly, runs `pytest -q` (55 tests), commits + pushes each verified step, asks at phase boundaries and on result-shaping choices.
**Read first:** [`learning/00-project-story.md`](learning/00-project-story.md), [`learning/05-open-set.md`](learning/05-open-set.md), [`decisions.md`](decisions.md) D18–D26.

**Useful commands:**
```bash
source .venv/bin/activate && pytest -q
python scripts/eval_openset.py --stage val --dino checkpoints/dinov2b_s0.pt          # D18 selection (~10 min)
python scripts/eval_openset.py --stage val --d25 --dino checkpoints/dinov2b_s0.pt    # LightGlue candidates (~30 min)
python scripts/eval_wild_novelty.py --dino checkpoints/dinov2b_s0.pt                 # wild all-new check
python scripts/eval_openset.py --stage test --final --dino ...                       # ONCE, after D26
python scripts/summarize_results.py                                                  # -> results/summary.md
```

**Accounts / compute:** use adikaul07@gmail.com for everything (Drive, Colab, Gmail, git). Training on Colab T4; local `mps` for eval. Weights: adikaul07 Drive `MyDrive/tiger-reid-runs/` (older runs in the umich Drive).
**Wild data contacts (Nov 12–14):** Harihar (Panthera) pointed to GBIF only; full set needs WII permission. GBIF Rajaji: 269/305 images cached, no individual IDs. Records carry fine coordinates: never commit them.

---

## Nov 16, 2026 (Mon) — chasing the false yes's

**Done**
- Error analysis of the D18 open-set failures on val (contact sheets in the scratchpad, not committed).
- D25 (pre-declared): DISK + LightGlue re-check (`src/tiger_reid/local_match.py`, kornia 0.7.3), 5 candidates. Not adopted (best 0.890 < 0.893).
- `eval_openset.py`: `Ctx.add_set` for extra image sets, generic `select_extra` (D24/D25), LightGlue term in `fuse_scores`. `eval_wild_novelty.py` now reuses it.

**Findings**
- Known queries that look new are **view changes** (rear, crouched, frontal, behind a fence) whose side isn't in the gallery. Unknowns that look known are **other tigers from the same zoo** (same ground, fences, light).
- The 46 val unknowns are only **7 tigers**; entity 250 looks like a mislabeled image of known 247 (same scene and moment, 159 SIFT inliers). Kept in (D25 data note).
- LightGlue: known median 76 inliers vs unknown median 22 (SIFT 16 vs 6), but the hardest 10% of knowns get ≤16 with either matcher. So the keep-95% false-yes rate is limited by gallery coverage, not the matcher.
- Turning it around (D18 pipeline, val): at **5% false "known" on val unknowns, 69% of known tigers are recognized automatically** (unknown_all false-known 9%); at 10%: 79% (unknown_all 14%). The rest would go to human review. Plain nearest distance: 51% / 58%. Caveat: 5% of 46 unknowns is ~2 images.

- Docs refresh: START HERE rewritten, roadmap, D26 logged as *on hold*, learning notes 05 (open-set operating points) and 06 (local matching), README pipeline step 3.

**Next**
- User decision (on hold): D26, threshold by a false-yes target with a "review" bucket, instead of keep-95% of knowns.

---

## Nov 15, 2026 (Sun) — Stage A open-set selection

**Done**
- Parked wild Rajaji labeling (groups not well sorted).
- Stage A (D18) on val with `dinov2b_s0.pt`:

| step | AUROC [95% CI] | AUROC (all) | rank-1 | mAP |
|---|---|---|---|---|
| S1 nearest | 0.772 [0.69–0.85] | 0.837 | 82% | 0.819 |
| S2 centroid | 0.775 | 0.815 | 82% | 0.819 |
| S3 margin | 0.795 | 0.843 | 82% | 0.819 |
| margin + TTA | 0.807 | 0.856 | 82% | 0.820 |
| + ENS w=0.75 (dropped) | 0.807 | 0.815 | 81% | 0.831 |
| **+ SIFT w=1.0 (chosen)** | **0.883 [0.82–0.94]** | **0.901** | 82% | 0.820 |

**Findings**
- SIFT re-checking the top-10 candidates is the big open-set win (+0.08 AUROC over margin+TTA); embeddings rank, SIFT confirms. It doesn't change rank-1/mAP because it only feeds the novelty score.
- ENS with ResNet50 hurts rank-1 at w≤0.5 and only ties AUROC at 0.75 → dropped by the rule.
- AUROC rose monotonically with SIFT weight and 1.0 is the grid edge, so a larger weight might do better on val; not tried (grid was fixed in advance, and tuning further on 46 unknowns would overfit).

**Wild novelty check** (`scripts/eval_wild_novelty.py` → `results/wild_novelty.csv`): all 269 Rajaji crops are wild tigers, so none can be in the zoo gallery. Each is scored against the val gallery (test untouched); thresholds keep 95% / 90% of val known queries recognized.

| score | new set | AUROC [CI] | false 'known' @95% / @90% |
|---|---|---|---|
| margin+TTA+SIFT (chosen) | val unknowns (46) | 0.883 [0.81–0.94] | 48% / 46% |
| margin+TTA+SIFT (chosen) | wild all (269) | 0.885 [0.83–0.93] | 49% / 42% |
| margin+TTA+SIFT (chosen) | wild good only (81) | 0.875 | 59% / 49% |
| nearest (TTA) | val unknowns | 0.781 | 80% / 52% |
| nearest (TTA) | wild all | 0.948 [0.91–0.98] | 33% / 9% |

- The chosen score ranks wild tigers as new about as well as val unknowns (AUROC 0.88), but at a threshold that keeps 95% of known tigers, **about half of new tigers (wild or zoo) still get accepted as known**. AUROC hides this; a few known queries with high novelty scores push the threshold up. Those thresholds come from 67 known queries, so they're noisy.
- Plain nearest distance flags wild images much better (9% false known at 90%): wild night/IR shots sit far from every zoo image. The margin score only looks at the gap between the top two tigers, so it ignores that absolute distance. Part of nearest's win is 'this image looks unusual' (domain shift), not 'this is a different tiger'.
- Good-quality wild images are harder (59% false known): the clearer the tiger, the more it looks like a gallery tiger.
- Repeat tigers in the wild set (same animal, several shots) mean fewer independent tigers than 269, so the wild CIs are too narrow.

**D24 (pre-declared, then run):** 9 extra candidates adding nearest distance (N+S, M+N+S) on val. None beat D18: best is margin + 0.5·nearest + SIFT, AUROC 0.864 vs 0.883, and every one has an equal or worse false-"known" rate on val unknowns. **D18 pipeline stays.** Adding absolute distance hurts on zoo unknowns; it only helped on wild because wild images look unlike any zoo image (domain shift). The false-yes problem (~half of new tigers accepted at keep-95%) stays a reported limitation.

**Next**
- Download dinov2b seed 1 (17:58) and seed 2 checkpoints → `--stage test --final` once per seed → mean ± sd.

---

## Nov 14, 2026 (Sat) — wild-data reply

**Done**
- Read Abishek Harihar's reply (Panthera): use the GBIF dataset; missing images need WII permission.
- Probed GBIF dataset `e61455a4-…`: 305 records, all with one image; **0 individual IDs**; 269 images still served from GBIF's image cache at full 1024×768, 36 not (504, origin `ibif.gov.in` down).

**Findings**
- Usable as an *unlabeled* zoo→wild set, not a re-ID benchmark as-is. Weak labels: sex (cross-sex pairs are certainly different tigers), camera station + time (same-camera bursts are almost surely the same tiger).
- The IDs exist in Harihar's own population work, so a catalogNumber → tiger-ID table (no images needed) is the cheapest thing to ask for.

**Next**
- **Later (D23):** fetched 269 images (`scripts/fetch_wild_rajaji.py`), cropped them with a COCO detector (224/269 boxes >0.5), and Claude noted flank and quality for each: 87 left / 64 right usable, 12 frontal, 106 unusable (dark, blur, partial, multi-tiger). Proposals: 133 groups, only 15 with 2+ images (event groups + 7 visual merges incl. 4 images with biologist circle marks). Most cross-event matching is left to the user: Claude's stripe matching at these resolutions was too uncertain to merge more.
- **Next:** ~~user reviews `data/wild_rajaji/review.html`~~ parked Nov 15 (groups not well sorted). Stage A still waits on `dinov2b_s0.pt`.

---

## Nov 13, 2026 (Fri) — B2 (336 px) and B3 (ViT-B)

**Done**
- **B2 `dinov2s_336` (D22): plain DINOv2-S at 336×336, fp16, seed 0, T4 (adikaul07 Colab).** Best epoch 13, early-stopped at 21, 60 s/epoch (vs 36 s at 224), 21 min.
- **B3 `dinov2b` (D22): DINOv2 ViT-B/14, plain settings, 224, backbone lr 2e-5, fp16, seed 0, T4.** Best epoch 11, early-stopped at 19, 53 s/epoch, 17 min. Weights 345 MB in the adikaul07 Drive `tiger-reid-runs/dinov2b_s0_20261113-0637/`.

| method (val) | rank-1 | rank-5 | mAP | AUROC | AUROC (all) | best ep | last-8-ep mean mAP / rank-1 / AUROC |
|---|---|---|---|---|---|---|---|
| **DINOv2-S 224 (plain)** | 76% [66–85] | 85% | **0.765** | 0.76 [0.68–0.84] | 0.82 | 11 (stop 19) | **0.721 / 73%** / 0.78 |
| B1 dinov2s_reg | 66% [54–76] | 78% | 0.723 | 0.79 [0.70–0.86] | 0.78 | 27 of 30 | 0.718 / 65% / 0.79 |
| B2 dinov2s_336 | 78% [69–88] | 84% | 0.740 | 0.75 [0.67–0.84] | 0.81 | 13 (stop 21) | 0.688 / 66% / 0.76 |
| **B3 dinov2b (ViT-B)** | **82% [73–91]** | 82% | **0.819** | 0.77 [0.69–0.85] | **0.84** | 11 (stop 19) | **0.748 / 78%** / 0.77 |
| dinov2b seed 1 (17:58) | 82% [73–91] | 82% | 0.772 | 0.77 [0.68–0.85] | 0.76 | 7 (stop 15) | 0.679 / 72% / 0.76 |
| dinov2b seed 1 rerun (18:37) | 78% [67–87] | 82% | 0.727 | 0.84 [0.77–0.91] | 0.85 | 13 (stop 21) | 0.643 / 69% / 0.76 |
| dinov2b seed 2 | 84% [75–91] | 84% | 0.803 | 0.83 [0.75–0.90] | 0.84 | 7 (stop 15) | 0.731 / 74% / 0.83 |
| **dinov2b, 4 runs (mean ± sd)** | 81 ± 3% | 83% | **0.780 ± 0.040** | 0.80 ± 0.04 | 0.82 | | 0.700 / 73% / 0.78 |

**Findings**
- **Higher resolution doesn't help.** B2 loses on the selection rule (val mAP 0.740 < 0.765). Its post-peak epochs are worse than the 224 run (mAP 0.69 vs 0.72), and it costs 1.7× the time. Its slightly higher best-epoch rank-1 (+1 image of 67) is noise.
- It overfits the same way as the plain 224 run (train loss 0.02 by epoch 19) and peaks around epochs 11–13 like it. So the limit is the 86 training tigers, not image detail.
- **Pattern so far:** every DINOv2-S variant levels off at val mAP ~0.70–0.72, and single-epoch peaks swing ±0.03. Picking configs by one peak on 67 queries is noisy, so the last-8-epoch mean is reported next to it.
- Harmless warning with fp16: "lr_scheduler.step() before optimizer.step()". GradScaler skips the first optimizer step while it calibrates the loss scale, which only drops the first LR value of warmup.
- Log: `results/train_logs/dinov2s_336_s0.csv`; weights in the adikaul07 Drive `MyDrive/tiger-reid-runs/dinov2s_336_s0_20261113-0044/`.

- **ViT-B is the first change that clearly helps:** val mAP 0.819 vs 0.765 at the best epoch, and 0.748 vs 0.721 on the last-8-epoch average (rank-1 78% vs 73%). It wins by both views, not just one lucky epoch. CIs still overlap (one seed), so seeds 1–2 come next.
- It learns faster (rank-1 82% by epoch 2) and memorizes faster (train loss 0.006 at epoch 19), so early stopping matters.
- rank-5 = rank-1 (82%): the 12 missed queries aren't near-misses. Their true tiger is outside the top 5, so these are hard images (pose/lighting), not ranking noise.
- **Open-set AUROC doesn't move with training** (0.75–0.79 for every model). The "is this tiger new?" score needs its own fix, which is Stage A.
- Log: `results/train_logs/dinov2b_s0.csv`.

- **Evening: `dinov2b` seed 1, run twice.** The notebook was run twice with `SEED = 1` (13 and 19 min). Same seed, same code (68ebbc5), same T4 type, yet the curves split from epoch 1 (mAP 0.441 vs 0.416), because cuDNN/fp16 GPU kernels aren't bit-deterministic. Both runs are kept and logged, and neither is picked after the fact.
- **Run-to-run noise is as big as the model differences we've been reading.** Best-epoch mAP across the 3 ViT-B runs is 0.819 / 0.772 / 0.727 (sd 0.046), and the AUROC of the same-seed pair is 0.77 vs 0.84. B3's "clear win" over DINOv2-S (0.765, one run) was largely seed 0 luck: mean 0.773 vs 0.765, and the last-8 mean is 0.690 vs 0.721. **ViT-B vs ViT-S is undecided until DINOv2-S also has 3 runs.**
- Lesson: with 67 val queries, decide configs on ≥3 runs per arm, not one. The B1/B2 "no gain" verdicts were within this noise too, but they were also slower or no better on the last-8 mean, so they stay dropped.
- Logs: `results/train_logs/dinov2b_s1.csv`, `dinov2b_s1_rerun.csv`.
- **Night: `dinov2b` seed 2:** best epoch 7, mAP 0.803, rank-1 84%, AUROC 0.83. The 4-run mean is val mAP 0.780 ± 0.040, against DINOv2-S at 0.765 (1 run). ViT-B is probably a bit better, but the gap (+0.015) is smaller than run noise, so DINOv2-S needs seeds 1 and 2 before we pick. Best epochs are 7–13, so 30 epochs with patience 8 is plenty. Log: `results/train_logs/dinov2b_s2.csv`.
- **DINOv2-S seed 1:** rank-1 76%, rank-5 79%, mAP 0.742, AUROC 0.68 [0.58–0.78] (lowest open-set score of any trained run), best epoch 13, stop 21, last-8 mAP 0.675. Two S runs average mAP 0.753, against 0.780 over four B runs. B leads on every view so far, and S seed 2 completes the comparison. Log: `results/train_logs/dinov2s_arcface_s1.csv`.
- **DINOv2-S seed 2:** rank-1 73%, mAP 0.738, AUROC 0.78, best epoch 16, stop 24, 15 min. DINOv2-S over 3 runs: mAP 0.748 ± 0.014 (much steadier than ViT-B's ± 0.040), rank-1 75%, AUROC 0.74 ± 0.06.
- **Decision (pre-declared rule, mean val mAP over runs):** ViT-B 0.780 > ViT-S 0.748, so `dinov2b` is the base. ViT-B also leads on rank-1 (+6 pts) and AUROC (+0.06). Not significant on mAP (Welch t ≈ 1.5, n = 4 vs 3). Last-8 means tie (0.700 vs 0.703): the bigger model peaks higher, then overfits to the same level. Stage A uses `dinov2b_s0.pt`, as named in D22 before the other seeds ran (picking the best-scoring seed now would tune on val). Log: `results/train_logs/dinov2s_arcface_s2.csv`.

**Next**
- Stage A (D18) on `dinov2b_s0.pt`, then test once on dinov2b seeds 0–2. Pick S vs B on the 3-run mean val mAP, then Stage A (D18) on that model.

---

## Nov 12, 2026 (Thu) — first full training run (Colab)

**Done**
- Colab pipeline working: `notebooks/colab_train.ipynb` (opened from Drive since the repo is private; token via Colab secret; Drive mounted at the start). Pinned `timm==1.0.30`, `pytorch-metric-learning==2.9.0` + a fail-fast version check.
- **ResNet50 + ArcFace, seed 0, 30 epochs on a T4 (32 s/epoch, 16 min).**

**Results (val, 67 queries / 46 matched unknowns, 95% CI)**

| method | rank-1 | rank-5 | mAP | AUROC |
|---|---|---|---|---|
| ResNet50 untrained | 37% | 60% | 0.22 | 0.51 |
| DINOv2 untrained | 58% | 75% | 0.31 | 0.52 |
| SIFT + RANSAC | 45% | 52% | 0.24 | **0.83** |
| **ResNet50 + ArcFace** | **70% [60–81]** | **84%** | **0.69** | 0.75 [0.67–0.84] |

**Findings**
- Training triples mAP (0.22 → 0.69) and lifts rank-1 by 33 points over the same backbone untrained. Biggest gain in the project so far.
- Open-set: 0.51 → 0.75, so the trained embedding now gives a meaningful "same tiger" distance. SIFT is still better at spotting new tigers on val (0.83), so SIFT fusion is worth trying.
- Val mAP climbed fast to ~0.6 by epoch 8, then slowly to 0.69; best epoch = 30 (last), LR had decayed to 0. More epochs may add a little.
- Log: `results/train_logs/resnet50_arcface_s0.csv` (values from run.log, 3 decimals).
- Decisions D15 (DINOv2 next, lr 3e-5), D16 (SIFT's 0.83 val AUROC is mostly small-sample luck; fusion = top-k re-scoring), D17 (run order).

**DINOv2-S + ArcFace (lr 3e-5, D15), seed 0, T4, 12 min**

| method (val) | rank-1 | rank-5 | mAP | AUROC | AUROC (all) |
|---|---|---|---|---|---|
| ResNet50 + ArcFace | 70% [60–81] | 84% | 0.69 | 0.75 [0.67–0.84] | 0.76 |
| **DINOv2-S + ArcFace** | **76% [66–85]** | **85%** | **0.76** | 0.76 [0.68–0.84] | **0.82** |

- DINOv2 starts far ahead (epoch 1: mAP 0.37, rank-1 61%, vs ResNet50's 0.18 / 43%), peaks at epoch 11, early-stopped at 19.
- **Overfits fast:** train loss → 0.02 by epoch 19 (ResNet50 still 1.3 at epoch 30); val mAP bounces 0.70–0.77 after epoch 11. Picking the max of a noisy curve on 67 queries flatters the val number slightly (both models), so test is the honest comparison.
- Gains over ResNet50 (+6 rank-1, +0.07 mAP) are inside the CIs for one seed; AUROC is a tie. Promising, not yet proven.
- Problem → fix: exposed GitHub token in a git error (token revoked/replaced); notebook now passes it via a credential helper, and resets a reused runtime to `origin/main` instead of pulling.

**B1 `dinov2s_reg` (D19): DINOv2-S + RandomErasing + crop scale 0.5–1 + drop-path 0.1, seed 0, T4 (adikaul07 Colab), 30 epochs, 19.5 min**

| method (val) | rank-1 | rank-5 | mAP | AUROC | AUROC (all) | best ep | last-8-ep mean mAP / rank-1 / AUROC |
|---|---|---|---|---|---|---|---|
| DINOv2-S + ArcFace | 76% [66–85] | 85% | 0.76 | 0.76 [0.68–0.84] | 0.82 | 11 (stop 19) | 0.721 / 73% / 0.78 |
| **B1 dinov2s_reg** | 66% [54–76] | 78% | 0.72 | 0.79 [0.70–0.86] | 0.78 | 27 of 30 | 0.718 / 65% / 0.79 |

- **Regularization stopped the memorizing** (train loss 1.8 at epoch 30 vs 0.02 at epoch 19) **but didn't improve val.** It learns more slowly (mAP 0.58 vs 0.69 at epoch 8), then **plateaus at ~0.72 from epoch 19**, so more epochs wouldn't help.
- **The first DINOv2 run's 0.765 was a one-epoch spike.** Averaged over each run's last 8 epochs, mAP is tied (0.721 vs 0.718). B1's curve is smooth, so its number is less inflated by picking the best epoch.
- **Rank-1 is clearly worse** (−10 pts at the chosen epoch, −8 pts on the last-8 average; flat at 65.7% for the last 10 epochs). Open-set AUROC is +0.03 matched / −0.04 all-unknowns: no real change.
- By the D13/D19 rule (val mAP), **B1 loses; the plain DINOv2-S config stays the base.** Log: `results/train_logs/dinov2s_reg_s0.csv`; weights in the adikaul07 Drive `MyDrive/tiger-reid-runs/dinov2s_reg_s0_20261112-2346/`.
- B2/B3 switched to the plain settings (`dinov2s_336`, `dinov2b`; D22).

**Problems → fixes**
- Drive connector was on the umich account → switched to adikaul07, so run folders can now be read directly.
- "Notebook not found" from the GitHub link (private repo) → uploaded the notebook to Drive.
- pml 3.0 on Colab turned `MPerClassSampler` into a batch sampler (`length_before_new_iter` gone) → pin 2.9.0 + version guard.
- Drive auth pop-up timed out after training; then the runtime was recycled and the run lost → mount Drive at the start.

---

## Nov 11, 2026 (Wed) — Phases 2–3: evaluation + baselines; Phase 4 started

**Phase 4 (afternoon)**
- `model.py` (`EmbeddingNet`: timm → Linear → BN → L2, 512-d), `training.py` (ArcFace / triplet+semi-hard miner, `MPerClassSampler` m=4, AdamW + warmup/cosine, early stop on val mAP, guard asserts for train-only / no open-set / no flips), `scripts/train.py`, `configs/*.yaml`, smoke test. Decisions D12–D13. 38/38 tests.
- **Sanity run** (ResNet50+ArcFace, 3 epochs, val): loss 34.8 → 24.4, val mAP 0.18 → 0.29 → **0.31** (untrained 0.22), rank-5 0.75. Log: `results/train_logs/sanity_resnet50_arcface_3ep.csv`.
- **Full local run died at epoch 3** (no error; probably sleep / memory on 8 GB): val mAP 0.18 → 0.29 → **0.40**, rank-1 0.55, AUROC 0.66. Log kept as `results/train_logs/local_interrupted_resnet50_arcface_s0.csv`. → Moved training to Colab: `notebooks/colab_train.ipynb` + `train.py --num-workers`, GPU name logged (D14).
- Speed on the M2: fp32 0.84 s/step (bs 32), bf16 autocast 1.15 s, channels_last 10 s (!) → stay fp32. ~70 s/epoch incl. val pass; first epochs slower (warm-up / memory pressure on 8 GB).

**Done (morning)**
- **Evaluation module** (`metrics.py`): rank-k, mAP, open-set AUROC, 95% bootstrap CIs; `evaluate()` takes any distance matrix, `evaluate_sets()` takes embeddings. `splits.eval_sets()` builds gallery/query/unknown sets (D9–D10). `results.log_result()` appends runs to `results/results.csv` with seed, config and commit.
- **Dataset** (`dataset.py`): `ReidDataset`, `eval_transform` / `train_transform` (no flips, tested), `extract_embeddings()`, `pick_device()` (mps).
- **Baselines:** `scripts/eval_chance.py` (random floor), `scripts/eval_pretrained.py` (any timm model, untrained), `sift_baseline.py` + `scripts/eval_sift.py` (SIFT, ratio 0.75, homography RANSAC, inlier count; untuned; ~2.3 s/query on 8 cores).
- `scripts/summarize_results.py` → `results/summary.md`.
- Learning notes: [`02-metrics.md`](learning/02-metrics.md), [`00-project-story.md`](learning/00-project-story.md). Decisions D9–D11.
- **33/33 tests pass.**

**Results (test, 95% CI)**

| method (no training) | rank-1 | rank-5 | mAP | AUROC (matched) | AUROC (all) |
|---|---|---|---|---|---|
| chance (3 seeds) | 1.1% | 1–4% | 1.5% | 0.48 | 0.52 |
| ResNet50 ImageNet | 42.9% [33–53] | 62.6% | 0.304 | 0.63 [0.49–0.75] | 0.51 |
| DINOv2 ViT-S/14 | 50.5% [40–62] | 61.5% | 0.375 | 0.53 [0.41–0.67] | 0.57 |
| SIFT + RANSAC | 53.8% [43–64] | 60.4% | 0.311 | 0.645 [0.55–0.73] | 0.60 |

**Findings**
- Untrained features already match about half the queries (they capture shape/pose/coat), but the three baselines' rank-1 CIs overlap, so there's no clear winner. DINOv2 has the best mAP.
- **Only SIFT spots new tigers above chance.** An inlier count is an absolute "same pattern" score; untrained embedding distances have no such meaning. That gap is what metric learning has to close.
- **Open-set protocol artifact:** with all open-set images as unknowns, ResNet50 AUROC was 0.37 (below chance), because known queries are deliberately atypical (D5) while unknowns are mostly typical. Matched unknowns → 0.51. Fixed (D10), but the matched test set is only **19 images / 6 entities**, so AUROC CIs are ~±0.1 (even random embeddings span 0.28–0.66).
- Val→test swings are large (SIFT AUROC 0.83 → 0.65): small-sample noise.
- Input size barely matters for untrained ResNet50 (val mAP 0.219 / 0.192 / 0.212 for 224², 224×448, 288×576) → 224×224 (D11).

**Problems → fixes**
- Unfair open-set comparison → matched unknowns + `auroc_all` + bootstrap CIs (D10); old results.csv rows deleted and re-run.
- DINOv2 crashed ("input height 224 doesn't match 518") → pass `img_size` to timm for ViTs.
- zsh doesn't word-split `$var` in `for` loops → `${sz%x*} ${sz#*x}`.

**Next**
- Phase 4: metric learning (see START HERE).

---

## Nov 10, 2026 (Tue) — Part 0 → Phase 1

**Mode switch:** due dates dropped; moving to **rapid development**: Claude writes the code, explains briefly, tests + commits + pushes after every step, and checks in at phase boundaries. `CLAUDE.md` and `docs/roadmap.md` rewritten (Phases 0–8, no dates; Part 1 classifier dropped). The TODO skeleton below gets filled in by Claude.

**Done**
- Switched to rapid mode (CLAUDE.md, roadmap, README status).
- **Pseudo-clips implemented** (`pseudo_clips.py`: 32×32 mean-centered thumbnails → cosine → connected components within each entity; `assign_pseudo_clips()`), 5 pytest tests. `scripts/make_pseudo_clips.py` sweeps thresholds → `results/pseudo_clips/sweep.csv` + grids.
- Finding: **raw-pixel cosine is useless** for this: within-entity median 0.885 vs between-entity 0.873. **Mean-centered** separates: 0.38 vs 0.09 (99th pct between = 0.52).
- Sweep: t=0.4 → 260 clips, 47 entities with 1 clip; t=0.5 → 439 clips, 28 with 1 clip; t≥0.7 → mostly singleton clips (too fine, leaks).
- Visual check: t=0.5 still splits same-scene frames (entity 27); t=0.4 is better but not perfect, and it chains entity 153's 98 images (day + dark) into one clip, which is the safe direction. Chose **t=0.4** (see `docs/decisions.md` D4).
- **New `docs/decisions.md`**: a log of every big decision (D1–D8 so far) with why, alternatives and consequences.
- **Splits frozen: `splits/splits_v1.csv`** (+ `.json` with settings/counts) via `scripts/make_splits.py`: train 1,379 / val 67 / test 91 / openset_val 221 / openset_test 129 images. `tiger_reid.splits.load_splits()` reads it.
- `tests/test_splits.py`: on the real file, checks every image appears once, no clip spans two splits, open-set entities are never in known splits, test/val entities have training images, and no test image is >0.4 similar to a same-entity train image. 11/11 tests pass.
- Made `tiger_reid` importable: added `pyproject.toml` and ran `pip install -e .` in `.venv`. Notebooks can now `from tiger_reid... import ...`.
- `src/tiger_reid/data.py`: `load_labeled()` returns the 1,887 labeled images (`entity_id`, `filename`, `path`). Checked: 1,887 rows, 107 entities.
- `src/tiger_reid/pseudo_clips.py`: skeleton for pseudo-clips. **Three TODOs are mine**: `image_features`, `cosine_similarity_matrix`, `group_by_threshold`. `show_clips` (plotting) is done.
- New learning note: [`01-pseudo-clips.md`](learning/01-pseudo-clips.md) (which features detect "same video", chaining, how to pick a threshold without labels).

**Findings**
- **Schedule slip:** 5 weeks since day 1, and Part 0 is still open. The plan has Part 3 starting now; the core freeze is ~Dec 13. Logged in roadmap "Need to know".
- Crops vary a lot in size (≈150–900 px per side), so features must resize to a fixed size before comparing.

**Problems → fixes**
- First split put the **biggest** clip into test for 2-clip entities (tie in "distinctness"), giving test 236 / val 274 images and starving training. Caught by the toy unit test → rule: the largest clip always stays in train. Re-built v1 (no results existed yet).
- `import tiger_reid` failed (package not installed) → `pyproject.toml` + editable install.

**My to-dos**
- [x] Decide the re-plan → rapid mode
- [x] Choose pseudo-clip threshold → 0.4
- [ ] Optional: glance at `results/pseudo_clips/*.png` to see what the clips look like
- [ ] Carry-overs from Oct 6: tidy notebook 00 Findings cell; read ATRW paper Sec. 3; email a professor; GPU access

**Next**
- Phase 2: `metrics.py` (rank-1/5, mAP, open-set AUROC) + hand-checked unit tests.
- Phase 3: `ReidDataset`, then the off-the-shelf ResNet50 and SIFT baselines.

---

## Oct 6, 2026 (Mon) — Part 0, day 1

**Done**
- Created the private repo [Adi-Kaul/tiger-reid](https://github.com/Adi-Kaul/tiger-reid) with README, CLAUDE.md (learn-as-we-go rules), roadmap, and learning notes.
- Wrote `scripts/download_atrw.py` and downloaded the ATRW re-ID set (train images, train annotations, test images) into `data/atrw/`.
- Downloaded the ATRW paper to `docs/papers/` (local only).
- Set up the local `.venv` (PyTorch, timm, pytorch-metric-learning, OpenCV all import; Apple `mps` GPU available).
- Made `notebooks/00_explore_atrw.ipynb` (works in Colab and locally) and opened it in Colab.

**Findings**
- The re-ID **test set has no ID labels** on LILA → we'll build our own query/gallery/open-set splits from the labeled train set.
- **No clip/video info** anywhere: CSV is just `id, filename`; keypoint JSON is pose only. Sorted filenames share an ID with their neighbor only 1.3% of the time (≈ random), so filenames are shuffled.
- **Images per entity: min 10, median 13, max 98** (mean ≈ 17.6). Skewed: most entities have 10–15 images, a few have far more. Use a balanced (m-per-class) sampler so big entities don't dominate training.
- **1,887 labeled images** out of 3,392 in the train folder. The other 1,505 have no ID or keypoints, so we're ignoring them.
- **107 IDs = 107 "entities", not tigers.** Paper: an entity is one *side* of one tiger, because left and right flanks have different stripes and wild cameras rarely see both. Full re-ID set = 182 entities from 92 tigers (3,649 boxes); LILA's labeled train release has 107 of them (1,887 boxes), the rest are in the unlabeled test set.
- We **can't tell which two entities are the same tiger**, so "hold out individuals" really means "hold out entities": a tiger's other flank may still be in training. Fine for stripe matching (different patterns), but say so in the report.
- Paper says original crops were named with camera, shot and frame IDs; the LILA release renamed them to plain numbers, which is why clip info is gone. Frames were also sampled 1-in-10 and filtered for being "significantly different", so near-duplicates may be milder than feared.
- **Grid of 20 entities × 6 images (Part 0 checkpoint) — what I saw:** most rows look like they come from one video; a few mix two videos. Poses vary a lot: in water, sitting, some views from behind. → Near-duplicates are common, so pseudo-clips are needed; many entities may only have one clip; same video = same background, so the model could cheat on background; water/back views are hard cases for the per-pose breakdown.
- Data dates to 2019 because ATRW was released for the CVWC 2019 challenge; benchmarks stay frozen on purpose.

**Problems → fixes**
- Colab "Open from GitHub" returned 404 → repo is private and Colab wasn't authorized for private repos. Fix: check "Include private repos" and authorize (or revoke Colaboratory in GitHub settings and redo).
- No clip info for splitting → plan: reconstruct **pseudo-clips** by grouping near-duplicate images within each ID, then split by those.

**My to-dos**
- [x] Notebook Q1–Q3 answered
- [x] Part 0 checkpoint: 20-entity grid
- [ ] Tidy Findings cell + add Observations cell (fill in "skewed", remove leftover hint text) and save notebook to GitHub
- [ ] Read the ATRW paper (Sec. 3 explains entities; 107 vs 92 answered)
- [ ] Email a wildlife ecology or CV professor
- [ ] Set up GPU access (cluster account or Kaggle)

**Next**
- **Open question for me:** what visual clues tell me two photos are from the same video? (Turn that into something measurable → pseudo-clips.)
- Optional: save the grid as `results/part0_grid.png`, clear notebook outputs before future Colab saves (notebook is 6.5 MB).
- Then design the splits (pseudo-clips, open-set holdout of ~20% of individuals) and save them to a file.
- Dataloader skeleton for me to fill in → Part 0 checkpoint (20 tigers × several images).
