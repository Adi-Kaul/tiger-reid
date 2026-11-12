# Progress log

One entry per working day, newest first. Each entry: what got done, findings, problems and how they were solved, what's left for me, and what's next.

---

## ▶ START HERE (handoff for the next session, updated Nov 11)

**Where we are:** Phases 0–3 done. **Phase 4 in progress.** Two full runs on Colab T4 (val, seed 0): **ResNet50+ArcFace** rank-1 70%, mAP 0.69, AUROC 0.75; **DINOv2-S+ArcFace (lr 3e-5)** rank-1 **76%**, mAP **0.76**, AUROC 0.76 (best epoch 11, early stop at 19, 12 min). DINOv2 leads on rank-1/mAP, ties on AUROC; one seed and overlapping CIs. Weights in Drive `MyDrive/tiger-reid-runs/`. Notebook in Drive: `colab_train_v4.ipynb` (older copies superseded). **Next (D17):** ResNet50+triplet, then best on val → 3 seeds → `--eval-test` once → SIFT top-k fusion (D16).
**Improvements on the table (roadmap Phase 4):** fine-tuned DINOv2, SIFT re-ranking/fusion of top-k (SIFT val AUROC 0.83 > trained 0.75), higher input resolution, more epochs (still creeping up at ep 30 when LR hit 0), ≥3 seeds.
**Mode:** rapid development (see `CLAUDE.md`). Claude writes code, explains briefly, runs `pytest -q`, commits + pushes after each verified step, and asks at phase boundaries.
**Read first:** [`learning/00-project-story.md`](learning/00-project-story.md) (the whole project in plain words + Phase 4 plan), [`decisions.md`](decisions.md) (D1–D11), `results/summary.md`.

**Phase 4 to-do, in order:**
1. `src/tiger_reid/model.py`: timm backbone → linear projection → 512-d L2-normalized embedding.
2. `scripts/train.py` + `configs/*.yaml`: `pytorch-metric-learning`, `MPerClassSampler` (m=4), ArcFace (or triplet + miner), AdamW, ~30 epochs, `train_transform()` (no flips), early stopping on **val mAP**, seed fixed, checkpoints → `checkpoints/` (gitignored), final eval on test via `evaluate_sets()` + `log_result()`.
3. `tests/test_train_smoke.py`: 2 batches × 16 images on CPU; loss finite, shapes right.
4. Runs: ResNet50+ArcFace → ResNet50+triplet → DINOv2-S+ArcFace (`vit_small_patch14_dinov2.lvd142m`, pass `img_size`) → MegaDescriptor → best × 3 seeds → `scripts/summarize_results.py`.
5. **Before MegaDescriptor:** check whether its pretraining data includes ATRW (likely). If yes, skip it or report separately as contaminated, and log a decision.

**Bar to beat (test):** rank-1 ~54% (SIFT), mAP ~0.38 (DINOv2), AUROC ~0.65 (SIFT). Test CIs are wide (rank-1 ±10 pts, AUROC ±0.1), so only claim clear wins.

**Useful commands:**
```bash
source .venv/bin/activate && pytest -q
python scripts/eval_pretrained.py --model resnet50 --size 224 224 --stage val
python scripts/eval_sift.py --stage test          # ~5 min
python scripts/summarize_results.py              # -> results/summary.md
```

**Wild data (Nov 12):** no public wild tiger re-ID dataset (images + individual IDs) found. Emails sent Nov 12 asking for wild data: Abishek Harihar (Panthera/NCF, Rajaji NP camera-trap DB on GBIF: 305 images, CC BY-NC, no IDs public, image server down), Wenshuo Li (Bristol, used Rajaji data in arXiv 2512.14878), LILA (info@lila.science). Not contacted (no verified address): WhoAmI authors (WCS India / Edinburgh, 32 wild tigers), China NTLNP camera-trap Amur tiger study. Other leads: wild leopard/hyena camera-trap re-ID sets (arXiv 2607.00804) as a zoo→wild proxy; Zenodo 5573206 = 10 years of Indian tiger capture histories (CSV, CC0, locations centred) for the map/SCR part. Any received data stays out of the repo; drop/coarsen coordinates.
**Account (Nov 12): use adikaul07@gmail.com for everything** (Drive, Colab, Gmail, git). Runs so far live in the umich Drive `MyDrive/tiger-reid-runs/`; new runs go to the adikaul07 Drive (upload `notebooks/colab_train.ipynb` there; separate Colab GPU quota). **Colab GPU quota hit (Nov 12)** on umich: B1 not run yet. Stage A needs `dinov2s_arcface_s0.pt` + Colab `resnet50_arcface_s0.pt` downloaded into `checkpoints/` (the local resnet50 file is the epoch-3 leftover).
**Still open (optional):** tidy notebook 00.

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

**Problems → fixes**
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
