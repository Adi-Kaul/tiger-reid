# Decision log

Every choice that shapes the results, with the reason and the cost. Newest at the bottom. Use this to explain the project's "why" (report, presentation, viva).

Format: **Decision**, **Why**, **Alternatives**, **Consequences**.

---

## D1. Treat ATRW IDs as entities (one flank of one tiger), not tigers (Oct 6, 2026)
- **Decision:** The 107 IDs are "entities". Held-out "individuals" in the open-set test are held-out entities.
- **Why:** The ATRW paper defines an entity as one side of one tiger, since left and right stripe patterns differ. LILA doesn't say which two entities belong to the same tiger.
- **Alternatives:** None possible without the original metadata.
- **Consequences:** An open-set entity's *other* flank may be in training. That's fine for stripe matching (the patterns are different), but the report must say so.

## D2. Build our own splits from the labeled train set (Oct 6, 2026)
- **Decision:** Use only the 1,887 labeled train images and build our own train/val/test/open-set splits from them.
- **Why:** LILA's ATRW re-ID test set has no identity labels, so it can't be scored.
- **Alternatives:** Submit to the old CVWC 2019 evaluation server (closed).
- **Consequences:** Our numbers aren't directly comparable to the ATRW leaderboard. Comparisons are within this project, on one frozen split.

## D3. Reconstruct pseudo-clips from 32×32 mean-centered thumbnails (Nov 10, 2026)
- **Decision:** Within each entity, link images whose thumbnail cosine similarity is above a threshold. Connected groups are "pseudo-clips". Splits are made by pseudo-clip, never by frame.
- **Why:** ATRW images are video frames. Frames from one video are near-duplicates (same background, pose, light). With one in training and one in test, the model "recognizes" by memory, not stripes, and scores are inflated (**leakage**). The clip info was lost in the LILA release, so we approximate it.
  - Thumbnails keep the scene layout and drop the stripes, so they measure "same video", not "same tiger".
  - **Mean-centered**, because raw pixels give every pair a cosine of ~0.75–0.97: same-entity median 0.885 vs different-entity 0.873, no separation. Centered gives 0.38 vs 0.09.
- **Alternatives:** Perceptual hashes (similar idea); pretrained CNN features (group by content/pose, not scene, so they'd merge different videos with the same pose).
- **Consequences:** Pseudo-clips are an approximation, and some leakage may remain. State this as a limitation.

## D4. Pseudo-clip threshold = 0.4 (Nov 10, 2026)
- **Decision:** Freeze the threshold at **0.4**.
- **Why:** The threshold is "how similar must two images be to count as the same video".
  - **Too high (under-merging):** one real video is cut into several fake clips; pieces land in both train and test → **hidden leakage, inflated scores.** This is the dangerous error, because the numbers look *good*. At 0.5 the grids already show same-scene frames split apart (entity 27).
  - **Too low (over-merging):** different videos glue into one clip (e.g. entity 153's 98 images = 1 clip). Nothing leaks, but an entity with only one clip can't give any test images → **smaller test set.** This is the safe error.
  - When unsure, choose the error that makes results honest-but-smaller rather than bigger-but-inflated.
- **Alternatives (sweep, `results/pseudo_clips/sweep.csv`):**

  | threshold | clips | entities with 1 clip | entities usable for test |
  |---|---|---|---|
  | 0.35 | 214 | 59 | 48 |
  | **0.40** | **260** | **47** | **60** |
  | 0.45 | 328 | 35 | 72 |
  | 0.50 | 439 | 28 | 79 |
  | 0.70 | 1,112 | 3 | 104 (mostly single images, which would leak heavily) |
- **Consequences:**
  - Closed-set test is small: ~50 known entities × usually 1 image. One query is roughly 1–2% of rank-1, so differences of a few points are noise. Report the spread over several seeds.
  - The open-set test is unaffected (held-out entities contribute all their images).
  - Some residual leakage is possible (e.g. entity 27's fence frame still splits off at 0.4).

## D5. Test/val clip = the entity's least similar clip (Nov 10, 2026) (default, can revisit)
- **Decision:** For a known entity with ≥2 pseudo-clips, the **largest clip always stays in train**. Of the remaining clips, the one whose best match to the entity's other clips is lowest goes to **test**; with ≥3 clips, the next least similar goes to **val**. Ties go to the smaller clip.
- **Why the largest-clip rule:** the first version had no such rule. With exactly 2 clips, both are equally "distinct" (it's the same closest pair seen from either side), so the tie-break often sent the *big* clip to test, removing most of an entity's training images (test 236 / val 274 images). A unit test caught it, and the rule was added before any results existed.
- **Why:** An extra guard against D4's residual leakage: the most visually distinct scene is the least likely to be a split-off piece of a training video.
- **Alternatives:** A random clip: more typical test images, but sometimes picks a near-duplicate of training.
- **Consequences:** Extra clips are often single odd images (close-ups, dark frames, odd angles), so the test set leans **harder than average** and scores will look somewhat pessimistic. Say so in the report. Changing to random only needs a re-run of `scripts/make_splits.py` with a new split version (`v2`), and it must happen before any results are compared.

**Resulting split (`splits/splits_v1.csv`, seed 0, threshold 0.4):**

| split | images | entities |
|---|---|---|
| train | 1,379 | 86 |
| val | 67 | 27 |
| test | 91 | 47 |
| openset_val | 221 | 10 |
| openset_test | 129 | 11 |

## D6. Open-set holdout = 20% of entities, split half val / half test (Nov 10, 2026)
- **Decision:** Randomly (seed 0) hold out 20% of entities, never trained on. Half are `openset_val` (choose the "new tiger" distance threshold), half `openset_test` (report).
- **Why:** Open-set numbers are only valid if those tigers were never seen in training, and the threshold must not be tuned on the entities we report on.
- **Consequences:** ~86 known entities left for training. All of the holdout's images count, so the open-set test is larger than the closed-set one.

## D7. Rapid-development mode; drop the plain-classifier part (Nov 10, 2026)
- **Decision:** Switch from learn-first to results-first. Drop the old Part 1 "train a classifier to see it fail" exercise.
- **Why:** The project was well behind the original plan. The open-set experiment makes the same point (a classifier can't say "new tiger").

## D8. No horizontal-flip augmentation (Nov 10, 2026)
- **Decision:** Never flip images horizontally in training.
- **Why:** A flipped left flank is a mirrored stripe pattern that no real tiger has. It can look like a *different* entity's flank, which confuses identity.

## D9. Evaluation protocol (Nov 11, 2026)
- **Decision:** Gallery = train images of the known entities. Query = `test` (or `val` while choosing settings). Unknown = `openset_test` (or `openset_val`). Score with cosine distance for embedding methods; SIFT passes its own distance matrix. Closed-set metrics are rank-1, rank-5 and mAP; open-set uses AUROC of nearest-gallery distance. Every run is appended to `results/results.csv` with its seed, config and git commit.
- **Why:** One shared scorer, so SIFT, off-the-shelf and trained models are compared on identical terms. The gallery is train-only, so queries and gallery always come from different pseudo-clips (no near-duplicate matches).
- **Alternatives:** Use val+test images in the gallery as well (a bigger gallery, but it mixes in images that are used as queries).
- **Consequences:** Chance level is rank-1 ≈ 1.1%, AUROC ≈ 0.5, measured with random embeddings. With 91 known queries and 129 unknowns, **even noise moves AUROC by ±0.03 between seeds**, so main results are reported as mean ± spread over ≥3 seeds.

## D10. Open-set AUROC uses matched unknowns; all-unknowns reported second; bootstrap CIs (Nov 11, 2026)
- **Decision:** The primary `auroc` uses open-set images **outside each new entity's largest clip** (`eval_sets()["unknown"]`: 46 val / 19 test images). `auroc_all` (every open-set image) is reported second. Every evaluation also reports 95% bootstrap CIs for rank-1 and AUROC.
- **Why:** Known-tiger queries are each entity's least typical clip (D5): close-ups, dark frames, odd angles, all far from the gallery. With *all* new-tiger images as unknowns (mostly typical shots), known tigers look "newer" than new tigers. ResNet50 got AUROC **0.37, below chance**. Matching the unknowns to the same kind of image gives **0.51**. The comparison has to be like-for-like.
- **Alternatives:** All unknowns only (more images but biased against every model); change D5 to random test clips (more leakage risk).
- **Consequences:** The primary test AUROC rests on **19 images from 6 entities**, so it's very noisy: even random embeddings span roughly 0.28–0.66 (95% CI). Open-set claims need CIs and multiple seeds, and only big differences count. Old `results.csv` rows (old protocol) were deleted and re-run.

## D11. Off-the-shelf baselines: ResNet50 and DINOv2 ViT-S at 224×224 (Nov 11, 2026)
- **Decision:** Two untrained baselines: ImageNet ResNet50 and self-supervised DINOv2 ViT-S/14, pooled features + cosine, 224×224 input.
- **Why 224×224:** On val, ResNet50 at 224×224 / 224×448 / 288×576 had overlapping CIs; we choose by **val mAP** (the most stable metric): 0.219 / 0.192 / 0.212 → 224×224.
- **Why DINOv2 too:** a stronger "free" bar. A trained model must beat the best thing you get without training, not just ImageNet.

## D12. Embedding model and losses (Nov 11, 2026)
- **Decision:** `EmbeddingNet` = timm backbone (pooled features) → Linear → BatchNorm → L2-normalize, 512-d. Two losses: **ArcFace** (margin 28.6°, scale 64, pml defaults) and **triplet** (margin 0.2, semi-hard miner).
- **Why:** A normalized embedding plugs into the same cosine evaluation as the baselines (D9). ArcFace is the standard strong choice for re-ID with few images per class; triplet is the classic metric-learning baseline to compare against. BatchNorm before normalizing is the usual "BNNeck"-style trick that stabilizes ArcFace training.
- **Alternatives:** Sub-center ArcFace, circle loss, multi-similarity loss (later, if ArcFace plateaus); no projection head (use raw 2048-d features).
- **Consequences:** ArcFace learns one weight vector per *training* entity; those are thrown away after training, so new tigers still work (we only keep the embedding).

## D13. Training protocol (Nov 11, 2026)
- **Decision:** `MPerClassSampler` (m=4 images per entity, batch 32 = 8 entities); AdamW (backbone lr 1e-4, head 1e-3, ArcFace weights 1e-2, wd 1e-4); 1-epoch linear warmup then cosine; up to 30 epochs; **early stopping on val mAP** (patience 8, best checkpoint kept). Augmentation = `train_transform()` (crop/scale, ±10° rotation, colour jitter, **no flips**). Test is evaluated **once per final config** (`--eval-test`), never during training. The training script asserts: train split only, no open-set entity, no flip transform.
- **Why:** Balanced batches guarantee positive pairs and stop entity 153 (98 images) dominating. mAP is the most stable metric on our small val set (D11). Keeping test out of the loop is what makes the test numbers honest.
- **Alternatives:** Select on val rank-1 (noisier); select on open-set val AUROC (would use open-set entities for model selection, which our rules forbid for training choices beyond the threshold).
- **Consequences:** Val has only 67 queries, so "best epoch" is itself noisy; we report test for the selected epoch regardless of how it looks.

## D14. Full training runs on Colab GPU (Nov 11, 2026)
- **Decision:** Full training runs go through `notebooks/colab_train.ipynb` on a Colab T4: clone the repo (read-only token as a Colab secret), download ATRW, run `scripts/train.py --num-workers 2`, and copy the train log, new `results.csv` rows and checkpoint to Google Drive. The rows are then merged into the repo's `results/results.csv` locally. The GPU name is logged in each row's config (`device`).
- **Why:** The 8 GB M2 is too slow (~70 s/epoch, 30 epochs ≈ 35 min per run) and the first full ResNet50 run died silently after epoch 3. Ablations and ≥3 seeds need cheap runs.
- **Alternatives:** Keep training on mps with checkpoint resume (slow, still fragile); a paid cloud GPU (unnecessary at this data size).
- **Consequences:** CUDA and mps don't give bit-identical numbers for the same seed, so compare models run on the same device type. Same code, same frozen split file, same seeds.

## D15. Second model: DINOv2 ViT-S/14 + ArcFace, backbone lr 3e-5 (Nov 12, 2026)
- **Decision:** After ResNet50+ArcFace (val rank-1 70%, mAP 0.69, AUROC 0.75), the next run is `vit_small_patch14_dinov2.lvd142m` + ArcFace, same protocol (D13) except **backbone lr 3e-5** (ResNet50 used 1e-4). One run at 3e-5 only, no lr sweep.
- **Why DINOv2:**
  1. Best untrained features: val rank-1 58% vs 37%, mAP 0.31 vs 0.22 for ResNet50. A better start usually means a better finish.
  2. Self-supervised pretraining (142M images, matching patches/views) favors fine local texture, which is what stripe patterns are; ImageNet-label pretraining (ResNet50) only needs "is it a tiger", not "which tiger".
  3. DINOv2 features are among the strongest in recent wildlife re-ID comparisons.
  4. CNN vs ViT at similar size (22M vs 25M params) is a fair, reportable comparison.
- **Why lr 3e-5:** 1e-4 is on the high side for fine-tuning a pretrained ViT on ~1,300 images and risks overwriting what it already knows; common practice is 1e-5–5e-5. Chosen from standard practice before seeing any DINOv2 training result, not tuned on val or test.
- **Alternatives:** Same lr as ResNet50 (1e-4; identical settings, but likely suboptimal for a ViT); sweep {1e-4, 3e-5} and pick on val (cleaner but one extra run; do it later if DINOv2 underperforms); layer-wise lr decay (more knobs).
- **Consequences:** ResNet50 and DINOv2 differ in backbone lr as well as architecture, so the report says "each with a standard fine-tuning lr", not "identical settings". Risks: the untrained head start may vanish after fine-tuning; ViTs can be unstable on small data. If DINOv2 ends up below ResNet50 on val, try 1e-4 before concluding.

## D16. How to read open-set AUROC, and the plan for SIFT fusion (Nov 12, 2026)
- **Decision:** Treat SIFT's val AUROC of 0.83 as a hint, not as evidence that SIFT beats the trained model at open-set. SIFT fusion stays on the list as a modest experiment: re-score only the embedding's **top-k (~10)** gallery matches with SIFT inlier counts and combine the scores; choose k and the weighting on val/openset_val.
- **Why:**
  1. The same untuned SIFT dropped from 0.83 on val to 0.645 on test: small-sample luck.
  2. Unknown sets are tiny (val 46 images / ~5 tigers, test 19 / 6), so a few odd images move AUROC by ~0.1.
  3. CIs overlap on val: SIFT [0.75, 0.90] vs ResNet50+ArcFace [0.67, 0.84].
  4. Still worth trying: SIFT measures exact stripe-point alignment while the embedding measures overall appearance, so they tend to make different mistakes and fusing them can help.
  5. Full SIFT costs ~2.3 s per query vs milliseconds for embeddings; checking only the top-k keeps it practical.
- **Scale used when describing AUROC:** 0.5 = chance; 0.6–0.7 weak; 0.7–0.8 useful but many errors; 0.8–0.9 good; 0.9+ strong.
- **Alternatives:** Full SIFT over the whole gallery (slow); drop SIFT (loses a complementary signal).
- **Consequences:** Open-set claims in the report come with CIs and ≥3 seeds; the real comparison is on test after the final model is chosen.

## D17. Phase 4 run order (Nov 12, 2026)
- **Decision:** (1) DINOv2+ArcFace, (2) ResNet50+triplet, (3) pick the best on val, run it with 3 seeds, evaluate on test once, (4) SIFT top-k fusion, (5) more epochs / higher resolution if time allows.
- **Why:** DINOv2 has the best chance of being the winner, so it goes first; triplet is the standard loss comparison the report needs; test is touched once, only after the choice is made on val.
- **Consequences:** MegaDescriptor stays optional and is only run after checking whether its pretraining data includes ATRW (contamination).

## D18. openset_val may choose the open-set scoring rule, from a list fixed in advance (Nov 12, 2026)
- **Decision:** The "is this a new tiger?" score and any fusion weights are chosen on **val + openset_val** (matched unknowns, D10), from the candidates below, which are written down **before any of them is evaluated**. Training configs are still chosen on val mAP only (D13). Test is evaluated once, for the final pipeline.
- **Candidates:**
  - **S1 nearest:** distance to the nearest gallery image (the current score).
  - **S2 centroid:** distance to the nearest *entity mean embedding* (mean of that entity's L2-normalized gallery embeddings, re-normalized).
  - **S3 margin:** with d1 ≤ d2 the per-entity nearest-image distances of the best and second-best entities, score = −(d2 − d1). A known tiger has one clear winner; a new tiger is ambiguous.
  - **+TTA:** embeddings averaged over 5 fixed views (full image + 4 corner crops at 90% scale, **no flips**), then re-normalized.
  - **+ENS:** distance matrix = w·DINOv2 + (1−w)·ResNet50, w ∈ {0.25, 0.5, 0.75}.
  - **+SIFT:** SIFT inlier counts against the gallery images of the top-10 entities by embedding distance; fused score = z(embedding score) − w·z(log1p(max inliers)), w ∈ {0.25, 0.5, 1}; z-scores use the val known + unknown pool.
- **Selection protocol:** pick the best of S1–S3 by openset_val AUROC; then add TTA, ENS, SIFT one at a time, keeping each only if AUROC rises **≥ 0.01** and val rank-1 / mAP don't drop. Every candidate's numbers go in `results/openset_val_grid.csv`, kept or not.
- **Why:** The scoring rule is part of the open-set detector, the same role as the threshold (D6), not part of training. A short pre-declared list plus the ≥0.01 rule limits how much we can overfit 46 unknown images.
- **Alternatives:** Fix one rule by reasoning only (no selection; can't tell which idea helps until test); free search over many rules (would overfit openset_val).
- **Consequences:** openset_val AUROC of the chosen pipeline is optimistically biased (it was used to choose), so only test and the 5-fold estimate (D20) are reported as results.

## D21. Run order revised for open-set focus (Nov 12, 2026), replaces D17
- **Decision:** (A) better open-set scoring on the saved checkpoints (D18), (B) DINOv2 overfitting fixes / higher resolution / ViT-B on Colab, chosen on val mAP, (C) best config × 3 seeds → test once with the chosen scoring, (D) 5-fold rotated open-set estimate (D20). ResNet50+triplet and DINOv2+triplet become optional.
- **Why:** DINOv2 is the base model (val rank-1 76%, mAP 0.76); open-set AUROC (0.76) is the weak spot and the project goal is to push it toward 0.9. Scoring changes are cheap and need no training, so they go first.

## D19. Overfitting fixes, higher resolution, ViT-B, fp16 on CUDA (Nov 12, 2026)
- **Decision:** Three Colab runs, chosen on **val mAP** (D13), each also reporting openset_val AUROC as a secondary check:
  - **B1 `dinov2s_reg`:** DINOv2-S at 224 + RandomErasing (p 0.5, 2–15% of the image) + crop scale (0.5, 1.0) + drop-path 0.1.
  - **B2 `dinov2s_reg_336`:** B1 at 336×336.
  - **B3 `dinov2b_reg`:** DINOv2-B/14 + B1 settings at 224, backbone lr 2e-5 (a bigger model needs a gentler lr).
  - B2 and B3 use fp16 autocast on CUDA (`amp: true`), with the loss in fp32 (ArcFace's scale 64 overflows fp16).
- **Why:**
  - DINOv2-S hit train loss 0.02 by epoch 19 while val bounced around: it memorizes the 86 training tigers. Over-confident embeddings also make new tigers look "known".
  - Erasing and stronger crops stop it relying on one region (or the background).
  - Drop-path is the standard ViT regularizer.
  - 336 px gives 24×24 patches instead of 16×16, so finer stripe detail.
  - ViT-B is the next DINOv2 size up.
  - fp16 roughly halves T4 time and memory.
- **Alternatives:** Label smoothing / sub-center ArcFace (later if needed); freezing early blocks; fewer epochs.
- **Consequences:**
  - B1 changes only regularization vs the first DINOv2 run (same precision), so it is a clean comparison.
  - B2/B3 also switch to fp16, a small confound noted in the report.
  - Defaults (`drop_path 0`, no erase, scale 0.7–1, fp32) reproduce the earlier runs exactly.
- **Outcome B1 (Nov 12):** val mAP 0.723 (best epoch 27) vs 0.765 for the plain run, which loses by the selection rule. Over the last 8 epochs, mAP is tied (0.718 vs 0.721), rank-1 is lower (65% vs 73%) and AUROC is about the same (0.79 vs 0.78). The regularization stopped memorization (train loss 1.8 vs 0.02) without improving generalization. The plain DINOv2-S config stays the base. Whether B2/B3 keep these settings is pending the user's call.

## D22. B2/B3 use the plain DINOv2-S settings, not B1's regularization (Nov 12, 2026), revises D19
- **Decision:** B2 = `dinov2s_336` (plain DINOv2-S at 336×336) and B3 = `dinov2b` (DINOv2-B/14 at 224, backbone lr 2e-5). Both have no RandomErasing, crop scale 0.7–1, no drop-path, and early stopping as before. Both keep fp16 autocast (`amp: true`) for T4 time and memory. Configs renamed from `*_reg*`, which were never run.
- **Why:** B1 showed the regularization costs rank-1 (~8 pts on the last-8-epoch average) without gaining mAP or open-set AUROC. Using the plain settings means each run changes one thing (resolution or model size) against the current best config.
- **Alternatives:** Keep B1 settings as D19 planned (bigger inputs/models might need them more, but B1 is evidence against); drop-path only (a new untested mix).
- **Consequences:** B2/B3 compare cleanly with the plain `dinov2s_arcface` run except for fp16 (noted in the report). ViT-B may overfit faster; early stopping (patience 8) handles that, and if it does, regularization can be revisited for B3 only.
