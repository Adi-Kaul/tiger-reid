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
