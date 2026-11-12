# Roadmap

Rapid-development mode (since Nov 10, 2026): no due dates, aim for results fast, and test, commit and push after every step. Each phase ends with a checkpoint that stands on its own, so stopping early still leaves something finished.

## Need to know

- **ATRW re-ID test set has no identity labels on LILA.** We build our own query/gallery/open-set splits from the 1,887 labeled train images (107 entities).
- **107 IDs are entities (one flank of one tiger), not tigers.** We can't link the two flanks of one tiger, so "held-out individuals" means held-out entities. Say so in the report.
- **No clip info in ATRW re-ID.** Filenames are shuffled. Pseudo-clips (near-duplicate groups within an entity) stand in for video clips when splitting.
- **Every big decision goes in `docs/decisions.md`** (decision, why, alternatives, consequences).
- **Open-set test is tiny (19 matched unknown images, 6 entities)**: AUROC CIs are ~±0.13. Only large differences count.
- **Closed-set test is small (91 images, 47 entities)**: a few points of rank-1 is noise. Run ≥3 seeds for the main models.
- **Freeze the splits once** (`splits/splits_v1.csv`). Every experiment uses the same file, or the comparisons mean nothing.
- **Open-set entities never touch training**, including for choosing hyperparameters. Choose the open-set threshold on `openset_val`, report on `openset_test`.
- **No horizontal flips** in augmentation (mirrors the stripes).
- **License CC BY-NC-SA 4.0.** Cite Li et al. 2019 (arXiv:1906.05586).
- **Compute: full training runs go on Colab** (`notebooks/colab_train.ipynb`, T4 GPU, results → Google Drive `tiger-reid-runs/`), see D14. The 8 GB M2 is too slow (~70 s/epoch) and the first full run died at epoch 3. Local `mps` is fine for tests, smoke runs and evaluation.
- Still open from Oct 6 (optional now): email a professor about wild camera-trap data.

## Phase 0: Setup ✅ mostly
- [x] Repo, docs, download script, ATRW re-ID downloaded
- [x] Exploration notebook + 20-entity grid
- [x] `tiger_reid` installable, `data.load_labeled()`
- [x] Switch to rapid mode

## Phase 1: Pseudo-clips + frozen splits ✅
- [x] Pseudo-clips: thumbnail cosine similarity + connected components; threshold 0.4 frozen (decisions D3–D4)
- [x] Splits: 20% entities open-set (half val, half test); known entities split by clip into train/val/test (D5–D6)
- [x] Leakage tests on the real split file
- **Checkpoint:** `splits/splits_v1.csv` committed, tests green.

## Phase 2: Evaluation module ✅
- [x] Rank-1/5, mAP, open-set AUROC + unit tests (`metrics.py`, D9)
- [x] `eval_sets()`, `log_result()` → `results/results.csv`; chance floor logged
- **Checkpoint:** one `evaluate()` used by every method.

## Phase 3: Baselines ✅
- [x] `ReidDataset`
- [x] Off-the-shelf ResNet50 + DINOv2 features (no training): test rank-1 42.9% / 50.5%, AUROC ≈ chance (D11)
- [x] Open-set protocol fixed: matched unknowns + bootstrap CIs (D10)
- [x] SIFT + RANSAC matcher: test rank-1 53.8%, AUROC 0.645 (only baseline above chance on open-set)
- [x] `results/summary.md` generator
- **Checkpoint:** baseline rows in `results/results.csv`.

## Phase 4: Metric learning (core)
- [x] Embedding model (timm backbone → 512-d normalized)
- [x] Training script + configs; smoke test
- [x] Colab training notebook (D14)
- [x] ResNet50 ArcFace (val rank-1 70%, mAP 0.69, AUROC 0.75; Colab T4)
- [x] DINOv2-S ArcFace, lr 3e-5 (val rank-1 76%, mAP 0.76, AUROC 0.76)
- [x] B1 DINOv2-S + erasing/crop/drop-path (val rank-1 66%, mAP 0.72, AUROC 0.79): no gain, plain config stays (D19 outcome)
- [ ] B2 `dinov2s_336` / B3 `dinov2b`, plain settings (D22)
- [ ] Runs: ResNet50 triplet; best × 3 seeds; test once; (MegaDescriptor optional)
- **Improvements under active consideration** (try after the first full runs, chosen on val):
  - **DINOv2 fine-tuned:** best untrained mAP (test 0.375); likely the strongest backbone.
  - **SIFT re-ranking:** re-score the embedding's top-k with SIFT inlier counts. SIFT is the only baseline above chance at spotting new tigers, so this mainly targets open-set AUROC.
  - **Higher resolution once trained:** stripes are fine detail; size didn't matter untrained (D11) but may after fine-tuning (e.g. 224×448 for flank crops).
  - **≥3 seeds** for the final configs (cheap on Colab).
- **Checkpoint:** closed-set table vs. baselines.

## Phase 5: Open-set + robustness
- [ ] Threshold on `openset_val`; AUROC / TPR / rank-1 on test
- [ ] Zoo→wild stress test (grayscale, blur, crop); optional augmentation retrain
- [ ] Per-pose breakdown if keypoints allow
- **Checkpoint:** core results frozen.

## Phase 6: Detection (short)
- [ ] MegaDetector on ~100 ATRW detection frames → recall at IoU 0.5

## Phase 7: Map demo
- [ ] Synthetic camera grid in a park polygon, simulated sightings, Folium per-cell activity

## Phase 8: Write-up
- [ ] 6–8 page report (with responsible use + limitations), slides. SCR optional.

## Risks

| Risk | Mitigation |
|---|---|
| Pseudo-clips too coarse/fine | Choose threshold visually, lean toward merging; report it as a limitation |
| Re-ID accuracy plateaus | Pretrained wildlife backbone (MegaDescriptor); honest ablations over chasing a number |
| Open-set threshold unstable | Choose on `openset_val`; report the full ROC |
| Zoo ≠ wild | IR/blur/crop stress test and augmentations |
| No partner data for the map | Synthetic grid is the default plan |
| Too long on plumbing | The re-ID model is the project; existing tools for everything else |
