# Roadmap

Rapid-development mode (since Nov 10, 2026): no due dates, aim for results fast, and test, commit and push after every step. Each phase ends with a checkpoint that stands on its own, so stopping early still leaves something finished.

## Need to know

- **ATRW re-ID test set has no identity labels on LILA.** We build our own query/gallery/open-set splits from the 1,887 labeled train images (107 entities).
- **107 IDs are entities (one flank of one tiger), not tigers.** We can't link the two flanks of one tiger, so "held-out individuals" means held-out entities. Say so in the report.
- **No clip info in ATRW re-ID.** Filenames are shuffled. Pseudo-clips (near-duplicate groups within an entity) stand in for video clips when splitting.
- **Every big decision goes in `docs/decisions.md`** (decision, why, alternatives, consequences).
- **Closed-set test is small (91 images, 47 entities)**: a few points of rank-1 is noise. Run ≥3 seeds for the main models.
- **Freeze the splits once** (`splits/splits_v1.csv`). Every experiment uses the same file, or the comparisons mean nothing.
- **Open-set entities never touch training**, including for choosing hyperparameters. Choose the open-set threshold on `openset_val`, report on `openset_test`.
- **No horizontal flips** in augmentation (mirrors the stripes).
- **License CC BY-NC-SA 4.0.** Cite Li et al. 2019 (arXiv:1906.05586).
- Compute: local Mac `mps` is enough for this dataset size; Colab if training gets slow.
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

## Phase 2: Evaluation module
- [ ] Rank-1/5, mAP, open-set AUROC + unit tests
- **Checkpoint:** one `evaluate()` used by every method.

## Phase 3: Baselines
- [ ] `ReidDataset`
- [ ] Off-the-shelf ImageNet ResNet50 features (no training)
- [ ] SIFT + RANSAC matcher
- **Checkpoint:** baseline rows in `results/results.csv`.

## Phase 4: Metric learning (core)
- [ ] Embedding model (timm backbone → 512-d normalized)
- [ ] Training script + configs; smoke test
- [ ] Runs: ResNet50 ArcFace, ResNet50 triplet, ViT/DINOv2 ArcFace, MegaDescriptor
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
