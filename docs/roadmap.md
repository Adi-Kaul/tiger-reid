# Roadmap

Seven parts, 1–3 weeks each. Every part ends with a checkpoint that works on its own, so stopping early still leaves something finished.

Started: Oct 6, 2026.

## Need to know

Things to keep in mind or act on soon. Updated as we go.

- **Start the ATRW download now.** The re-ID part is small (~220 MB); detection images are ~3.4 GB.
- **ATRW's re-ID test set has no identity labels on LILA.** Only the train annotations (ID ↔ filename CSV) are labeled, so we'll likely build our own query/gallery and open-set splits from the labeled train set. Confirm once downloaded.
- **License is CC BY-NC-SA 4.0** (non-commercial). Fine for coursework; cite Li et al. 2019 (arXiv:1906.05586).
- **Get GPU access early.** Colab Pro, Kaggle (free ~30 GPU-hrs/week), or the university cluster. Cluster accounts can take days to approve.
- **Email a professor this week** (wildlife ecology or CV) about advising / real camera-trap data. Replies take time.
- **Decide the splits once, in Part 0, and save them to a file.** Every later experiment (baseline, metric learning, open-set) must use the exact same splits or the comparisons mean nothing.
- **Hold out the open-set individuals from day one.** If those ~20% of tigers leak into any training run, the open-set numbers are invalid.
- **Week 10 is the freeze.** Core results are final by then; stretch work only after.

## Part 0 — Setup and foundations (week 1)

- **Learn:** PyTorch tensors, `Dataset` / `DataLoader`, a training loop ([PyTorch tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)).
- **Build:** Download ATRW, write a dataloader, display crops grouped by tiger ID.
- **Checkpoint:** A notebook showing 20 tigers with several images each.

This week:
- [x] Set up the repo
- [ ] Set up a GPU environment
- [ ] Download ATRW and confirm the re-ID split loads
- [ ] Read the ATRW paper and one recent tiger re-ID paper (e.g. a transformer-based follow-up)
- [ ] Run MegaDetector on 50 ATRW frames to see crop quality
- [ ] Email one wildlife ecology or CV professor about advising or data access

## Part 1 — CNNs and detection (weeks 2–3)

- **Learn:** Convolutional networks, transfer learning, what an object detector outputs ([CS231n notes](https://cs231n.github.io/)).
- **Build:** Run MegaDetector on ATRW frames. Fine-tune a pretrained ResNet as a plain classifier over the known tigers.
- **Checkpoint:** Detection recall + classifier test accuracy. You'll see why plain classification can't handle new tigers.

## Part 2 — Classic baseline (weeks 4–5)

- **Learn:** Keypoints, SIFT descriptors, feature matching, RANSAC ([OpenCV docs](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)).
- **Build:** A SIFT matcher that ranks gallery images for each query.
- **Checkpoint:** Rank-1 and mAP for the baseline. This is the number to beat.

## Part 3 — Metric learning, the core (weeks 6–8)

- **Learn:** Embeddings, triplet loss, ArcFace, hard-negative mining, how rank-1 and mAP are computed.
- **Build:** Re-ID embedding model with `pytorch-metric-learning`; compare CNN vs ViT backbones. Try MegaDescriptor as a starting backbone.
- **Checkpoint:** Closed-set results table vs. the Part 2 baseline.

## Part 4 — Open-set and robustness (weeks 9–10)

- **Learn:** Open-set recognition, distance thresholds, ROC / AUROC, data augmentation.
- **Build:** "New tiger" detection on held-out individuals; flank-aware matching; results by pose and quality. IR / blur augmentations for the zoo-to-wild gap.
- **Checkpoint:** Core results frozen.

## Part 5 — Mapping (weeks 11–12)

- **Learn:** Coordinate systems, shapefiles, GeoPandas, Folium.
- **Build:** Synthetic camera grid inside a real park boundary (WDPA), sightings plotted per grid cell.
- **Checkpoint:** Interactive demo map.

## Part 6 — Write-up + optional stretch (weeks 13–14)

- **Learn (stretch):** Capture-recapture and the `secr` R package.
- **Build:** 6–8 page workshop-style report (with responsible-use section), slides, optionally SCR density from simulated detections.
- **Checkpoint:** Final report and presentation.

## Risks

| Risk | Mitigation |
|---|---|
| ATRW download / format issues | Start in week 1; community mirrors exist |
| Re-ID accuracy plateaus | Pretrained wildlife backbone; honest ablations over chasing a number |
| Open-set threshold unstable | Calibrate on a validation split of held-out individuals; report full ROC |
| Zoo ≠ wild | IR/blur/crop augmentations; test on any real camera-trap tiger images |
| No partner data for the map | Synthetic grid is the default plan |
| Stretch goals eat core time | Freeze core by end of week 10 |
| Too long on plumbing | The re-ID model is the project; use existing tools for everything else |
