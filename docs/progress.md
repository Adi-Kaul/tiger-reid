# Progress log

One entry per working day, newest first. Each entry: what got done, findings, problems and how they were solved, what's left for me, and what's next.

---

## Nov 10, 2026 (Tue) — Part 0 → Phase 1

**Mode switch:** due dates dropped; moving to **rapid development**: Claude writes the code, explains briefly, tests + commits + pushes after every step, and checks in at phase boundaries. `CLAUDE.md` and `docs/roadmap.md` rewritten (Phases 0–8, no dates; Part 1 classifier dropped). The TODO skeleton below gets filled in by Claude.

**Done**
- Switched to rapid mode (CLAUDE.md, roadmap, README status).
- **Pseudo-clips implemented** (`pseudo_clips.py`: 32×32 mean-centered thumbnails → cosine → connected components within each entity; `assign_pseudo_clips()`), 5 pytest tests. `scripts/make_pseudo_clips.py` sweeps thresholds → `results/pseudo_clips/sweep.csv` + grids.
- Finding: **raw-pixel cosine is useless** for this: within-entity median 0.885 vs between-entity 0.873. **Mean-centered** separates: 0.38 vs 0.09 (99th pct between = 0.52).
- Sweep: t=0.4 → 260 clips, 47 entities with 1 clip; t=0.5 → 439 clips, 28 with 1 clip; t≥0.7 → mostly singleton clips (too fine, leaks).
- Visual check: t=0.5 still splits same-scene frames (entity 27); t=0.4 is better but not perfect, and it chains entity 153's 98 images (day + dark) into one clip, which is the safe direction. Leaning t=0.4; waiting on confirmation.
- Made `tiger_reid` importable: added `pyproject.toml` and ran `pip install -e .` in `.venv`. Notebooks can now `from tiger_reid... import ...`.
- `src/tiger_reid/data.py`: `load_labeled()` returns the 1,887 labeled images (`entity_id`, `filename`, `path`). Checked: 1,887 rows, 107 entities.
- `src/tiger_reid/pseudo_clips.py`: skeleton for pseudo-clips. **Three TODOs are mine**: `image_features`, `cosine_similarity_matrix`, `group_by_threshold`. `show_clips` (plotting) is done.
- New learning note: [`01-pseudo-clips.md`](learning/01-pseudo-clips.md) (which features detect "same video", chaining, how to pick a threshold without labels).

**Findings**
- **Schedule slip:** 5 weeks since day 1, and Part 0 is still open. The plan has Part 3 starting now; the core freeze is ~Dec 13. Logged in roadmap "Need to know".
- Crops vary a lot in size (≈150–900 px per side), so features must resize to a fixed size before comparing.

**Problems → fixes**
- `import tiger_reid` failed (package not installed) → `pyproject.toml` + editable install.

**My to-dos**
- [ ] **Decide the re-plan** (see Next)
- [ ] Before coding: predict how many pseudo-clips a typical entity (~13 images) will have, and what a good threshold looks like
- [ ] Fill in the 3 TODOs in `pseudo_clips.py`, run them on 3–4 entities in a new notebook `01_pseudo_clips.ipynb`, look at `show_clips` grids
- [ ] Carry-overs from Oct 6: tidy notebook 00 Findings cell; read ATRW paper Sec. 3; email a professor; GPU access

**Next**
- Choose and freeze the threshold → pseudo-clips for all 107 entities → save to `data/splits/` (and a small summary table in `results/`).
- Then design the splits: open-set holdout (~20% of entities), and query/gallery by pseudo-clip.
- Then the `Dataset` skeleton (mine to fill) → Part 0 closed.

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
