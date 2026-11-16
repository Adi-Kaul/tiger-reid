# Tiger Re-ID

Semester project: find tigers in camera-trap images, identify each individual by its stripe pattern, and plot sightings on a park map.

**Research question:** Can a deep metric-learning model match or beat classic stripe-matching tools (ExtractCompare / SIFT-style matching) at identifying individual tigers, while also flagging tigers it has never seen before?

## Pipeline

```
camera-trap image
   │
   ▼
[1] Detection + crop      MegaDetector → tiger bounding box → crop
   │
   ▼
[2] Re-ID embedding  ★    CNN / ViT backbone trained with triplet or ArcFace loss
   │                      → a vector where same tiger = close, different tiger = far
   ▼
[3] Match or "new tiger" ★  nearest tigers in the gallery; their top-10 re-checked with
   │                      SIFT stripe matching; "new" (or human review) if nothing fits
   │
   ▼
[4] Sightings table       (tiger ID, camera, time)
   │
   ▼
[5] Map                   synthetic camera grid inside a real park boundary (Folium)
   │
   ▼
[6] Stretch: SCR          density + activity centers (R `secr`)
```

★ = the research contribution. Everything else is plumbing built from existing tools.

## Scope

| Tier | What | Must finish? |
|---|---|---|
| Core | Detection + cropping, re-ID model, open-set "new tiger" detection, evaluation vs. SIFT baseline | Yes |
| Demo | Interactive sightings map with synthetic or blurred camera locations | Yes |
| Stretch A | Spatial capture-recapture: density and activity centers | If time allows |
| Stretch B | Exploratory human–tiger conflict risk layer | If time allows |

Out of scope: movement prediction. Sparse camera detections can't support it; SCR is the defensible alternative.

## Data

- **[ATRW](https://lila.science/datasets/atrw)** (Amur Tiger Re-identification in the Wild): 92 Amur tigers with boxes, pose keypoints and identities. These are zoo images, so wild camera-trap images (infrared, blur, partial bodies) will be harder. The report should say so.
- **[LILA](https://lila.science/)**: other labeled camera-trap data for detector testing.
- **[Protected Planet / WDPA](https://www.protectedplanet.net/)**: park boundary shapefiles.

Datasets are **not** committed. They go in `data/` (gitignored). See [`data/README.md`](data/README.md).

## Repo layout

```
configs/          experiment configs (backbone, loss, hyperparameters)
data/             datasets, gitignored
docs/
  roadmap.md      the semester plan, part by part, with checklists
  learning/       concept notes written as we go
notebooks/        exploration and visual checks
results/          metrics tables and plots that go in the report
scripts/          one-off runnable scripts (download, run detector, evaluate)
src/tiger_reid/   the reusable code: datasets, models, losses, evaluation
```

## Evaluation

| What | Metric | Setup |
|---|---|---|
| Detection | Recall of tiger crops | MegaDetector vs. ground-truth boxes |
| Zoo → wild | Share of wild tigers flagged "new" | 269 wild Rajaji camera-trap images (GBIF), all unseen |
| Closed-set re-ID | Rank-1, Rank-5, mAP | Standard ATRW query/gallery split |
| Open-set re-ID | AUROC for "new tiger", false-yes rate at a fixed threshold, rank-1 on known | ~20% of individuals held out of training |
| Hard cases | Rank-1 per condition | By pose, flank side, image quality |
| Baseline | Same metrics | SIFT keypoint matching, identical splits |
| Ablations | Rank-1, mAP | CNN vs ViT, triplet vs ArcFace, flank-aware vs flank-blind |

**Split by individual and by video clip, never by frame.** Frames from one clip are near-duplicates and inflate scores.

## Responsible use

A map of where identified tigers are is useful to poachers.

- The demo uses a synthetic camera grid, or real coordinates blurred to coarse cells (e.g. 5 km).
- Public outputs show activity per grid cell, not per tiger.
- Any real partner data stays out of this repo, slides and posters.
- The conflict-risk layer, if built, is labeled exploratory and unvalidated.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .        # makes `import tiger_reid` work in notebooks and scripts
```

Training runs on a single GPU (Colab Pro, Kaggle, or the university cluster).

## Status

See [`docs/roadmap.md`](docs/roadmap.md) for the plan and [`docs/progress.md`](docs/progress.md) for the daily log. Currently finishing **Phase 4: metric learning** (rapid-development mode): model and open-set pipeline chosen on val, test run pending the threshold rule (D26).
