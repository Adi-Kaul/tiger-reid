# Tiger Re-ID

Recognizing individual tigers from their stripe patterns, and knowing when a tiger is one the system has never seen before.

Every tiger's stripes are unique, like a fingerprint. Conservation teams photograph tigers with camera traps and match the stripes by hand to work out which animal is which. Those identities are how a population gets counted and how individual animals are tracked over the years. This project builds a model that does the matching automatically. It also tests the part most demos skip: saying "this is a new tiger" instead of forcing a match.

## Why I'm building this

I grew up in India and I've loved animals for as long as I can remember. India is home to about three quarters of the world's wild tigers. Counting and following them still depends on people matching stripes across huge numbers of camera-trap photos. The funding to build and maintain modern tools for that work in India largely doesn't exist. This started as a semester project, and I plan to keep chipping away at it until it's something that can make a real difference to how India's tigers are monitored.

## What it does

```
camera-trap photo
      |
      v
 1. Find and crop the tiger
      |
      v
 2. Turn the crop into a "stripe fingerprint"     (a neural network trained so that the
      |                                            same tiger lands close together)
      v
 3. Compare against every known tiger
      |
      +--> clear match, confirmed by stripe matching  ->  "this is tiger #153"
      |
      +--> nothing fits                                ->  "possibly a new tiger: check"
      |
      v
 4. Sightings table and a map of activity per grid cell (never exact locations)
```

Steps 2 and 3 are the research. The rest uses existing tools.

## Results

**Short version:** the trained model is good at recognizing tigers it already knows, but it can't yet tell when a tiger is new. Every choice was tuned on a validation set that turned out to be too small, and the test set showed it.

Final numbers are on the **test set**, run once per training run (3 runs, mean ± sd) after every choice was locked in on validation: 91 photos of 47 known tigers, and 19 photos of 6 tigers kept out of training entirely. The model learns 86 tigers from about 1,400 zoo photos.

| Approach | Right tiger is the top match | Ranking quality (mAP) | Spotting new tigers (AUROC) |
|---|---|---|---|
| Off-the-shelf ResNet50, no tiger training | 43% | 0.30 | 0.63 |
| Off-the-shelf DINOv2-S, no tiger training | 51% | 0.38 | 0.53 |
| Classic stripe matching (SIFT, no learning) | 54% | 0.31 | **0.64** |
| DINOv2 ViT-B trained on tigers + new-tiger scoring (3 runs) | **60 ± 4%** | **0.61** | 0.50 ± 0.02 |

AUROC is the chance that a new tiger looks "newer" than a known one: 0.5 is a coin flip, 1.0 is perfect.

- **What held up:** training on tigers roughly doubles ranking quality over off-the-shelf models (mAP 0.61 vs 0.30–0.38). When the system is confident enough to answer on its own, it names the right tiger 83–87% of the time. About half of known-tiger photos get answered automatically; the rest go to a person.
- **What didn't:** spotting new tigers is at chance, below plain stripe matching. A threshold set on validation to let 5% of new tigers through let **74%** through on test. In a real survey that merges two animals into one and corrupts the population count.

### Validation said otherwise

The same pipeline looked much better on validation, where every choice was made:

| | Validation (1 run) | Test (3 runs) |
|---|---|---|
| Right tiger is the top match | 82% | 60% |
| Ranking quality (mAP) | 0.82 | 0.61 |
| Spotting new tigers (AUROC) | 0.88 | 0.50 |
| New tigers accepted at the "5%" threshold | 5% | 74% |

Every new-tiger choice was made on 46 photos of only 7 held-out tigers. With so few, a pipeline can look reliable on validation and still not generalize. The test set is just as small (19 photos of 6 tigers, AUROC 95% CI 0.37–0.59), but all three runs agree. The test numbers above stay as reported. Working out why validation misled me is the next step.

**Along the way (validation only):** error analysis traced many false matches to known-tiger photos taken from angles the database doesn't cover (from behind, head-on, through a fence). A distance-based score, a modern learned matcher (LightGlue) and masking out the background all failed to help. Masking showed that part of stripe matching's signal was actually the zoo enclosure in the background, which won't carry over to the wild.

## How I approached it

- **Honest evaluation first.** The dataset's photos are video frames, and near-identical frames in both training and testing inflate scores. The video information was lost in the public release, so I rebuilt approximate "clips" from image similarity and split by tiger and by clip, never by frame.
- **Decisions written down before results.** Every choice that could shape a number goes into a [decision log](docs/decisions.md) (27 entries) with the reasoning and the alternatives. New ideas are listed with a pass/fail rule before they're run, so I can't pick the winner after the fact.
- **Failures stay in.** Higher resolution, extra regularization, distance-based scoring, the learned matcher and background masking all failed to help, and they're reported with their numbers. So is the test result.
- **Protecting the animals.** Tiger locations are valuable to poachers. No real coordinates go in this repo, and any map shows activity per 5 km grid cell, not per tiger.

## Stack

Python, PyTorch, [timm](https://github.com/huggingface/pytorch-image-models) (DINOv2 ViT-B backbone), [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) (ArcFace loss), OpenCV (SIFT + RANSAC), [kornia](https://github.com/kornia/kornia) (DISK + LightGlue), scikit-learn, pandas, pytest (61 tests). Training runs on Google Colab GPUs (T4). Evaluation runs locally on Apple Silicon (MPS).

## Repo tour

```
src/tiger_reid/   reusable code: splits, model, training, metrics, open-set scoring, matching
scripts/          entry points: make splits, train, evaluate, wild-data checks
tests/            unit tests, including leakage checks on the real split file
splits/           the frozen split file every experiment uses
results/          result tables (results.csv, summary.md, open-set grids)
docs/             decisions, progress log, roadmap, limitations, plain-language notes
notebooks/        data exploration and the Colab training notebook
```

| Read this | For |
|---|---|
| [docs/learning/00-project-story.md](docs/learning/00-project-story.md) | The whole project explained from zero |
| [docs/decisions.md](docs/decisions.md) | Every choice that shapes a result, and why |
| [docs/limitations.md](docs/limitations.md) | What these results do and don't show |
| [docs/learning/05-open-set.md](docs/learning/05-open-set.md) | The new-tiger problem in detail |
| [docs/acknowledgments.md](docs/acknowledgments.md) | Data, people, tools and references |
| [docs/progress.md](docs/progress.md) | Day-by-day log |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q
python scripts/download_atrw.py                     # dataset goes into data/ (not committed)
python scripts/eval_openset.py --stage val --dino checkpoints/dinov2b_s0.pt
```

Training: `scripts/train.py` with a config from `configs/`, or `notebooks/colab_train.ipynb` on a Colab GPU. Datasets and model weights are never committed. See [data/README.md](data/README.md).

## Status

- **Done:** leakage-safe splits, evaluation, baselines, trained models, new-tiger scoring, the wild-photo check, the one-time test run (3 training runs).
- **Next:** work out why validation and test disagree, then fix how new-tiger choices are made (more held-out tigers to choose with, fewer tuned knobs) and check it on a fresh held-out set. After that: robustness to wild conditions (night infrared, blur, partial bodies), detection, and the map.
- **Longer term:** real wild data with known identities from Indian field teams. Without it, these results remain a zoo-to-wild estimate.

Plan: [docs/roadmap.md](docs/roadmap.md).

## Data and licenses

- **ATRW** (Amur Tiger Re-identification in the Wild), CC BY-NC-SA 4.0, via [LILA BC](https://lila.science/datasets/atrw).
- **Rajaji National Park camera-trap database** (Wildlife Institute of India, via GBIF), CC BY-NC 4.0, [doi:10.15468/tc8f8r](https://doi.org/10.15468/tc8f8r).

Both are non-commercial. Full credits and citations: [docs/acknowledgments.md](docs/acknowledgments.md).

The code is MIT-licensed ([LICENSE](LICENSE)). Dataset images that appear in this repo (notebook outputs, `results/pseudo_clips/`) stay under the ATRW license, CC BY-NC-SA 4.0.
