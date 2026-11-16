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

## Results so far

All numbers below are on the **validation set**: 67 photos of known tigers and 46 photos of tigers held out of training entirely. The final test set is deliberately still untouched. It gets run once, after every choice is locked in, so the reported numbers can't have been tuned on it.

| Approach | Right tiger is the top match | Ranking quality (mAP) | Spotting new tigers (AUROC) |
|---|---|---|---|
| Classic stripe matching (SIFT, no learning) | 45% | 0.24 | 0.83 |
| Off-the-shelf image model, no tiger training | 58% | 0.31 | 0.52 |
| ResNet50 trained on tigers | 70% | 0.69 | 0.75 |
| DINOv2 ViT-B trained on tigers (mean of 4 runs) | **81%** | **0.78** | 0.80 |
| ViT-B + new-tiger scoring with a stripe-matching re-check (1 run) | 82% | 0.82 | **0.88** |

AUROC is the chance that a new tiger looks "newer" than a known one: 0.5 is a coin flip, 1.0 is perfect. The model learns 86 tigers from about 1,400 zoo photos.

**The finding I care most about.** A high AUROC hid a real problem. If the system is tuned to recognize 95% of known tigers, about half of new tigers get wrongly accepted as known ones. That held for unseen zoo tigers and for 269 wild camera-trap photos from Rajaji National Park in India. That mistake merges two animals into one and corrupts a population count.

I traced it to a handful of known-tiger photos taken from angles the database doesn't cover: from behind, head-on, or through a fence. No matching method can line up stripes that aren't visible, so those photos look "new". Keeping them recognized forces a lenient threshold, and new tigers slip through. I tried two fixes, a distance-based score and a modern learned matcher (LightGlue). Both are reported as not helping. The practical answer is to set the threshold by an acceptable rate of false matches and send uncertain photos to a person. At a 5% false-match rate, about 69% of known tigers are recognized automatically.

## How I approached it

- **Honest evaluation first.** The dataset's photos are video frames, and near-identical frames in both training and testing inflate scores. The video information was lost in the public release, so I rebuilt approximate "clips" from image similarity and split by tiger and by clip, never by frame.
- **Decisions written down before results.** Every choice that could shape a number goes into a [decision log](docs/decisions.md) (26 entries) with the reasoning and the alternatives. New ideas are listed with a pass/fail rule before they're run, so I can't pick the winner after the fact.
- **Failures stay in.** Higher resolution, extra regularization, distance-based scoring and the learned matcher all failed to help, and they're reported with their numbers.
- **Protecting the animals.** Tiger locations are valuable to poachers. No real coordinates go in this repo, and any map shows activity per 5 km grid cell, not per tiger.

## Stack

Python, PyTorch, [timm](https://github.com/huggingface/pytorch-image-models) (DINOv2 ViT-B backbone), [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) (ArcFace loss), OpenCV (SIFT + RANSAC), [kornia](https://github.com/kornia/kornia) (DISK + LightGlue), scikit-learn, pandas, pytest (55 tests). Training runs on Google Colab GPUs (T4). Evaluation runs locally on Apple Silicon (MPS).

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

- **Done:** leakage-safe splits, evaluation, baselines, trained models, new-tiger scoring, the wild-photo check.
- **Next:** pick the threshold rule, run the test set once across 3 training runs, then robustness to wild conditions (night infrared, blur, partial bodies), detection, and the map.
- **Longer term:** real wild data with known identities from Indian field teams. Without it, these results remain a zoo-to-wild estimate.

Plan: [docs/roadmap.md](docs/roadmap.md).

## Data and licenses

- **ATRW** (Amur Tiger Re-identification in the Wild), CC BY-NC-SA 4.0, via [LILA BC](https://lila.science/datasets/atrw).
- **Rajaji National Park camera-trap database** (Wildlife Institute of India, via GBIF), CC BY-NC 4.0, [doi:10.15468/tc8f8r](https://doi.org/10.15468/tc8f8r).

Both are non-commercial. Full credits and citations: [docs/acknowledgments.md](docs/acknowledgments.md).
