# Open-set recognition: scores, AUROC, and the false-yes problem

Code: `src/tiger_reid/openset.py`, `scripts/eval_openset.py`, `scripts/eval_wild_novelty.py`. Decisions D10, D18, D24–D26.

## The question
Closed-set re-ID asks "which known tiger is this?". Open-set re-ID also asks "is this one of the known tigers at all?". In the field the second question matters more: a camera trap keeps photographing tigers that aren't in the catalogue yet.

The model answers with a **novelty score** per query (higher = more likely new) and a **threshold**: above it the query is called "new", below it the query is matched to its nearest known tiger.

## Novelty scores we tried (val, ViT-B seed 0)

| score | idea | val AUROC |
|---|---|---|
| nearest | distance to the single nearest gallery image | 0.772 |
| centroid | distance to the nearest tiger's mean embedding | 0.775 |
| **margin** | best tiger vs second-best tiger: a known tiger has one clear winner, a new tiger is ambiguous | 0.795 |
| margin + TTA | embeddings averaged over 5 crops of the image | 0.807 |
| **margin + TTA + SIFT re-check (chosen, D18)** | count matched stripe points against the top-10 tigers' photos; many matches = known | **0.883** |
| + nearest distance (D24) | add back "how far from everything" | 0.864 (worse) |
| + DISK + LightGlue (D25) | learned stripe matcher instead of / next to SIFT | 0.890 (not +0.01) |

Fusion: each part is **z-scored** (shifted and scaled using the val known + unknown pool) so they're on the same scale, then added with a weight. On test the val z-score parameters are reused, never refit.

## AUROC: what it does and doesn't tell you
AUROC = the chance that a random new tiger scores higher than a random known one. It needs no threshold: 0.5 = coin flip, 1.0 = perfect. That's why we chose pipelines on it.

What it hides: **where the mistakes are.** An AUROC of 0.88 can still mean that, at any threshold strict enough to keep almost every known tiger, many new tigers slip through. That's exactly what happened here.

## The false-yes problem (Nov 15–16)
- **False yes** = a new tiger accepted as a known one. In practice it merges two animals into one and corrupts population counts. A **false no** (a known tiger called new) only costs a person a check.
- At a threshold keeping 95% of known tigers recognized, the chosen pipeline accepts **48%** of val unknowns and **49%** of the 269 wild Rajaji tigers as known.
- **Why:** the threshold is set by the hardest few known queries. These are rear, frontal, crouched or behind-a-fence shots whose visible side isn't in the gallery. With nothing to match, they look as new as a new tiger. LightGlue gets a median 76 inliers on known queries vs 22 on unknowns, but the hardest 10% of knowns get ≤16. No matcher can line up stripes that aren't visible.
- The new tigers that get accepted are mostly **other tigers from the same zoo** (same ground, fences, light).

## Operating points: turning the question around
Instead of "keep 95% of knowns", fix the **false-yes rate** you can live with and see how many known tigers still get recognized automatically. The rest go to human review.

| false-yes target (val) | knowns auto-recognized |
|---|---|
| 5% | 69% |
| 10% | 79% |
| 20% | 82% |

Choosing that target is D26 (on hold, user decision). Small numbers matter here: 5% of 46 unknowns is about 2 images.

## Gotchas
- **Few unknowns:** val has 46 unknown images of only 7 tigers; test has 19 of 6. Any false-yes rate is a few images. Report counts next to percentages.
- **Don't tune on what you report.** Pipelines and thresholds come from val (`openset_val`). Test is run once. Every extra candidate we add after seeing val results (D24, D25) is a second look and is labeled as such.
- **Domain shift can fake success.** Plain nearest distance flags wild images well (AUROC 0.95) only because night/IR camera-trap shots look unlike any zoo photo. It is worse on unseen zoo tigers (0.78).
- **Label noise:** openset_val entity 250 looks like known entity 247 (same scene, same moment). Kept in, flagged.
