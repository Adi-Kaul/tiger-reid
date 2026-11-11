# Embeddings, metric learning and triplet loss

Code: `src/tiger_reid/model.py`, `src/tiger_reid/training.py`. Decisions D12–D13.

## The idea
A **classifier** answers "which of these 86 known tigers is it?" and has no way to say "none of them". A **metric-learning** model instead maps every image to a point (an *embedding*, here a 512-d unit vector) so that photos of the same tiger land close together and different tigers land far apart. Identifying a tiger = find the nearest gallery embedding. Saying "new tiger" = the nearest one is still far away. That's why re-ID uses embeddings: the model works for tigers it never saw in training.

## Our model
`timm backbone (ImageNet / DINOv2 pretrained) → pooled features → Linear → BatchNorm → L2-normalize`. Normalizing puts every embedding on a sphere, so cosine distance is the only thing that matters, the same distance used for the untrained baselines.

## Triplet loss
Take an **anchor**, a **positive** (same tiger) and a **negative** (different tiger). Loss = `max(0, d(a,p) − d(a,n) + margin)`: push the negative at least `margin` further away than the positive. Most random triplets are already "easy" (loss 0) and teach nothing, so a **miner** picks informative ones. We use *semi-hard* triplets: the negative is further than the positive but still inside the margin.

## Why batches of m images per tiger
A triplet needs a positive in the same batch. With 86 entities and random batches of 32, most images would have no partner. `MPerClassSampler(m=4)` builds each batch from 8 entities × 4 images, so every image has 3 positives and 28 negatives. It also stops the one entity with 98 images from dominating.

## Gotchas
- Near-duplicate frames make training positives too easy. That's one reason the splits are by pseudo-clip (`01-pseudo-clips.md`).
- Training loss falling ≠ generalizing: we select the epoch on **val mAP**, not loss.
