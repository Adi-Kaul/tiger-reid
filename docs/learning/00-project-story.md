# The project, explained from zero

Plain-language walkthrough of what we built and why (written Nov 11, 2026, end of Phase 3). Useful for the report intro and the presentation. Details and reasoning for each choice: [`../decisions.md`](../decisions.md).

## 1. The problem
Conservationists count wild tigers with **camera traps** (motion-triggered cameras). To estimate how many tigers live in a park, you need to know *which* tiger is in each photo, so nobody gets counted twice. Each tiger's stripes are unique, like a fingerprint, but matching them by eye is slow and error-prone.

**Goal:** given a tiger photo, say either "this is tiger #42, seen before" or "this is a tiger we've never seen".

## 2. What "machine learning" means here
Writing explicit rules ("a stripe shaped like X at position Y…") is hopeless. Instead you show the computer many examples with the right answers and it adjusts itself. The thing being adjusted is a **neural network**: a big function with millions of internal knobs (**weights**). A photo goes in, numbers come out. **Training** = turning the knobs a little at a time so the output gets better.

## 3. The data
**ATRW**: Amur tiger photos from zoo videos; we use the 1,887 labeled ones. The 107 labels are **entities, not tigers**: a tiger's left and right sides have different stripes, so each side is its own identity. Wild cameras usually see one side anyway.

## 4. Why not a "which tiger is this?" classifier
A classifier has a fixed answer list (tiger 1…86). It can't say **"none of these"**, and adding a tiger means retraining.

So we use **embeddings**: the network turns each photo into a list of numbers (e.g. 512), which you can think of as a **point on a map**. Same tiger → close; different tigers → far.
- **Identify:** nearest known photo on the map → its tiger.
- **New tiger:** even the nearest known photo is far away → probably never seen.
Adding a tiger = put its photos on the map. No retraining.

## 5. How we measure success
It works like a police lineup. The **gallery** is the database of known photos (training images: 1,379 photos, 86 entities). A **query** is a photo to identify.

| Score | Question |
|---|---|
| **Rank-1** | How often is the single closest photo the right tiger? |
| **mAP** | Are *all* of that tiger's photos ranked near the top? (stricter) |
| **AUROC** | How often does a random new tiger look "farther" from everything than a random known one? 0.5 = coin flip, 1.0 = perfect |

More: [`02-metrics.md`](02-metrics.md).

## 6. The big trap: cheating without meaning to (leakage)
The photos are **video frames**. Two frames a split second apart are near-identical (same background, pose, light). If one is in the gallery and the other is a query, the model "recognizes" the *scene*, not the stripes, and the score looks great while meaning nothing. Real camera-trap photos are never near-copies of your database.

**Fix:** never let one video sit on both sides of the split. The publishers deleted video info, so we **reconstructed it** ("pseudo-clips"):
1. Shrink each photo to 32×32. Stripes vanish; the scene remains.
2. Measure how similar each pair of thumbnails is.
3. Same entity + similarity > **0.4** → same pseudo-clip.

**Why 0.4 (D4):** too high chops real videos into pieces that land on both sides, which is hidden cheating (the dangerous error, because numbers look *good*). Too low glues videos together, so we lose test photos but nothing cheats (the safe error). We chose safe. More: [`01-pseudo-clips.md`](01-pseudo-clips.md).

**Frozen split** (`splits/splits_v1.csv`):

| Group | Photos | Purpose |
|---|---|---|
| train | 1,379 | training + gallery |
| val | 67 | choosing settings |
| test | 91 | final known-tiger scores |
| openset_val | 221 | choosing the "new tiger" cutoff |
| openset_test | 129 | final new-tiger scores |

**Open-set** entities (21) are never shown in training, or "detecting new tigers" would be fake. **Val vs. test:** if you tweak settings until the test score rises, you've tuned to the test and the number stops meaning anything. Choices are made on val; test is only for final numbers.

## 7. Two problems we caught
1. **Wrong clip in test.** The first rule sometimes sent a tiger's *biggest* clip to test, starving training. A unit test caught it → "biggest clip always stays in training".
2. **Unfair open-set comparison.** Known-tiger test photos were deliberately odd shots (least-similar clip), while new-tiger photos were typical. Known tigers looked *more* unfamiliar, and AUROC came out 0.37 (worse than chance). Fixed by comparing like with like: new-tiger photos are also taken from outside their main clip. That leaves only **19** new-tiger test photos, so every number gets a **95% confidence interval**. Overlapping ranges = no real difference.

## 8. Baselines (no tiger training)

| Method | What it is | Rank-1 | mAP | New-tiger AUROC |
|---|---|---|---|---|
| Chance | random | 1% | 0.015 | 0.5 |
| ResNet50 | network trained on everyday photos (ImageNet), as-is | 43% | 0.30 | 0.63 (range includes 0.5) |
| DINOv2 | self-supervised network, as-is | 50% | **0.38** | 0.53 (≈ chance) |
| SIFT + RANSAC | classic 2004 method: match distinctive stripe points geometrically | **54%** | 0.31 | **0.65 (clearly above chance)** |

- Untrained networks get about half right: they already capture shape, pose and coat colour.
- Only SIFT spots new tigers. Its score is a count of matched stripe points, so a new tiger scores low with *everyone*. Untrained network distances have no fixed meaning, because nobody taught them what "same tiger" is. Phase 4 teaches exactly that.

## 9. Phase 4 plan: teaching the network "same tiger"
**Training loop:** show a batch → compute a **loss** (one number for "how wrong is the map") → calculus tells which way to nudge every knob (**gradient descent**) → repeat. One pass over the data = an **epoch** (~30 of them).

| Choice | Why |
|---|---|
| Start from pretrained ResNet50 / DINOv2 (**fine-tuning**) | 1,379 photos is far too few to learn vision from scratch; adjust a network that already knows edges and textures |
| **ArcFace** loss | Trains as a classifier over the 86 training tigers, but forces tight clusters with gaps; afterwards the classifier part is thrown away, only the map is kept. Usually best on small data |
| **Triplet** loss | Anchor / same-tiger / different-tiger: same must be closer by a margin. Intuitive, slower. Compared against ArcFace (ablation) |
| Batches of 4 photos × several tigers | Losses need same-tiger pairs in each batch; stops tiger 153 (98 photos) dominating |
| Augmentation (crops, rotation, colour), **no flips** | Teaches the network to ignore irrelevant changes; a mirror flip makes a stripe pattern no real tiger has (D8) |
| **Early stopping** on val mAP | Small data → **overfitting** (memorizing photos/backgrounds). Keep the best epoch |
| 3 **seeds** for the best model | Seeds fix the randomness; 3 runs show real effect vs. luck |

**Run order:** smoke test → ResNet50 + ArcFace → ResNet50 + triplet → DINOv2 + ArcFace → MegaDescriptor → best × 3 seeds.

**Caution, MegaDescriptor:** pretrained on animal re-ID datasets that very likely **include ATRW**, so it may have seen our test photos. Check its training sources first. If ATRW is included, skip it or report it separately with a warning.

**Expectations:** biggest gain should be new-tiger AUROC (that's what the loss teaches), and rank-1/mAP should rise too. If training doesn't clearly beat SIFT given the wide intervals, that's a legitimate finding, reported honestly.

## 10. What Phase 4 found on val (Nov 11–16, before the test run)
- **Training works.** DINOv2 ViT-B + ArcFace: val rank-1 **81%**, mAP **0.78 ± 0.04** (4 runs). The untrained baselines got ~50% rank-1 on test. Bigger images, extra regularization and longer training didn't help; the bigger backbone did.
- **Spotting new tigers:** the best "is this new?" score is the **margin** between the two closest known tigers, plus a **SIFT stripe re-check** of the top-10 candidates. Val AUROC went from 0.77 to **0.88**. Details: [05-open-set.md](05-open-set.md).
- **But AUROC hid a problem.** If the threshold keeps 95% of known tigers recognized, about **half of new tigers get accepted as known**. Those are false yes's, the costly mistake. That holds for unseen zoo tigers and for 269 wild camera-trap tigers from Rajaji (India).
- **Why:** a few known-tiger photos show a side or view that isn't in the gallery (rear, frontal, behind a fence). Nothing can match stripes that aren't visible, so these look "new" and drag the threshold up. A much better learned matcher (LightGlue) didn't fix it: [06-local-matching.md](06-local-matching.md).
- **What's left to decide (D26):** pick the threshold by an acceptable false-yes rate (e.g. 5%: about 69% of known tigers recognized automatically, the rest checked by a person). Then run the test set once.


## 11. The test run (Nov 20): validation misled us
Thresholds were set per D26 (5% and 10% false-yes targets on val), then the test set was run **once per training run** (seeds 0, 1, 2). Full table: [`results/summary.md`](../../results/summary.md).

| | val (seed 0) | test (3 runs) |
|---|---|---|
| rank-1 | 82% | 60 ± 4% |
| mAP | 0.82 | 0.61 ± 0.01 |
| new-tiger AUROC | 0.88 | 0.50 ± 0.02 (chance) |
| new tigers accepted at the "5%" threshold | 5% | 74% |

- **Recognizing known tigers held up:** mAP 0.61 vs 0.30–0.38 for off-the-shelf models, and 83–87% right when the system answers on its own.
- **Spotting new tigers did not:** at chance on test, below SIFT alone (0.64). The val thresholds don't transfer.
- **Likely cause:** every new-tiger choice (D18–D27) was made on 46 photos of 7 held-out tigers. That's too few to choose from. Test has 19 photos of 6 tigers, so its estimate is noisy too, but all three runs agree.
- **What this means:** the test numbers are the reported result. Any fix needs a different way of choosing (more held-out tigers, fewer knobs) and a fresh held-out set, since this test set has now been seen.
