# Limitations

What the results in this repo do and don't show. These go into the final report as well. Decision numbers (D1, D2, ...) point to [decisions.md](decisions.md).

## The data

**Zoo tigers, not wild ones.** ATRW was filmed in zoos: daylight, clear views, mostly side-on walking shots. Real camera traps take night infrared photos, blurred motion and partial bodies. Every accuracy number here is a zoo number. The 269 wild Rajaji photos only test one thing: whether wild tigers get flagged as new.

**Amur tigers, not Bengal tigers.** ATRW is the Amur (Siberian) subspecies. The Indian tigers this project is ultimately for are Bengal tigers, with different coats and habitats. How well the model transfers is unmeasured.

**IDs are one side of one tiger.** Left and right stripe patterns differ, so ATRW labels each flank separately, and the public release doesn't say which two flanks belong to the same animal. So "107 identities" means 107 flanks. A held-out "new" flank may belong to a tiger whose other side is in training (D1). That doesn't help stripe matching, but the report has to say so.

**Our own splits, not the official benchmark.** The public ATRW test set has no identity labels, so all splits come from the 1,887 labeled training photos. The numbers can't be compared directly with published ATRW leaderboards (D2).

**Reconstructed clips.** The photos are video frames, and the release lost the video information. Clips were rebuilt from image similarity (D3, D4). Some near-duplicate frames may still sit on both sides of a split, which would inflate scores slightly.

**At least one label error.** One "new" validation tiger (entity 250) looks like the same animal as a known tiger (247), photographed at the same moment. It was kept in rather than removed after seeing results (D25).

## Small evaluation sets

| Set | Photos | Tigers (flanks) |
|---|---|---|
| Validation, known tigers | 67 | 27 |
| Validation, new tigers (matched) | 46 | 7 |
| Test, known tigers | 91 | 47 |
| Test, new tigers (matched) | 19 | 6 |

One photo moves rank-1 by 1–2 points. Confidence intervals are wide (test rank-1 about ±10 points, new-tiger AUROC about ±0.1), so only large differences count. The test photos also lean harder than average, because each known tiger's least typical clip went to test (D5).

**Training runs are noisy.** The same configuration with the same seed on a Colab GPU gave validation mAP 0.77 and 0.73. Model comparisons use the mean of 3–4 runs, and even then ViT-B's lead over ViT-S isn't statistically significant (D22).

## The method

**The new-tiger problem isn't solved.** At a threshold that keeps 95% of known tigers recognized, about half of new tigers are accepted as known. The cause is known-tiger photos whose visible side isn't in the database: rear, head-on, crouched, behind a fence. No matcher can line those stripes up (D25). The proposed fix is a human-review step, not a better model (D26).

**Several looks at the same validation set.** The new-tiger scoring was chosen on validation, then revisited twice (D24, D25) after seeing results. Each round was declared in advance with a pass/fail rule, and the final choice didn't change. Still, validation numbers are optimistic. The test set is the honest number, and it's run once.

**Background can leak in.** Crops include the enclosure. Tigers from the same zoo share ground, fences and lighting, and both the network and the stripe matchers partly match on that.

**The detection step isn't built yet.** Results use ATRW's hand-drawn boxes. Wild photos were cropped with an off-the-shelf COCO detector, and 45 of 269 had no confident box. That crop script isn't in the repo yet, so this step can't be reproduced from the code.

## The wild data

- **No identities.** The Rajaji images on GBIF have no individual IDs, so they can't test matching, only "is this flagged as new?". Hand-labeling them (D23) is parked.
- **Repeat animals.** Several of the 269 photos show the same tiger, so they're fewer than 269 independent tests, and the confidence intervals on wild numbers are too narrow.
- **Domain shift flatters some scores.** A simple distance score flags wild photos well (AUROC 0.95) mostly because night and infrared shots look unlike any zoo photo, not because it recognizes a different tiger. On unseen zoo tigers it's worse (0.78).

## What this doesn't claim

- That it's ready for field use. It's a research prototype trained on zoo footage.
- That it replaces expert identification. The intended use is suggesting likely matches and flagging uncertain ones for a person.
- Anything about population size, density or movement. Those need real survey data and methods such as spatial capture-recapture.
