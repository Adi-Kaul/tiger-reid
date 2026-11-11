# Re-ID metrics: rank-k, mAP, AUROC

Code: `src/tiger_reid/metrics.py`. Tests with these exact examples: `tests/test_metrics.py`.

Everything starts from a **distance matrix**: one row per query, one column per gallery image, lower = more similar. Embedding models use cosine distance. SIFT can pass its own distances (e.g. minus the inlier count). Every method gets scored by the same code.

## Rank-k
Sort the gallery by distance for each query. **Rank-k** = fraction of queries with the right tiger somewhere in the top k.

## mAP (mean average precision)
Rank-1 only looks at the first hit. **AP** for one query looks at *every* correct gallery image: at each rank where a correct one appears, compute precision (correct so far / items seen so far), then average those. mAP = mean AP over queries.

Worked example (gallery = A, A, B):

| query | ranking | hits at ranks | AP |
|---|---|---|---|
| A | A ✓, B, A ✓ | 1, 3 | (1/1 + 2/3) / 2 = 0.833 |
| B | A, A, B ✓ | 3 | 1/3 = 0.333 |

Rank-1 = 0.5, mAP = 0.583. mAP is stricter: query B got rank-3 right but is punished for the two wrong images above it.

## Open-set AUROC
For each query, take its distance to the **nearest** gallery image. Known tigers should be close; new tigers should be far.
**AUROC** = the chance that a random new-tiger query is farther than a random known-tiger query. 1.0 = perfect separation, 0.5 = coin flip. It doesn't need a threshold; choosing the actual threshold is Phase 5.

Example: known nearest distances 0.1, 0.2; unknown 0.3, 0.05 → of the 4 (unknown, known) pairs, 2 have the unknown farther → AUROC 0.5.

## Our protocol (decisions D9)
- **Gallery** = train images of the 86 known entities.
- **Query** = `test` images (91); while choosing settings, `val` (67).
- **Unknown** = `openset_test` (129); while choosing settings, `openset_val` (221).

## Gotchas
- **Small test set = noisy numbers.** Random embeddings give AUROC anywhere from 0.465 to 0.528 across 3 seeds. A difference of a couple of points between two models means nothing. Report the mean ± spread over seeds.
- Chance rank-1 here ≈ 1.1% (each entity has ~16 of the 1,379 gallery images).
- Never choose settings on `test`/`openset_test`; that's what `val` is for.
- Ties: sorted stably (gallery order), so results are deterministic.

## Links
- Market-1501 paper (where the mAP protocol for re-ID comes from): Zheng et al., 2015
- ROC/AUROC intuition: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
