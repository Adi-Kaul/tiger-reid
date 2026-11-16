# Local matching: SIFT + RANSAC and DISK + LightGlue

Code: `src/tiger_reid/sift_baseline.py`, `src/tiger_reid/local_match.py`. Decisions D18, D25.

## Idea
An embedding squeezes a whole photo into one vector. **Local matching** instead finds small distinctive spots (keypoints) in two photos, pairs up spots that look alike, and checks whether the pairs agree on one geometric transform. If they do, the same stripe pattern is in both photos. This is how classic tiger tools (ExtractCompare, Hotspotter) work.

## SIFT + RANSAC (baseline, 2004)
1. **SIFT** finds up to 500 keypoints per image (max side 400 px), each with a 128-number descriptor of the local gradients.
2. **Ratio test** (Lowe, 0.75): keep a match only if the best candidate is clearly better than the second best.
3. **RANSAC homography:** repeatedly fit a transform to 4 random matches and count how many others agree within 8 px. The best count = **inliers**.

More inliers = more likely the same tiger. The score has a fixed meaning without training, which is why SIFT was the only untrained baseline above chance at spotting new tigers (test AUROC 0.64).

In the pipeline (D18) SIFT only re-checks the embedding's **top-10 tigers**. It's cheap, and it raised val AUROC from 0.81 to 0.88.

## DISK + LightGlue (D25)
- **DISK:** a learned keypoint detector and descriptor (kornia pretrained, 1024 points, max side 512 px).
- **LightGlue:** a learned matcher (transformer) that decides which keypoints correspond, using the context of all points at once.
- Same RANSAC inlier count afterwards, so it's comparable with SIFT. About 75 ms per image pair on the M2 (MPS), so only each top-10 tiger's 5 nearest gallery photos are checked.

## What we found
| | known queries, median | hardest 10% of knowns | unknowns, median |
|---|---|---|---|
| SIFT inliers | 16 | ≤6 | 6 |
| LightGlue inliers | 76 | ≤16 | 22 |

LightGlue separates *typical* known tigers from new ones much better. But the hard known cases (rear, frontal, crouched, behind a fence) get as few matches as a new tiger with either method. Their visible side simply isn't in the gallery. Combined with SIFT it reached val AUROC 0.890 (vs 0.883), below the pre-declared +0.01 bar, so it wasn't adopted.

## Gotchas
- **Background matches:** RANSAC is happy to match ground, rocks and fences. Same-zoo look-alikes get a floor of ~20 LightGlue inliers partly from that. Masking the background is a possible next step.
- **Homography assumes a flat surface.** A tiger's flank is roughly flat from the side; strong pose changes break it.
- **Near-duplicates inflate inliers** (159 SIFT inliers between two frames from one moment). That's why splits are by clip.

References: Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004; Tyszkiewicz et al., "DISK", NeurIPS 2020; Lindenberger et al., "LightGlue", ICCV 2023.
