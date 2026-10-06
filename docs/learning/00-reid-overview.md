# Re-ID in one page

The mental model for the whole project. Read this before Part 0.

## Classification vs. re-identification

- **Classification:** "Which of these 92 tigers is this?" The model has one output per tiger. Add a 93rd tiger and you have to retrain. It also has no way to say "none of these."
- **Re-identification:** "Which images show the *same* tiger?" The model maps each image to a vector (an **embedding**). Same tiger → vectors close together; different tigers → far apart. A new tiger just produces a vector that isn't close to anything, so no retraining needed.

Part 1 trains a classifier on purpose so you see this limitation firsthand.

## Query and gallery

- **Gallery:** images of tigers we already know, with IDs.
- **Query:** a new image. Embed it, find the nearest gallery embeddings, and the closest one's ID is the prediction.
- **Rank-1:** fraction of queries whose single nearest gallery match is the right tiger.
- **Rank-5:** right tiger is somewhere in the top 5.
- **mAP:** rewards ranking *all* correct gallery matches high, not just the first one. Stricter than rank-1.

## Closed-set vs. open-set

- **Closed-set:** every query tiger is in the gallery. Standard benchmark.
- **Open-set:** some queries are tigers never seen before. We pick a distance threshold: farther than it → "new tiger." Measured with **AUROC** (how well distance separates known from unknown, across all thresholds).

## Why splits matter so much

ATRW comes from video. Two frames a fraction of a second apart are nearly identical. If one is in training and the other in test, the model "recognizes" it by memory, not by stripes, and scores look great for the wrong reason. So we split **by video clip and by individual**, never by frame.

## Tiger-specific gotchas

- **Flanks:** a tiger's left and right stripe patterns are different. A left-side photo can't be matched to a right-side photo of the same tiger. "Flank-aware" matching only compares same-side images.
- **Zoo vs. wild:** ATRW is zoo footage: good lighting, color, full bodies. Real camera traps are often night-time infrared (grayscale), blurry and partial.

## Links

- ATRW paper: Li et al., 2019, "ATRW: A Benchmark for Amur Tiger Re-identification in the Wild" (arXiv:1906.05586)
- ATRW download: https://lila.science/datasets/atrw
