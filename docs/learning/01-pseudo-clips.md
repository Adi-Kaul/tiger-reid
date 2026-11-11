# Near-duplicates and pseudo-clips

## What it is

ATRW images are frames taken from zoo videos. The LILA release renamed the files to plain numbers, so the "which video did this come from" information is gone. A **pseudo-clip** is our reconstruction of it: inside one entity, group the images that look like they came from the same video.

How: turn each image into a vector, measure how similar every pair is, and link pairs above a threshold. Each linked group is one pseudo-clip.

## Why it matters here

If two frames from one video end up on different sides of the split (one in train, one in test), the model can match them by background, lighting and pose instead of stripes. Rank-1 looks great and means nothing. That's **leakage**. Splitting by pseudo-clip keeps every group on one side.

## Which features find "same video"?

We want *same scene*, not *same tiger*. Two photos of one tiger in different enclosures should land in **different** clips.

| Feature | Captures | Good for clips? |
|---|---|---|
| Tiny thumbnail (e.g. 32×32 pixels) | Layout, background colour, lighting, pose | Yes, and it's a good first try |
| Perceptual hash (pHash) | Same as above, compressed to 64 bits | Yes, the classic near-duplicate tool |
| Pretrained CNN embedding | Content ("a tiger lying down") | Partly. May group same-pose images from *different* videos |
| Re-ID embedding (later) | Stripes / identity | No, it's meant to say "same tiger", which is the wrong question |

## Gotchas

- **Chaining.** Connected components links A–B–C even if A and C look different. Good for a tiger walking across the frame. Bad if a too-low threshold makes everything one giant clip.
- **No ground truth.** There are no labels for clips. Choose the threshold by **looking**: print clip sizes, show grids, check a few entities by eye. Write down the threshold and why you chose it.
- **Choose it once, freeze it.** Choose the threshold before any model results exist. Never change it later to make the numbers look better.
- **Err toward merging.** If unsure, one too-big clip only makes the split a bit coarser. One video split in two leaks.
- **Entities with one clip.** If an entity's images all come from one video, it can't appear in both gallery and query without leaking. Those entities can still go fully into train, or fully into the open-set holdout.

## Links

- Cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- Connected components: https://en.wikipedia.org/wiki/Component_(graph_theory)
- Perceptual hashing (`imagehash` library): https://github.com/JohannesBuchner/imagehash
