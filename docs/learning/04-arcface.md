# ArcFace

Code: `pytorch_metric_learning.losses.ArcFaceLoss` via `training.build_loss()`. Decision D12.

## What it does
ArcFace is a classifier in disguise. It keeps one learnable unit vector **W_c** per training entity (a "class centre"). For an embedding **z** (also unit length), the logit for class c is the cosine of the angle θ_c between z and W_c. ArcFace adds an **angular margin** m to the correct class: the logit becomes cos(θ_y + m), and all logits are multiplied by a scale s before softmax.

Effect: to get a low loss, an image must sit not just closer to its own centre than to others, but closer **by at least m radians**. Classes become tight clusters with gaps between them, which is exactly what nearest-neighbour matching and an open-set distance threshold need.

## Why it's used for re-ID if it's a classifier
After training we **throw the class centres away** and keep only the embedding. New tigers never had a centre and don't need one: they get an embedding and are compared by cosine distance like everything else.

## Our settings
margin 28.6° (0.5 rad), scale 64 (pml defaults, also the paper's). The class-centre weights get a higher learning rate (1e-2) than the backbone (1e-4), because they start random.

## Compared with triplet
ArcFace looks at every class centre at every step (global view), while triplet only sees pairs inside a batch. With few images per class it usually trains faster and more stably. We run both to check that on our data (D12).

Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.
