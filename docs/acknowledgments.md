# Acknowledgments

## Data

**ATRW: Amur Tiger Re-identification in the Wild.** The core dataset: 1,887 labeled photos of Amur tigers. Collected by Li et al. and distributed by [LILA BC](https://lila.science/datasets/atrw) under CC BY-NC-SA 4.0.

**Camera trap database of Tiger from Rajaji National Park, Uttarakhand.** Harihar A, Pandav B, Hussein I (2019), Wildlife Institute of India, version 2.5, via GBIF, [doi:10.15468/tc8f8r](https://doi.org/10.15468/tc8f8r), CC BY-NC 4.0. Used only as a check that wild tigers get flagged as new. Images and location data are kept out of this repo.

## People

**Abhishek Harihar** (Panthera) replied to my request for wild tiger data and pointed me to the Rajaji dataset.

## Models and tools

- **DINOv2** (Meta AI): the pretrained vision backbone, via timm.
- **timm** (Ross Wightman and contributors): backbones.
- **pytorch-metric-learning** (Kevin Musgrave and contributors): ArcFace loss and samplers.
- **kornia**: DISK features and the LightGlue matcher.
- **OpenCV**: SIFT and RANSAC.
- **PyTorch, scikit-learn, pandas, NumPy, matplotlib, pytest.**
- **Google Colab**: GPU time for training.
- **GBIF API**: wild image retrieval.

This project was built with help from Claude (Anthropic) as a coding assistant. Design choices, decisions and their reasoning are logged in [decisions.md](decisions.md).

## References

1. Li S, Li J, Tang H, Qian R, Lin W. *ATRW: A Benchmark for Amur Tiger Re-identification in the Wild.* ACM Multimedia 2020. [arXiv:1906.05586](https://arxiv.org/abs/1906.05586)
2. Oquab M, et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
3. Deng J, Guo J, Xue N, Zafeiriou S. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019. [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
4. He K, Zhang X, Ren S, Sun J. *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
5. Lowe D. *Distinctive Image Features from Scale-Invariant Keypoints.* IJCV 2004.
6. Tyszkiewicz M, Fua P, Trulls E. *DISK: Learning Local Features with Policy Gradient.* NeurIPS 2020. [arXiv:2006.13566](https://arxiv.org/abs/2006.13566)
7. Lindenberger P, Sarlin P-E, Pollefeys M. *LightGlue: Local Feature Matching at Light Speed.* ICCV 2023. [arXiv:2306.13643](https://arxiv.org/abs/2306.13643)
8. Musgrave K, Belongie S, Lim S-N. *PyTorch Metric Learning.* 2020. [arXiv:2008.09164](https://arxiv.org/abs/2008.09164)
9. Riba E, Mishkin D, Ponsa D, Rublee E, Bradski G. *Kornia: an Open Source Differentiable Computer Vision Library for PyTorch.* WACV 2020. [arXiv:1910.02190](https://arxiv.org/abs/1910.02190)
