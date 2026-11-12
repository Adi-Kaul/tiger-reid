"""Embedding model: timm backbone -> linear projection -> BatchNorm -> L2-normalized vector.

The output is a unit vector, so cosine distance (metrics.cosine_distance) compares
two images directly, and the model plugs into dataset.extract_embeddings unchanged.
"""

import timm
import torch
import torch.nn.functional as F
from torch import nn


class EmbeddingNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", dim: int = 512, img_size=None,
                 pretrained: bool = True, drop_path: float = 0.0):
        super().__init__()
        kw = {"img_size": tuple(img_size)} if (img_size is not None and "vit" in backbone) else {}
        if drop_path > 0:
            kw["drop_path_rate"] = drop_path   # stochastic depth (regularization, D19)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.head = nn.Sequential(nn.Linear(self.backbone.num_features, dim), nn.BatchNorm1d(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone(x)), dim=1)

    def param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        """Lower lr for the pretrained backbone, higher for the new head."""
        return [{"params": self.backbone.parameters(), "lr": lr_backbone},
                {"params": self.head.parameters(), "lr": lr_head}]
