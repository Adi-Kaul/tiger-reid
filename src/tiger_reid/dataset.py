"""PyTorch Dataset for ATRW re-ID crops + standard transforms.

No horizontal flip anywhere (docs/decisions.md D8).
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform(size=(224, 224)):
    """Resize to a fixed (H, W), to tensor, ImageNet-normalize."""
    return T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(size=(224, 224), erase: bool = False, scale=(0.7, 1.0)):
    """Mild augmentation: random crop/scale, small rotation, colour jitter. No flips.
    erase=True adds RandomErasing (hides a random patch so the model can't rely on
    one region; D19)."""
    tf = [
        T.RandomResizedCrop(size, scale=tuple(scale), ratio=(0.75 * size[1] / size[0], 1.33 * size[1] / size[0])),
        T.RandomRotation(10),
        T.ColorJitter(0.3, 0.3, 0.2, 0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if erase:
        tf.append(T.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0))
    return T.Compose(tf)


class ReidDataset(Dataset):
    """Items are (image_tensor, label, row_index).

    label = a contiguous 0..K-1 index of entity_id (losses need that); the
    mapping is `self.classes` (label -> entity_id).
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or eval_transform()
        self.classes = np.sort(self.df["entity_id"].unique())
        lookup = {e: i for i, e in enumerate(self.classes)}
        self.labels = np.array([lookup[e] for e in self.df["entity_id"]])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df.at[i, "path"]).convert("RGB")
        return self.transform(img), int(self.labels[i]), i


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract_embeddings(model, df: pd.DataFrame, transform=None, device=None,
                       batch_size: int = 64, num_workers: int = 0) -> np.ndarray:
    """Run `model` over every image in df (in order) -> (N, D) numpy array."""
    device = device or pick_device()
    model = model.to(device).eval()
    loader = DataLoader(ReidDataset(df, transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    out = []
    for x, _, _ in loader:
        out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy()
