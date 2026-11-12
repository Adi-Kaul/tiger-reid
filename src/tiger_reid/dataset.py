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


class _CornerCrop:
    """Crop a (frac*H, frac*W) window from one corner (0=TL, 1=TR, 2=BL, 3=BR)."""

    def __init__(self, corner: int, frac: float = 0.9):
        self.corner, self.frac = corner, frac

    def __call__(self, img):
        w, h = img.size
        cw, ch = round(w * self.frac), round(h * self.frac)
        left = 0 if self.corner in (0, 2) else w - cw
        top = 0 if self.corner in (0, 1) else h - ch
        return img.crop((left, top, left + cw, top + ch))


def tta_transforms(size=(224, 224), frac: float = 0.9) -> list:
    """Test-time augmentation views (D18): the full image + 4 corner crops at 90%.
    No flips (a mirrored stripe pattern is a different tiger)."""
    base = eval_transform(size)
    return [base] + [T.Compose([_CornerCrop(c, frac), *base.transforms]) for c in range(4)]


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


def extract_embeddings_tta(model, df: pd.DataFrame, size=(224, 224), device=None,
                           batch_size: int = 64, num_workers: int = 0) -> np.ndarray:
    """Average L2-normalized embeddings over tta_transforms(), then re-normalize."""
    views = [extract_embeddings(model, df, tf, device, batch_size, num_workers)
             for tf in tta_transforms(size)]
    m = np.mean(views, axis=0)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
