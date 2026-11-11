import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tiger_reid.dataset import ReidDataset, extract_embeddings, train_transform


@pytest.fixture
def toy_df(tmp_path):
    rows = []
    for i, (ent, size) in enumerate([(42, (80, 50)), (7, (300, 120)), (42, (60, 60))]):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", size, (i * 80, 40, 40)).save(p)
        rows.append({"entity_id": ent, "path": p})
    return pd.DataFrame(rows)


def test_items_and_labels(toy_df):
    ds = ReidDataset(toy_df)
    x, y, i = ds[1]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert list(ds.classes) == [7, 42]
    assert list(ds.labels) == [1, 0, 1] and (y, i) == (0, 1)


def test_train_transform_shape(toy_df):
    ds = ReidDataset(toy_df, train_transform((192, 384)))
    assert ds[0][0].shape == (3, 192, 384)


def test_no_flip_in_transforms():
    from torchvision import transforms as T
    assert not any(isinstance(t, T.RandomHorizontalFlip) for t in train_transform().transforms)


def test_extract_embeddings_order_and_shape(toy_df):
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    emb = extract_embeddings(model, toy_df, device=torch.device("cpu"), batch_size=2)
    assert emb.shape == (3, 3)
    # image 1 is brighter red than image 0 -> larger mean red channel, in input order
    assert emb[1, 0] > emb[0, 0]
