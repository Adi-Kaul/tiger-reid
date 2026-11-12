"""Smoke test: a tiny untrained model trains for 2 batches on CPU with each loss."""

import math

import pandas as pd
import pytest
import torch
from PIL import Image

from tiger_reid.model import EmbeddingNet
from tiger_reid.training import (DEFAULTS, build_loss, check_train_df, make_train_loader,
                                 train_one_epoch)


@pytest.fixture
def toy_splits(tmp_path):
    rows = []
    for ent in range(8):
        for k in range(4):
            p = tmp_path / f"{ent}_{k}.jpg"
            Image.new("RGB", (60 + 5 * k, 40), (ent * 30, k * 50, 90)).save(p)
            rows.append({"entity_id": ent, "path": p, "split": "train"})
    rows.append({"entity_id": 99, "path": rows[0]["path"], "split": "openset_val"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("loss", ["arcface", "triplet"])
def test_two_batches_train(toy_splits, loss):
    torch.manual_seed(0)
    cfg = {**DEFAULTS, "backbone": "resnet18", "pretrained": False, "dim": 32, "size": [64, 64],
           "loss": loss, "batch_size": 16, "m_per_class": 4, "num_workers": 0}
    train_df = toy_splits[toy_splits.split == "train"].reset_index(drop=True)
    loader = make_train_loader(train_df, cfg)
    check_train_df(train_df, toy_splits, loader.dataset.transform)
    assert len(loader) == 2

    model = EmbeddingNet(cfg["backbone"], cfg["dim"], pretrained=False)
    loss_fn, miner = build_loss(cfg, n_classes=8)
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    before = [p.detach().clone() for p in loss_fn.parameters()]
    tr_loss = train_one_epoch(model, loss_fn, miner, loader, opt, None, torch.device("cpu"))
    assert math.isfinite(tr_loss)
    if loss == "arcface":   # class weights must be optimized too
        assert any(not torch.equal(a, b) for a, b in zip(before, loss_fn.parameters()))

    model.eval()
    x, _, _ = next(iter(loader))
    with torch.no_grad():
        z = model(x)
    assert z.shape == (16, 32)


def test_guard_rejects_openset_and_flips(toy_splits):
    from torchvision import transforms as T
    leaky = toy_splits.copy()
    leaky.loc[leaky.entity_id == 99, "split"] = "train"
    with pytest.raises(AssertionError):
        check_train_df(leaky[leaky.split == "train"].assign(entity_id=lambda d: d.entity_id),
                       toy_splits)
    train_df = toy_splits[toy_splits.split == "train"]
    with pytest.raises(AssertionError):
        check_train_df(train_df, toy_splits, T.Compose([T.RandomHorizontalFlip()]))
    with pytest.raises(AssertionError):
        check_train_df(toy_splits, toy_splits)   # includes a non-train row


def test_regularized_config_trains(toy_splits, tmp_path):
    """drop_path + RandomErasing + wider crop scale: still trains, still no flips."""
    from torchvision import transforms as T
    torch.manual_seed(0)
    cfg = {**DEFAULTS, "backbone": "resnet18", "pretrained": False, "dim": 32, "size": [64, 64],
           "batch_size": 16, "num_workers": 0, "drop_path": 0.1, "erase": True,
           "crop_scale": [0.5, 1.0]}
    train_df = toy_splits[toy_splits.split == "train"].reset_index(drop=True)
    loader = make_train_loader(train_df, cfg)
    tfs = loader.dataset.transform.transforms
    assert any(isinstance(t, T.RandomErasing) for t in tfs)
    check_train_df(train_df, toy_splits, loader.dataset.transform)
    model = EmbeddingNet(cfg["backbone"], cfg["dim"], pretrained=False, drop_path=0.1)
    loss_fn, _ = build_loss(cfg, n_classes=8)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)
    assert math.isfinite(train_one_epoch(model, loss_fn, None, loader, opt, None, torch.device("cpu")))

    # checkpoint round trip through load_model
    from tiger_reid.training import load_model
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": 1}, ckpt)
    m2, cfg2 = load_model(ckpt)
    x, _, _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), m2(x), atol=1e-6)
    assert cfg2["drop_path"] == 0.1


def test_new_configs_parse():
    import yaml
    from tiger_reid.data import REPO
    for name, reg in [("dinov2s_reg", True), ("dinov2s_336", False), ("dinov2b", False)]:
        cfg = {**DEFAULTS, **yaml.safe_load(open(REPO / "configs" / f"{name}.yaml"))}
        assert cfg["name"] == name
        # B1 has the D19 regularization; B2/B3 use the plain settings (D22)
        assert cfg["erase"] == reg and (cfg["drop_path"] == 0.1) == reg
        assert cfg["crop_scale"] == ([0.5, 1.0] if reg else [0.7, 1.0])
        if "vit" in cfg["backbone"]:
            assert cfg["size"][0] % 14 == 0
