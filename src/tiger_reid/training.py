"""Metric-learning training loop (docs/decisions.md D12-D13).

- Batches come from MPerClassSampler (m images per entity), so every batch has
  positive pairs for the loss to learn from.
- Loss: ArcFace (a classifier over training entities with an angular margin; only
  the embedding is kept) or triplet with semi-hard mining.
- Model selection: early stopping on **val mAP** (gallery = train images). Test and
  open-set test are never looked at here.
"""

import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader
from torchvision import transforms as T

from tiger_reid.data import REPO
from tiger_reid.dataset import ReidDataset, eval_transform, extract_embeddings, train_transform
from tiger_reid.metrics import evaluate_sets
from tiger_reid.model import EmbeddingNet
from tiger_reid.splits import eval_sets, load_splits

CKPT_DIR = REPO / "checkpoints"
LOG_DIR = REPO / "results" / "train_logs"

DEFAULTS = {
    "name": "run", "backbone": "resnet50", "pretrained": True, "dim": 512, "size": [224, 224],
    "loss": "arcface", "arcface_margin": 28.6, "arcface_scale": 64.0, "triplet_margin": 0.2,
    "m_per_class": 4, "batch_size": 32, "epochs": 30, "warmup_epochs": 1, "patience": 8,
    "lr_backbone": 1e-4, "lr_head": 1e-3, "lr_loss": 1e-2, "weight_decay": 1e-4,
    "num_workers": 4, "seed": 0,
    # regularization / speed (D19); defaults reproduce the first runs
    "drop_path": 0.0, "erase": False, "crop_scale": [0.7, 1.0], "amp": False,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_train_df(train_df: pd.DataFrame, splits: pd.DataFrame, transform=None):
    """Project rules, enforced at runtime: train split only, no open-set entity, no flips."""
    assert (train_df["split"] == "train").all(), "training data must be split == 'train' only"
    openset = set(splits.loc[splits.split.str.startswith("openset"), "entity_id"])
    leak = set(train_df["entity_id"]) & openset
    assert not leak, f"open-set entities in training data: {sorted(leak)[:5]}"
    if transform is not None:
        assert not any(isinstance(t, T.RandomHorizontalFlip) for t in transform.transforms), \
            "horizontal flips are not allowed (D8)"


def build_loss(cfg: dict, n_classes: int):
    """-> (loss_fn, miner or None). ArcFace has learnable class weights (optimize them)."""
    if cfg["loss"] == "arcface":
        return losses.ArcFaceLoss(n_classes, cfg["dim"], margin=cfg["arcface_margin"],
                                  scale=cfg["arcface_scale"]), None
    if cfg["loss"] == "triplet":
        return (losses.TripletMarginLoss(margin=cfg["triplet_margin"]),
                miners.TripletMarginMiner(margin=cfg["triplet_margin"], type_of_triplets="semihard"))
    raise ValueError(cfg["loss"])


def _check_pml_version():
    import pytorch_metric_learning as pml
    if not pml.__version__.startswith("2."):
        raise RuntimeError(f"pytorch-metric-learning {pml.__version__} found; this code needs 2.x "
                           "(3.x turned MPerClassSampler into a batch sampler). "
                           "Run: pip install pytorch-metric-learning==2.9.0")


def make_train_loader(train_df: pd.DataFrame, cfg: dict) -> DataLoader:
    _check_pml_version()
    cfg = {**DEFAULTS, **cfg}
    ds = ReidDataset(train_df, train_transform(tuple(cfg["size"]), cfg["erase"], cfg["crop_scale"]))
    sampler = MPerClassSampler(ds.labels, m=cfg["m_per_class"], batch_size=cfg["batch_size"],
                               length_before_new_iter=len(ds))
    nw = cfg["num_workers"]
    return DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=nw,
                      persistent_workers=nw > 0, drop_last=True)


def make_scheduler(opt, cfg: dict, steps_per_epoch: int):
    """Linear warmup, then cosine decay to 0, stepped every batch."""
    warm, total = cfg["warmup_epochs"] * steps_per_epoch, cfg["epochs"] * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_one_epoch(model, loss_fn, miner, loader, opt, sched, device, max_batches=None,
                    scaler=None) -> float:
    """scaler: a torch.amp.GradScaler -> fp16 autocast for the backbone (CUDA only).
    The loss is computed in fp32 (ArcFace's scale 64 overflows fp16)."""
    model.train()
    loss_fn.train()
    total, n = 0.0, 0
    for b, (x, y, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler is not None):
            emb = model(x)
        emb = emb.float()
        loss = loss_fn(emb, y, miner(emb, y)) if miner is not None else loss_fn(emb, y)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def evaluate_model(model, sets: dict, size, device, ci: bool = False, keys=None,
                   num_workers: int = 0) -> dict:
    keys = keys or list(sets)
    tf = eval_transform(tuple(size))
    emb = {k: extract_embeddings(model, sets[k], tf, device, num_workers=num_workers) for k in keys}
    return evaluate_sets(emb, {k: sets[k] for k in keys}, ci=ci)


def fit(cfg: dict, device, splits: pd.DataFrame | None = None, run_name: str | None = None):
    """Train with early stopping on val mAP. Returns (model with best weights, history df)."""
    cfg = {**DEFAULTS, **cfg}
    set_seed(cfg["seed"])
    splits = load_splits() if splits is None else splits
    run_name = run_name or f"{cfg['name']}_s{cfg['seed']}"

    train_df = splits[splits.split == "train"].reset_index(drop=True)
    loader = make_train_loader(train_df, cfg)
    check_train_df(train_df, splits, loader.dataset.transform)
    n_classes = len(loader.dataset.classes)

    model = EmbeddingNet(cfg["backbone"], cfg["dim"], cfg["size"], cfg["pretrained"],
                         cfg["drop_path"]).to(device)
    loss_fn, miner = build_loss(cfg, n_classes)
    loss_fn = loss_fn.to(device)
    groups = model.param_groups(cfg["lr_backbone"], cfg["lr_head"])
    if any(True for _ in loss_fn.parameters()):
        groups.append({"params": loss_fn.parameters(), "lr": cfg["lr_loss"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    sched = make_scheduler(opt, cfg, len(loader))
    use_amp = cfg["amp"] and device.type == "cuda"   # mps/cpu stay fp32
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    val_sets = eval_sets(splits, "val")
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{run_name}.pt"

    history, best, bad = [], -1.0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, loss_fn, miner, loader, opt, sched, device, scaler=scaler)
        m = evaluate_model(model, val_sets, cfg["size"], device, keys=["gallery", "query", "unknown"])
        row = {"epoch": epoch, "train_loss": tr_loss, "val_mAP": m["mAP"], "val_rank1": m["rank1"],
               "val_auroc": m["auroc"], "lr_backbone": opt.param_groups[0]["lr"],
               "sec": time.time() - t0}
        improved = m["mAP"] > best
        if improved:
            best, bad = m["mAP"], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt)
        else:
            bad += 1
        history.append(row)
        pd.DataFrame(history).to_csv(LOG_DIR / f"{run_name}.csv", index=False)
        print(f"ep {epoch:2d} loss {tr_loss:.3f} | val mAP {m['mAP']:.3f} r1 {m['rank1']:.3f} "
              f"auroc {m['auroc']:.3f} | {row['sec']:.0f}s{' *' if improved else ''}", flush=True)
        if bad >= cfg["patience"]:
            print(f"early stop: no val mAP gain for {bad} epochs", flush=True)
            break

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    return model, pd.DataFrame(history), state["epoch"]


def load_model(ckpt_path, device=None) -> tuple[EmbeddingNet, dict]:
    """Rebuild an EmbeddingNet from a checkpoint saved by fit() -> (model in eval mode, cfg).
    No pretrained download needed: all weights come from the checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = {**DEFAULTS, **state["cfg"]}
    model = EmbeddingNet(cfg["backbone"], cfg["dim"], cfg["size"], pretrained=False)
    model.load_state_dict(state["model"])
    if device is not None:
        model = model.to(device)
    return model.eval(), cfg
