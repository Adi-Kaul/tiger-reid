"""Train an embedding model, pick the best epoch on val mAP, log val (and optionally test).

    python scripts/train.py --config configs/resnet50_arcface.yaml
    python scripts/train.py --config configs/resnet50_arcface.yaml --seed 1 --eval-test

Use --eval-test only for final configs (chosen on val), once each.
"""

import argparse
import time

import torch
import yaml

from tiger_reid.dataset import pick_device
from tiger_reid.results import log_result
from tiger_reid.splits import eval_sets, load_splits
from tiger_reid.training import DEFAULTS, evaluate_model, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--num-workers", type=int, help="dataloader workers (Colab has 2 CPUs)")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="don't write results.csv (debug runs)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = {**DEFAULTS, **yaml.safe_load(f)}
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    device = pick_device()
    print(f"config: {cfg}\ndevice: {device}", flush=True)
    t0 = time.time()
    splits = load_splits()
    model, hist, best_epoch = fit(cfg, device, splits)
    cfg_log = {**cfg, "best_epoch": best_epoch, "epochs_run": len(hist),
               "train_minutes": round((time.time() - t0) / 60, 1),
               "device": torch.cuda.get_device_name() if device.type == "cuda" else device.type}
    method = f"{cfg['loss']} {cfg['backbone']}"

    stages = ["val"] + (["test"] if args.eval_test else [])
    for stage in stages:
        m = evaluate_model(model, eval_sets(splits, stage), cfg["size"], device, ci=True)
        if not args.no_log:
            log_result(method, stage, m, cfg_log, seed=cfg["seed"])
        print(f"[{stage}] best epoch {best_epoch}:",
              {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}, flush=True)


if __name__ == "__main__":
    main()
