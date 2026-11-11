"""Off-the-shelf baseline: a pretrained timm backbone's pooled features, no training.

    python scripts/eval_pretrained.py --model resnet50 --size 224 224 --stage val
    python scripts/eval_pretrained.py --model resnet50 --size 224 224 --stage test

Choose settings on --stage val; run --stage test only for the chosen setting.
"""

import argparse
import time

import timm
import torch

from tiger_reid.dataset import eval_transform, extract_embeddings, pick_device
from tiger_reid.metrics import evaluate_sets
from tiger_reid.results import log_result
from tiger_reid.splits import eval_sets, load_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    kw = {"img_size": tuple(args.size)} if "vit" in args.model else {}   # ViTs: resize pos. embeddings
    model = timm.create_model(args.model, pretrained=True, num_classes=0, **kw)   # pooled features
    cfg = model.pretrained_cfg
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(tuple(args.size)), T.ToTensor(),
                    T.Normalize(cfg.get("mean"), cfg.get("std"))])

    sets = eval_sets(load_splits(), args.stage)
    t0 = time.time()
    emb = {k: extract_embeddings(model, v, tf, device) for k, v in sets.items()}
    m = evaluate_sets(emb, sets)
    config = {"model": args.model, "size": args.size, "pretrained": cfg.get("hf_hub_id") or cfg.get("tag"),
              "features": "pooled, no training", "distance": "cosine"}
    log_result(f"pretrained {args.model}", args.stage, m, config, seed=0)
    print(f"{args.model} {args.size} [{args.stage}] dim={emb['gallery'].shape[1]} "
          f"({time.time() - t0:.0f}s on {device}):",
          {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    main()
