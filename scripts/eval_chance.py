"""Chance-level floor: random embeddings through the real eval pipeline.

    python scripts/eval_chance.py
Expect rank-1 ~ (avg gallery images per entity / gallery size) and AUROC ~ 0.5.
"""

import numpy as np

from tiger_reid.metrics import evaluate_sets
from tiger_reid.results import log_result
from tiger_reid.splits import eval_sets, load_splits


def main():
    sets = eval_sets(load_splits(), "test")
    for seed in range(3):
        rng = np.random.default_rng(seed)
        emb = {k: rng.normal(size=(len(v), 128)) for k, v in sets.items()}
        m = evaluate_sets(emb, sets)
        log_result("chance (random embeddings)", "test", m, {"dim": 128}, seed=seed)
        print(seed, {k: round(v, 3) for k, v in m.items()})


if __name__ == "__main__":
    main()
