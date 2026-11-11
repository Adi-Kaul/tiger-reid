"""Append one row per evaluated run to results/results.csv (with its config)."""

import json
import subprocess
from datetime import datetime

import pandas as pd

from tiger_reid.data import REPO

RESULTS_CSV = REPO / "results" / "results.csv"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def log_result(method: str, eval_split: str, metrics: dict, config: dict | None = None,
               seed: int | None = None, path=RESULTS_CSV) -> dict:
    """eval_split: "val" (choosing settings) or "test" (reporting)."""
    row = {"time": datetime.now().isoformat(timespec="seconds"), "method": method,
           "eval_split": eval_split, "seed": seed, "splits": "v1", "commit": _git_commit(),
           **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
           "config": json.dumps(config or {}, sort_keys=True)}
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    return row
