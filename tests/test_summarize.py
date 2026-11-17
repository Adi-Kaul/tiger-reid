"""summarize_results: D26 operating-point table (means over seeds)."""

import sys

import pandas as pd

sys.path.insert(0, "scripts")
from summarize_results import operating_points  # noqa: E402


def test_operating_points_table():
    base = {"method": "open-set pipeline: x", "eval_split": "test"}
    rows = []
    for seed, v in [(0, 0.1), (1, 0.3)]:
        r = {**base, "seed": seed}
        for k in ("fy5", "fy10", "keep95"):
            r |= {f"fy_{k}": v, f"fy_all_{k}": v, f"known_auto_{k}": 0.7, f"rank1_auto_{k}": 0.9}
        rows.append(r)
    rows.append({"method": "chance", "eval_split": "test", "seed": 0})
    t = operating_points(pd.DataFrame(rows))
    assert len(t) == 3 and (t.seeds == 2).all()
    assert t.iloc[0]["false yes, unknowns %"] == "20.0 ± 14.1"
    assert t.iloc[0]["sent to review (knowns) %"] == "30.0 ± 0.0"
    assert operating_points(pd.DataFrame([rows[-1]])).empty
