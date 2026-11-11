"""Paths and annotation loading for ATRW re-ID (plumbing, no ML here)."""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ATRW = REPO / "data" / "atrw"
TRAIN_IMG_DIR = ATRW / "atrw_reid_train" / "train"
TRAIN_CSV = ATRW / "atrw_anno_reid_train" / "reid_list_train.csv"


def load_labeled(data_root: Path | None = None) -> pd.DataFrame:
    """Return the 1,887 labeled train images as columns: entity_id, filename, path.

    `entity_id` is ATRW's "tiger_id", renamed because it is one *flank* of one tiger,
    not one tiger (see docs/progress.md, Oct 6).
    Pass `data_root` in Colab, where the data lives somewhere else.
    """
    root = Path(data_root) if data_root else ATRW
    df = pd.read_csv(root / "atrw_anno_reid_train" / "reid_list_train.csv",
                     header=None, names=["entity_id", "filename"])
    df["path"] = [root / "atrw_reid_train" / "train" / f for f in df["filename"]]
    return df.sort_values(["entity_id", "filename"]).reset_index(drop=True)
