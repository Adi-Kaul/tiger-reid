"""Build a local review page for hand-labeling Rajaji tiger identities.

Inputs (all under data/wild_rajaji/, gitignored):
  meta.csv            from scripts/fetch_wild_rajaji.py
  det_coco.json       best animal box per image (torchvision COCO detector)
  claude_notes.csv    per-image flank (L/R/F/X), quality (g/o/p), multi-tiger flag
  claude_merges.txt   proposed cross-event merges ("basis | key4 key4 ...")
Outputs:
  crops/<key>.jpg, proposals.csv, review.html (open in a browser; local only)

The page autosaves to localStorage and exports labels.csv. Then:
  python scripts/make_wild_review.py --import ~/Downloads/labels.csv
copies it to data/wild_rajaji/labels.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image

from tiger_reid.wild import propose_groups

ROOT = Path("data/wild_rajaji")


def load_merges(path: Path, key_by4: dict[str, str]) -> tuple[list[list[str]], dict[str, str]]:
    groups, basis = [], {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        why, keys = line.split("|")
        ks = [key_by4[k] for k in keys.split()]
        groups.append(ks)
        for k in ks:
            basis[k] = why.strip()
    return groups, basis


def make_crop(key: str, box, dst: Path) -> None:
    if dst.exists():
        return
    im = Image.open(ROOT / "images" / f"{key}.jpg").convert("RGB")
    if box:
        x0, y0, x1, y1 = box[0]
        px, py = (x1 - x0) * 0.08, (y1 - y0) * 0.08
        im = im.crop((max(0, x0 - px), max(0, y0 - py), min(im.width, x1 + px), min(im.height, y1 + py)))
    im.save(dst, quality=92)


def build() -> None:
    meta = {r["gbif_key"]: r for r in csv.DictReader(open(ROOT / "meta.csv")) if r["has_image"] == "1"}
    notes = {r["gbif_key"]: r for r in csv.DictReader(open(ROOT / "claude_notes.csv"))}
    det = json.load(open(ROOT / "det_coco.json"))
    key_by4 = {k[-4:]: k for k in meta}
    merges, basis = load_merges(ROOT / "claude_merges.txt", key_by4)

    rows = []
    for k, m in meta.items():
        n = notes[k]
        rows.append(dict(gbif_key=k, year=m["year"], station=m["station"], side=m["side"],
                         date=m["date"], sex=m["sex"], flank=n["flank"], quality=n["quality"],
                         multi=n["multi"] == "1", note=n["note"]))
    prop = propose_groups(rows, manual=merges)
    sizes: dict[str, int] = {}
    for p in prop.values():
        sizes[p] = sizes.get(p, 0) + 1
    for r in rows:
        k = r["gbif_key"]
        r["proposal"] = prop.get(k, "")
        if k in basis:
            r["basis"] = basis[k]
        elif r["proposal"] and sizes[r["proposal"]] > 1:
            r["basis"] = "event: same camera + day + flank"
        else:
            r["basis"] = ""

    (ROOT / "crops").mkdir(exist_ok=True)
    for r in rows:
        make_crop(r["gbif_key"], det.get(r["gbif_key"]), ROOT / "crops" / f"{r['gbif_key']}.jpg")

    cols = ["gbif_key", "flank", "quality", "proposal", "basis", "year", "station", "side", "date", "sex", "multi", "note"]
    with open(ROOT / "proposals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["proposal"] or "Z", r["date"])))

    page = (Path(__file__).parent / "wild_review_template.html").read_text()
    page = page.replace("/*DATA*/[]", json.dumps(rows))
    (ROOT / "review.html").write_text(page)
    n_groups = len({r["proposal"] for r in rows if r["proposal"]})
    n_multi = sum(1 for p, s in sizes.items() if s > 1)
    print(f"{len(rows)} images, {sum(1 for r in rows if r['proposal'])} usable, "
          f"{n_groups} proposals ({n_multi} with 2+ images) -> {ROOT / 'review.html'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imp", help="exported labels.csv to copy into data/wild_rajaji/")
    args = ap.parse_args()
    if args.imp:
        shutil.copy(args.imp, ROOT / "labels.csv")
        n = sum(1 for _ in csv.DictReader(open(ROOT / "labels.csv")))
        print(f"imported {n} labels -> {ROOT / 'labels.csv'}")
        return
    build()


if __name__ == "__main__":
    main()
