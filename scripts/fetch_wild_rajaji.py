"""Fetch the Rajaji NP camera-trap tiger images (GBIF dataset e61455a4, CC BY-NC).

The original image host is down; GBIF's image cache still serves most files at
full size. Writes data/wild_rajaji/images/<gbif_key>.jpg and meta.csv.

meta.csv keeps only what re-ID needs: no coordinates, no locality names
(project rule: no real tiger locations). Station is the opaque catalog code.
Data stays under data/ (gitignored); never commit it.

    python scripts/fetch_wild_rajaji.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path

DATASET = "e61455a4-352d-4c55-83ea-dbca254e3b29"
API = "https://api.gbif.org/v1"
CAT_RE = re.compile(r"CTP\d{5}(\d{4})(\d{3})([A-Z])(\d+)")


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def records() -> list[dict]:
    out, offset = [], 0
    while True:
        page = get_json(f"{API}/occurrence/search?datasetKey={DATASET}&limit=300&offset={offset}")
        out += page["results"]
        offset += len(page["results"])
        if page["endOfRecords"]:
            return out


def parse(rec: dict) -> dict:
    m = CAT_RE.match(rec["catalogNumber"])
    return {
        "gbif_key": rec["key"],
        "catalog": rec["catalogNumber"],
        "year": int(m[1]),
        "station": m[2],
        "side": m[3],  # camera A/B/C at a station (paired cameras see opposite flanks)
        "frame": int(m[4]),
        "date": rec.get("eventDate", "") or "",
        "sex": rec.get("sex", "") or "",
    }


def fetch(rec: dict, out_dir: Path) -> bool:
    dst = out_dir / f"{rec['key']}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return True
    src = rec["media"][0]["identifier"]
    url = f"{API}/image/cache/occurrence/{rec['key']}/media/{hashlib.md5(src.encode()).hexdigest()}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception:
        return False
    if not data.startswith(b"\xff\xd8"):
        return False
    dst.write_bytes(data)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wild_rajaji")
    args = ap.parse_args()
    root = Path(args.out)
    (root / "images").mkdir(parents=True, exist_ok=True)

    recs = records()
    with cf.ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(lambda r: fetch(r, root / "images"), recs))
    rows = [parse(r) | {"has_image": int(o)} for r, o in zip(recs, ok)]
    rows.sort(key=lambda r: (r["year"], r["station"], r["date"], r["side"], r["frame"]))
    with open(root / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} records, {sum(ok)} images -> {root}")


if __name__ == "__main__":
    main()
