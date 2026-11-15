"""Wild (Rajaji NP camera-trap) images: metadata parsing and identity proposals.

No individual IDs are published for this dataset, so identities are labeled by
hand. Proposals here only *suggest* groups for a human to confirm; they come
from metadata and manual visual notes, never from our re-ID models (that would
make the later evaluation circular).
"""
from __future__ import annotations

import re
from collections import defaultdict

CAT_RE = re.compile(r"CTP\d{5}(\d{4})(\d{3})([A-Z])(\d+)")


def parse_catalog(cat: str) -> dict:
    """CTP050612011177A00116a -> year 2011, station '177', camera 'A', frame 116."""
    m = CAT_RE.match(cat)
    if not m:
        raise ValueError(f"unrecognized catalog number: {cat}")
    return {"year": int(m[1]), "station": m[2], "camera": m[3], "frame": int(m[4])}


def propose_groups(rows: list[dict], manual: list[list[str]] | None = None) -> dict[str, str]:
    """Return {gbif_key: proposal_id}.

    rows: dicts with gbif_key, year, station, side, date, flank, quality, multi (bool).
    Rule 1 (event): same station + camera + date + flank, flank in L/R, quality g/o,
      and the event is not flagged as containing several tigers -> one proposal.
    Rule 2 (manual): each list in `manual` is merged into one proposal (union-find).
    Everything else stays a singleton. Unusable images (flank X) get no proposal.
    """
    parent: dict[str, str] = {}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    usable = [r for r in rows if r["flank"] in ("L", "R") and r["quality"] in ("g", "o")]
    for r in usable:
        parent[r["gbif_key"]] = r["gbif_key"]

    events: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("date"):
            events[(r["year"], r["station"], r["side"], r["date"])].append(r)
    for ev in events.values():
        if any(r.get("multi") for r in ev):
            continue
        by_flank: dict[str, list[str]] = defaultdict(list)
        for r in ev:
            if r["gbif_key"] in parent:
                by_flank[r["flank"]].append(r["gbif_key"])
        for keys in by_flank.values():
            for k in keys[1:]:
                union(keys[0], k)

    for group in manual or []:
        keys = [k for k in group if k in parent]
        for k in keys[1:]:
            union(keys[0], k)

    roots = sorted({find(k) for k in parent})
    flank = {r["gbif_key"]: r["flank"] for r in usable}
    counters = {"L": 0, "R": 0}
    name = {}
    for root in roots:
        f = flank[root]
        counters[f] += 1
        name[root] = f"{f}{counters[f]:02d}"
    return {k: name[find(k)] for k in parent}
