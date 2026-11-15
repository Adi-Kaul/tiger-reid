import pytest

from tiger_reid.wild import parse_catalog, propose_groups


def test_parse_catalog():
    assert parse_catalog("CTP050612011177A00116a") == {
        "year": 2011, "station": "177", "camera": "A", "frame": 116}
    with pytest.raises(ValueError):
        parse_catalog("nope")


def _row(k, flank="L", q="g", station="100", side="A", date="2011-12-10", multi=False):
    return dict(gbif_key=k, year=2011, station=station, side=side, date=date,
                flank=flank, quality=q, multi=multi)


def test_event_grouping_same_flank_only():
    rows = [_row("a"), _row("b"), _row("c", flank="R"), _row("d", date="2011-12-11")]
    g = propose_groups(rows)
    assert g["a"] == g["b"]
    assert g["c"] != g["a"]          # other flank never merged
    assert g["d"] != g["a"]          # other day stays separate
    assert g["c"].startswith("R") and g["a"].startswith("L")


def test_multi_tiger_event_not_merged_and_unusable_dropped():
    rows = [_row("a", multi=True), _row("b"), _row("x", flank="X", q="p"), _row("p", q="p")]
    g = propose_groups(rows)
    assert g["a"] != g["b"]
    assert "x" not in g and "p" not in g


def test_manual_merge_across_events():
    rows = [_row("a"), _row("b", station="200")]
    assert propose_groups(rows)["a"] != propose_groups(rows)["b"]
    g = propose_groups(rows, manual=[["a", "b"]])
    assert g["a"] == g["b"]
