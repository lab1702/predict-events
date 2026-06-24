import duckdb

from predict_events.config import Config
from predict_events.windowing import WindowInfo
from predict_events.rules import build_present_itemsets


def seed_baskets(con, rows):
    # rows: list of (window_id, event)
    con.execute("CREATE TABLE _pe_baskets(window_id BIGINT, event VARCHAR)")
    con.executemany("INSERT INTO _pe_baskets VALUES (?, ?)", rows)


def test_present_itemsets_size_1_and_2():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=2)
    info = WindowInfo(lo=0, hi=2, n_windows=3)
    build_present_itemsets(con, cfg, info)
    present = con.execute(
        "SELECT window_id, items FROM _pe_ant_present ORDER BY window_id, items"
    ).fetchall()
    assert present == [
        (0, ["a"]), (0, ["a", "b"]), (0, ["b"]),
        (1, ["a"]), (1, ["a", "b"]), (1, ["b"]),
        (2, ["a"]),
    ]
    ant = con.execute(
        "SELECT items, ant_count FROM _pe_antecedents ORDER BY items"
    ).fetchall()
    assert ant == [(["a"], 3), (["a", "b"], 2), (["b"], 2)]


def test_min_support_prunes_rare_items():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (1, "a"), (2, "a"), (3, "a"),
        (0, "rare"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=2, min_support=0.5)
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    build_present_itemsets(con, cfg, info)
    # "rare" appears in 1/4 windows < 0.5 -> excluded entirely, no pairs
    items = con.execute("SELECT DISTINCT items FROM _pe_ant_present").fetchall()
    assert items == [(["a"],)]
    ant = con.execute("SELECT items, ant_count FROM _pe_antecedents").fetchall()
    assert ant == [(["a"], 4)]


def test_horizon_excludes_late_antecedent_windows():
    # With horizon=1 and hi=2, valid antecedent windows are 0..1 (window 2 excluded).
    # 'late' appears only in window 2, so it must NOT appear in ant_present/antecedents.
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"),
        (1, "a"),
        (2, "a"), (2, "late"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", horizon=1, max_antecedent_size=2)
    info = WindowInfo(lo=0, hi=2, n_windows=2)  # (2 - 0 + 1) - 1
    build_present_itemsets(con, cfg, info)
    items = con.execute("SELECT DISTINCT items FROM _pe_ant_present ORDER BY items").fetchall()
    assert items == [(["a"],)]  # 'late' (window 2) excluded; no pairs
    ant = con.execute(
        "SELECT items, ant_count FROM _pe_antecedents ORDER BY items"
    ).fetchall()
    assert ant == [(["a"], 2)]  # 'a' present only in valid windows 0 and 1
