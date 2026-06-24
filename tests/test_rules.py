import duckdb

from predict_events.config import Config
from predict_events.windowing import WindowInfo
from predict_events.rules import build_present_itemsets, generate_rules


def seed_baskets(con, rows):
    con.execute("CREATE TABLE baskets(window_id BIGINT, event VARCHAR)")
    con.executemany("INSERT INTO baskets VALUES (?, ?)", rows)


def test_same_window_rule_metrics():
    # 4 windows. 'a' in all 4; 'b' in windows where 'a' is, 3 of 4.
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"), (2, "b"),
        (3, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1)
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)

    base = dict(con.execute("SELECT event, baserate FROM baserates").fetchall())
    assert base["a"] == 1.0           # 4/4
    assert base["b"] == 0.75          # 3/4

    row = con.execute(
        "SELECT support, confidence, lift FROM rules "
        "WHERE antecedent = ['a'] AND consequent = 'b'"
    ).fetchone()
    support, confidence, lift = row
    assert support == 0.75            # 3/4 windows have a&b
    assert confidence == 0.75         # 3/4 windows with a also have b
    assert lift == 1.0                # 0.75 / baserate(b)=0.75


def test_horizon_one_rule():
    # 'a' in window w predicts 'b' in window w+1.
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (1, "b"),
        (1, "a"), (2, "b"),
        (2, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", horizon=1, max_antecedent_size=1)
    # valid antecedent windows: 0,1,2 with hi=2 -> [0 .. 2-1=1]; n=(2-0+1)-1=2
    info = WindowInfo(lo=0, hi=2, n_windows=2)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)
    row = con.execute(
        "SELECT support, confidence FROM rules "
        "WHERE antecedent = ['a'] AND consequent = 'b'"
    ).fetchone()
    # antecedent windows with 'a': 0 and 1. Both have 'b' next window. 2/2.
    assert row == (1.0, 1.0)


def test_min_confidence_prunes():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"),
        (2, "a"),
        (3, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1, min_support=0.5, min_confidence=0.5)
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)
    # confidence(a->b) = 1/4 = 0.25 < 0.5 -> pruned
    rows = con.execute(
        "SELECT count(*) FROM rules WHERE consequent = 'b'"
    ).fetchone()[0]
    assert rows == 0
