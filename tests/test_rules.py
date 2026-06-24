import duckdb

from predict_events.config import Config
from predict_events.windowing import WindowInfo
from predict_events.rules import build_present_itemsets, generate_rules


def seed_baskets(con, rows):
    con.execute("CREATE TABLE _pe_baskets(window_id BIGINT, event VARCHAR)")
    con.executemany("INSERT INTO _pe_baskets VALUES (?, ?)", rows)


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

    base = dict(con.execute("SELECT event, baserate FROM _pe_baserates").fetchall())
    assert base["a"] == 1.0           # 4/4
    assert base["b"] == 0.75          # 3/4

    row = con.execute(
        "SELECT support, confidence, lift FROM _pe_rules "
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
        "SELECT support, confidence FROM _pe_rules "
        "WHERE antecedent = ['a'] AND consequent = 'b'"
    ).fetchone()
    # antecedent windows with 'a': 0 and 1. Both have 'b' next window. 2/2.
    assert row == (1.0, 1.0)


def test_min_confidence_prunes():
    # 'a' is frequent and predicts itself with confidence 1.0, but a->b has
    # low confidence. With min_support=0.0 the support filter never prunes,
    # so ONLY the confidence threshold can remove a->b.
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"),
        (2, "a"),
        (3, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1,
                 min_support=0.0, min_confidence=0.5)
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)
    # a->b: confidence = 1/4 = 0.25 < 0.5 -> pruned by confidence
    # (support = 0.25 >= min_support 0.0, so support does NOT prune it).
    ab = con.execute(
        "SELECT count(*) FROM _pe_rules WHERE antecedent = ['a'] AND consequent = 'b'"
    ).fetchone()[0]
    assert ab == 0
    # Sanity: b->a has confidence 1.0 and survives, proving rules are generated
    # and only the low-confidence rule was removed. (b->a is non-tautological,
    # unlike a self-rule, which is dropped at horizon 0.)
    ba = con.execute(
        "SELECT count(*) FROM _pe_rules WHERE antecedent = ['b'] AND consequent = 'a'"
    ).fetchone()[0]
    assert ba == 1


def test_drops_tautological_rules_at_horizon_zero():
    # {a,b} co-occur; at horizon 0 a rule whose consequent is one of its own
    # antecedent items (e.g. [a,b]->a) is tautological and must be dropped.
    con = duckdb.connect()
    seed_baskets(con, [(0, "a"), (0, "b"), (1, "a"), (1, "b")])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=2)
    info = WindowInfo(lo=0, hi=1, n_windows=2)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)
    taut = con.execute(
        "SELECT count(*) FROM _pe_rules WHERE list_contains(antecedent, consequent)"
    ).fetchone()[0]
    assert taut == 0


def test_keeps_recurrence_rules_at_horizon_one():
    # At horizon 1, [a]->a means 'a' recurs in the next window -- a genuine
    # prediction, not a tautology, so it must be kept.
    con = duckdb.connect()
    seed_baskets(con, [(0, "a"), (1, "a"), (2, "a")])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", horizon=1, max_antecedent_size=1)
    info = WindowInfo(lo=0, hi=2, n_windows=2)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)
    aa = con.execute(
        "SELECT count(*) FROM _pe_rules WHERE antecedent = ['a'] AND consequent = 'a'"
    ).fetchone()[0]
    assert aa == 1


def test_support_threshold_keeps_itemset_at_exact_boundary():
    # Regression: min_count was computed as `min_support * n_windows` and
    # compared against an integer count. Float error makes an itemset whose
    # support is *exactly* the threshold fail `count(*) >= min_count` and get
    # wrongly pruned. Here 0.28 * 25 == 7.000000000000001, so 'a' (present in
    # exactly 7 of 25 windows -> support exactly 0.28) must NOT be dropped at
    # min_support=0.28. 'b' co-occurs with 'a' so a real rule can form.
    con = duckdb.connect()
    rows = []
    for w in range(25):
        if w < 7:           # 'a' present in 7/25 windows -> support exactly 0.28
            rows.append((w, "a"))
        rows.append((w, "b"))  # 'b' present everywhere so a->b can form
    seed_baskets(con, rows)
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1, min_support=0.28)
    info = WindowInfo(lo=0, hi=24, n_windows=25)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)

    # 'a' survives level-1 support pruning...
    assert con.execute(
        "SELECT count(*) FROM _pe_freq_items WHERE event = 'a'"
    ).fetchone()[0] == 1
    # ...and as an antecedent itemset...
    assert con.execute(
        "SELECT count(*) FROM _pe_antecedents WHERE items = ['a']"
    ).fetchone()[0] == 1
    # ...so the a->b rule (support 0.7) is generated, not silently dropped.
    assert con.execute(
        "SELECT count(*) FROM _pe_rules "
        "WHERE antecedent = ['a'] AND consequent = 'b'"
    ).fetchone()[0] == 1


def test_rules_carry_support_count():
    # support_count is the number of windows backing the rule (the joint count).
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"),  # a in 3 windows; a&b co-occur in 2
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1)
    info = WindowInfo(lo=0, hi=2, n_windows=3)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)
    sc = con.execute(
        "SELECT support_count FROM _pe_rules WHERE antecedent = ['a'] AND consequent = 'b'"
    ).fetchone()[0]
    assert sc == 2
