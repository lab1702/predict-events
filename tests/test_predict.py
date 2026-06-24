import duckdb
import pytest

from predict_events.config import Config
from predict_events.windowing import WindowInfo
from predict_events.rules import build_present_itemsets, generate_rules
from predict_events.predict import (
    current_basket, predict_all, predict_target,
)


def seed_baskets(con, rows):
    con.execute("CREATE TABLE baskets(window_id BIGINT, event VARCHAR)")
    con.executemany("INSERT INTO baskets VALUES (?, ?)", rows)


def prepare(con, cfg, info):
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)


def test_current_basket_is_latest_window():
    con = duckdb.connect()
    seed_baskets(con, [(0, "a"), (1, "b"), (1, "c")])
    info = WindowInfo(lo=0, hi=1, n_windows=2)
    assert sorted(current_basket(con, info)) == ["b", "c"]


def test_predict_all_ranks_consequents():
    # Build history where 'a' strongly co-occurs with 'b'; latest window has 'a'.
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"), (2, "b"),
        (3, "a"),  # latest window: basket = {a}
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1, aggregation="max")
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    prepare(con, cfg, info)
    preds = predict_all(con, cfg, info)
    by_event = {p.event: p for p in preds}
    # a->b confidence = 3/4 = 0.75 (a present in 0,1,2,3; b in 0,1,2)
    assert by_event["b"].probability == pytest.approx(0.75)
    assert by_event["b"].n_rules == 1
    assert by_event["b"].fallback is False
    # results are sorted by probability desc
    assert [p.probability for p in preds] == sorted(
        (p.probability for p in preds), reverse=True
    )


def test_predict_target_with_supporting_rules():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"),  # latest basket = {a}
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1, aggregation="noisy_or")
    info = WindowInfo(lo=0, hi=2, n_windows=3)
    prepare(con, cfg, info)
    pred, supporting = predict_target(con, cfg, info, "b")
    assert pred.event == "b"
    assert pred.fallback is False
    assert pred.probability == pytest.approx(2 / 3)  # single rule conf 2/3
    assert len(supporting) == 1
    assert supporting[0].antecedent == ["a"]


def test_predict_target_falls_back_to_baserate():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "z"),
        (1, "a"),
        (2, "q"),  # latest basket = {q}, which matches no antecedent for z
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1)
    info = WindowInfo(lo=0, hi=2, n_windows=3)
    prepare(con, cfg, info)
    pred, supporting = predict_target(con, cfg, info, "z")
    assert pred.fallback is True
    assert pred.probability == pytest.approx(1 / 3)  # baserate(z) = 1/3
    assert supporting == []
