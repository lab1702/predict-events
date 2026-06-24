import duckdb
import pytest

from predict_events.aggregate import aggregation_expr, top_rule_order


def run(method):
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE _pe_matched("
        "antecedent VARCHAR[], confidence DOUBLE, lift DOUBLE)"
    )
    con.executemany(
        "INSERT INTO _pe_matched VALUES (?, ?, ?)",
        [
            (["a"], 0.5, 2.0),
            (["b"], 0.4, 3.0),
            (["a", "b"], 0.45, 1.0),  # most specific (largest antecedent)
        ],
    )
    expr = aggregation_expr(method)
    return con.execute(f"SELECT {expr} FROM _pe_matched").fetchone()[0]


def test_max():
    assert run("max") == pytest.approx(0.5)


def test_most_specific():
    # largest antecedent [a, b] -> its confidence 0.45
    assert run("most_specific") == pytest.approx(0.45)


def test_best_lift():
    # highest lift 3.0 ([b]) -> its confidence 0.4
    assert run("best_lift") == pytest.approx(0.4)


def test_noisy_or():
    # 1 - (1-0.5)(1-0.4)(1-0.45) = 1 - 0.165 = 0.835
    assert run("noisy_or") == pytest.approx(0.835)


def test_unknown_method():
    with pytest.raises(ValueError):
        aggregation_expr("median")


def test_top_rule_order_for_each_method():
    for method in ("max", "most_specific", "best_lift", "noisy_or"):
        assert isinstance(top_rule_order(method), str)


def test_top_rule_order_unknown_method():
    with pytest.raises(ValueError):
        top_rule_order("median")
