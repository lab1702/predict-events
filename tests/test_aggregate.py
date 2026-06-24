import duckdb
import pytest

from predict_events.aggregate import aggregation_expr


def run(method):
    con = duckdb.connect()
    con.execute("CREATE TABLE m(confidence DOUBLE, lift DOUBLE)")
    con.executemany(
        "INSERT INTO m VALUES (?, ?)", [(0.5, 2.0), (0.4, 3.0)]
    )
    expr = aggregation_expr(method)
    return con.execute(f"SELECT {expr} FROM m").fetchone()[0]


def test_noisy_or():
    # 1 - (1-0.5)(1-0.4) = 1 - 0.3 = 0.7
    assert run("noisy_or") == pytest.approx(0.7)


def test_max():
    assert run("max") == pytest.approx(0.5)


def test_best_lift():
    # highest lift is 3.0 -> its confidence is 0.4
    assert run("best_lift") == pytest.approx(0.4)


def test_unknown_method():
    with pytest.raises(ValueError):
        aggregation_expr("median")
