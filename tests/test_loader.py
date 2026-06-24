import duckdb
import pytest

from predict_events.config import Config
from predict_events.loader import build_source_expr, register_events


def test_build_source_expr_quotes_paths_but_not_tables():
    assert build_source_expr("data.csv") == "'data.csv'"
    assert build_source_expr("dir/data.parquet") == "'dir/data.parquet'"
    assert build_source_expr("s3://bucket/x.parquet") == "'s3://bucket/x.parquet'"
    assert build_source_expr("my_table") == "my_table"


def write_csv(path):
    path.write_text(
        "when,kind\n"
        "2024-01-01 00:00:00,login\n"
        "2024-01-01 01:00:00,purchase\n"
    )


def test_register_events_normalizes_columns(tmp_path):
    csv = tmp_path / "events.csv"
    write_csv(csv)
    con = duckdb.connect()
    cfg = Config(source=str(csv), timestamp_col="when", event_col="kind", window="1d")
    register_events(con, cfg)
    rows = con.execute("SELECT event FROM events ORDER BY ts").fetchall()
    assert rows == [("login",), ("purchase",)]
    cols = [c[0] for c in con.execute("DESCRIBE events").fetchall()]
    assert cols == ["ts", "event"]


def test_register_events_applies_where(tmp_path):
    csv = tmp_path / "events.csv"
    write_csv(csv)
    con = duckdb.connect()
    cfg = Config(
        source=str(csv), timestamp_col="when", event_col="kind",
        window="1d", where="kind = 'login'",
    )
    register_events(con, cfg)
    assert con.execute("SELECT count(*) FROM events").fetchone()[0] == 1


def test_register_events_rejects_empty(tmp_path):
    csv = tmp_path / "events.csv"
    write_csv(csv)
    con = duckdb.connect()
    cfg = Config(
        source=str(csv), timestamp_col="when", event_col="kind",
        window="1d", where="kind = 'nope'",
    )
    with pytest.raises(ValueError):
        register_events(con, cfg)
