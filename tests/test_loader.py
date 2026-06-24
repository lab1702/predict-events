import duckdb
import pytest

from predict_events.config import Config
from predict_events.loader import (
    _quote_ident,
    _quote_literal,
    build_source_expr,
    register_events,
)


def test_build_source_expr_quotes_paths_but_not_tables():
    assert build_source_expr("data.csv") == "'data.csv'"
    assert build_source_expr("dir/data.parquet") == "'dir/data.parquet'"
    assert build_source_expr("s3://bucket/x.parquet") == "'s3://bucket/x.parquet'"
    assert build_source_expr("my_table") == "my_table"


def test_build_source_expr_treats_schema_qualified_table_as_identifier():
    # A dotted name with no path separator and no data-file extension is a
    # schema-qualified table reference, not a file, and must stay unquoted.
    assert build_source_expr("main.events") == "main.events"
    assert build_source_expr("db.schema.tbl") == "db.schema.tbl"


def test_build_source_expr_escapes_single_quotes_in_file_paths():
    assert build_source_expr("weird'name.csv") == "'weird''name.csv'"


def test_quote_ident_escapes_embedded_double_quotes():
    assert _quote_ident("col") == '"col"'
    assert _quote_ident('a"b') == '"a""b"'


def test_quote_literal_escapes_embedded_single_quotes():
    assert _quote_literal("plain") == "'plain'"
    assert _quote_literal("o'brien") == "'o''brien'"


def test_register_events_accepts_source_table_named_events():
    # A base table named exactly "events" must not collide with the internal
    # normalized view (previously caused a self-referencing CREATE VIEW).
    con = duckdb.connect()
    con.execute("CREATE TABLE events(occurred_at TIMESTAMP, kind VARCHAR)")
    con.execute(
        "INSERT INTO events VALUES "
        "('2024-01-01 00:00:00', 'a'), ('2024-01-02 00:00:00', 'b')"
    )
    cfg = Config(source="events", timestamp_col="occurred_at", event_col="kind", window="1d")
    register_events(con, cfg)
    assert con.execute("SELECT count(*) FROM _pe_events").fetchone()[0] == 2


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
    rows = con.execute("SELECT event FROM _pe_events ORDER BY ts").fetchall()
    assert rows == [("login",), ("purchase",)]
    cols = [c[0] for c in con.execute("DESCRIBE _pe_events").fetchall()]
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
    assert con.execute("SELECT count(*) FROM _pe_events").fetchone()[0] == 1


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
