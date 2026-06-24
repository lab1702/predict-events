import duckdb
import pytest

from predict_events.config import Config
from predict_events.windowing import assign_windows, WindowInfo


def seed(con, rows):
    con.execute("CREATE TABLE raw(ts TIMESTAMP, ev VARCHAR)")
    con.executemany("INSERT INTO raw VALUES (?, ?)", rows)
    con.execute(
        "CREATE VIEW events AS SELECT CAST(ts AS TIMESTAMP) ts, "
        "CAST(ev AS VARCHAR) \"event\" FROM raw"
    )


def test_assigns_tumbling_daily_windows():
    con = duckdb.connect()
    seed(con, [
        ("2024-01-01 00:00:00", "a"),  # window 0
        ("2024-01-01 12:00:00", "b"),  # window 0
        ("2024-01-02 06:00:00", "a"),  # window 1
        ("2024-01-04 06:00:00", "c"),  # window 3
    ])
    cfg = Config(source="raw", timestamp_col="ts", event_col="ev", window="1d")
    info = assign_windows(con, cfg)
    assert isinstance(info, WindowInfo)
    assert (info.lo, info.hi) == (0, 3)
    assert info.n_windows == 4  # (3 - 0 + 1) - horizon 0
    baskets = con.execute(
        "SELECT window_id, event FROM baskets ORDER BY window_id, event"
    ).fetchall()
    assert baskets == [(0, "a"), (0, "b"), (1, "a"), (3, "c")]


def test_horizon_reduces_window_count():
    con = duckdb.connect()
    seed(con, [
        ("2024-01-01 00:00:00", "a"),
        ("2024-01-03 00:00:00", "b"),  # window 2
    ])
    cfg = Config(source="raw", timestamp_col="ts", event_col="ev",
                 window="1d", horizon=1)
    info = assign_windows(con, cfg)
    assert (info.lo, info.hi) == (0, 2)
    assert info.n_windows == 2  # (2 - 0 + 1) - 1


def test_rejects_insufficient_history():
    con = duckdb.connect()
    seed(con, [("2024-01-01 00:00:00", "a")])
    cfg = Config(source="raw", timestamp_col="ts", event_col="ev",
                 window="1d", horizon=1)
    with pytest.raises(ValueError):
        assign_windows(con, cfg)  # (0-0+1)-1 = 0 windows


def test_baskets_are_distinct():
    con = duckdb.connect()
    seed(con, [
        ("2024-01-01 00:00:00", "a"),  # window 0
        ("2024-01-01 06:00:00", "a"),  # window 0 again -> duplicate (0, "a")
        ("2024-01-01 12:00:00", "b"),  # window 0
    ])
    cfg = Config(source="raw", timestamp_col="ts", event_col="ev", window="1d")
    assign_windows(con, cfg)
    baskets = con.execute(
        "SELECT window_id, event FROM baskets ORDER BY window_id, event"
    ).fetchall()
    assert baskets == [(0, "a"), (0, "b")]  # duplicate (0,"a") collapsed


def test_rejects_empty_events():
    con = duckdb.connect()
    con.execute("CREATE TABLE raw(ts TIMESTAMP, ev VARCHAR)")
    con.execute(
        "CREATE VIEW events AS SELECT CAST(ts AS TIMESTAMP) ts, "
        'CAST(ev AS VARCHAR) AS event FROM raw'
    )
    cfg = Config(source="raw", timestamp_col="ts", event_col="ev", window="1d")
    with pytest.raises(ValueError):
        assign_windows(con, cfg)
