"""Assign events to tumbling time windows (baskets)."""

from dataclasses import dataclass

from predict_events.config import Config


@dataclass
class WindowInfo:
    lo: int
    hi: int
    n_windows: int


def assign_windows(con, cfg: Config) -> WindowInfo:
    """Create the `baskets` view and return window bounds + valid window count."""
    size = cfg.window_seconds()
    con.execute("CREATE OR REPLACE TABLE _win_anchor AS SELECT min(epoch(ts)) AS a FROM events")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW baskets AS
        SELECT DISTINCT
            CAST(floor((epoch(ts) - (SELECT a FROM _win_anchor)) / {size}) AS BIGINT)
                AS window_id,
            event
        FROM events
        """
    )
    lo, hi = con.execute("SELECT min(window_id), max(window_id) FROM baskets").fetchone()
    n_windows = (hi - lo + 1) - cfg.horizon
    if n_windows < 1:
        raise ValueError(
            f"insufficient history: {hi - lo + 1} windows with horizon {cfg.horizon}"
        )
    return WindowInfo(lo=lo, hi=hi, n_windows=n_windows)
