"""Load an arbitrary DuckDB-readable source into a normalized events view."""

from predict_events.config import Config

_FILE_HINTS = ("/", "\\", ".", "://")


def build_source_expr(source: str) -> str:
    """Quote file-ish sources for a FROM clause; leave bare table names alone."""
    if any(hint in source for hint in _FILE_HINTS):
        return f"'{source}'"
    return source


def register_events(con, cfg: Config) -> None:
    """Create a view `events(ts, event)` from the configured source."""
    if "://" in cfg.source:
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception:
            pass  # best effort; local sources don't need it
    src = build_source_expr(cfg.source)
    where = f"WHERE {cfg.where}" if cfg.where else ""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW events AS
        SELECT CAST("{cfg.timestamp_col}" AS TIMESTAMP) AS ts,
               CAST("{cfg.event_col}" AS VARCHAR) AS event
        FROM {src}
        {where}
        """
    )
    count = con.execute("SELECT count(*) FROM events").fetchone()[0]
    if count == 0:
        raise ValueError("no events loaded from source (check source/where/columns)")
