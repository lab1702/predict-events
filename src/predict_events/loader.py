"""Load an arbitrary DuckDB-readable source into a normalized events view."""

from predict_events.config import Config

# A source is treated as a file (and quoted as a string literal) when it
# contains a path separator / URL scheme, or ends with a recognized data-file
# extension. Anything else is a (possibly schema-qualified) table identifier
# and is left bare so references like `main.events` keep working.
_PATH_MARKERS = ("/", "\\", "://")
_DATA_EXTENSIONS = (
    ".parquet", ".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson",
    ".arrow", ".feather", ".xlsx", ".gz", ".zst",
)


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier, escaping embedded double quotes."""
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    """Quote a SQL string literal, escaping embedded single quotes."""
    return "'" + value.replace("'", "''") + "'"


def build_source_expr(source: str) -> str:
    """Quote file-ish sources for a FROM clause; leave table names alone.

    A bare dotted name such as ``main.events`` is a schema-qualified table
    reference, not a file, so it is left unquoted; a name ending in a known
    data-file extension (or containing a path separator/URL scheme) is quoted
    as a string literal.
    """
    looks_like_file = any(m in source for m in _PATH_MARKERS) or (
        source.lower().endswith(_DATA_EXTENSIONS)
    )
    if looks_like_file:
        return _quote_literal(source)
    return source


def register_events(con, cfg: Config) -> None:
    """Create a view `_pe_events(ts, event)` from the configured source."""
    if "://" in cfg.source:
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception:
            pass  # best effort; local sources don't need it
    src = build_source_expr(cfg.source)
    where = f"WHERE {cfg.where}" if cfg.where else ""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW _pe_events AS
        SELECT CAST({_quote_ident(cfg.timestamp_col)} AS TIMESTAMP) AS ts,
               CAST({_quote_ident(cfg.event_col)} AS VARCHAR) AS event
        FROM {src}
        {where}
        """
    )
    count = con.execute("SELECT count(*) FROM _pe_events").fetchone()[0]
    if count == 0:
        raise ValueError("no events loaded from source (check source/where/columns)")
