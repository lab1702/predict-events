"""Generic event prediction via market basket analysis on DuckDB."""

__all__ = ["Config", "analyze"]


def __getattr__(name):
    # Lazy re-exports so importing the package doesn't pull in duckdb until used.
    if name == "Config":
        from predict_events.config import Config

        return Config
    if name == "analyze":
        from predict_events.api import analyze

        return analyze
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
