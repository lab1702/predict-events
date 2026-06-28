"""Generic event prediction via market basket analysis on DuckDB."""

from predict_events.api import analyze
from predict_events.config import Config

__all__ = ["Config", "analyze"]
