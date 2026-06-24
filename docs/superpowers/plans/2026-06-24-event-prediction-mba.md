# Event Prediction via Market Basket Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic, DuckDB-backed event-prediction tool that treats the most recent time window as a market basket and reports how likely a target event is, with supporting evidence.

**Architecture:** A thin Python layer (CLI + importable library) orchestrates DuckDB, which does all set/aggregation math in SQL. Pipeline: load any DuckDB-readable source → assign tumbling time windows → mine association rules across a configurable horizon → match the current window's events against rules → aggregate matched rules per target into a probability.

**Tech Stack:** Python 3.10+, `duckdb` (Python module), `argparse` (stdlib), `pytest` (dev). src-layout package.

## Global Constraints

- Python 3.10+ (uses `X | None` type syntax and `list[...]` generics).
- Runtime dependency: `duckdb` only. Dev dependency: `pytest` only. No pandas, no other libs.
- src-layout: importable package lives at `src/predict_events/`.
- All heavy set/aggregation math runs as SQL inside DuckDB, not in Python loops.
- Console script name: `predict-events` → `predict_events.cli:main`.
- This is a local analysis tool; user-supplied column/source/where strings are interpolated into SQL. That is acceptable for this tool — do NOT add SQL-escaping machinery (YAGNI), but DO quote identifiers with double quotes.
- Windows are tumbling (non-overlapping). Window ids are contiguous integers `floor((epoch(ts) - anchor) / size_seconds)`, anchor = `min(epoch(ts))`.
- Horizon `H`: an antecedent in window `w` is paired with consequents in window `w+H`. Valid antecedent windows are `window_id` in `[lo, hi - H]`. `N = (hi - lo + 1) - H` is the denominator for support/baserate.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/predict_events/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: installable package `predict_events`; `pytest` runnable.

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import predict_events
    assert predict_events.__name__ == "predict_events"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events'`

- [ ] **Step 3: Create the package and config files**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "predict-events"
version = "0.1.0"
description = "Generic event prediction via market basket analysis on DuckDB"
requires-python = ">=3.10"
dependencies = ["duckdb>=1.0"]

[project.optional-dependencies]
dev = ["pytest>=7"]

[project.scripts]
predict-events = "predict_events.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`src/predict_events/__init__.py`:
```python
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
```

`.gitignore`:
```
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
build/
dist/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/predict_events/__init__.py tests/test_smoke.py .gitignore
git commit -m "chore: scaffold predict-events package"
```

---

### Task 2: Duration parsing

**Files:**
- Create: `src/predict_events/duration.py`
- Test: `tests/test_duration.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_duration(text: str) -> int` (seconds). Accepts `<int><unit>` where unit ∈ `s,m,h,d,w` (optional whitespace). Raises `ValueError` on bad input.

- [ ] **Step 1: Write the failing test**

`tests/test_duration.py`:
```python
import pytest

from predict_events.duration import parse_duration


def test_parses_each_unit():
    assert parse_duration("30s") == 30
    assert parse_duration("5m") == 300
    assert parse_duration("2h") == 7200
    assert parse_duration("7d") == 7 * 86400
    assert parse_duration("1w") == 604800


def test_allows_internal_whitespace():
    assert parse_duration(" 7 d ") == 7 * 86400


@pytest.mark.parametrize("bad", ["", "7", "d", "7x", "-3d", "1.5h"])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_duration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.duration'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/duration.py`:
```python
"""Parse human duration strings like '7d' into seconds."""

import re

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_PATTERN = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$")


def parse_duration(text: str) -> int:
    """Return the number of seconds for a duration string like '7d' or '30m'."""
    match = _PATTERN.match(text)
    if match is None:
        raise ValueError(
            f"invalid duration {text!r}; expected <int><unit> with unit in s,m,h,d,w"
        )
    value, unit = match.groups()
    return int(value) * _UNITS[unit]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_duration.py -v`
Expected: PASS (all 9 cases)

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/duration.py tests/test_duration.py
git commit -m "feat: add duration parsing"
```

---

### Task 3: Config dataclass

**Files:**
- Create: `src/predict_events/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `predict_events.duration.parse_duration`.
- Produces: `Config` dataclass with fields:
  `source: str`, `timestamp_col: str`, `event_col: str`, `window: str`,
  `horizon: int = 0`, `max_antecedent_size: int = 2`, `min_support: float = 0.0`,
  `min_confidence: float = 0.0`, `aggregation: str = "noisy_or"`, `where: str | None = None`.
  Method `window_seconds() -> int`. Validation in `__post_init__` raising `ValueError`.
  Valid aggregation values: `"noisy_or"`, `"max"`, `"best_lift"`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import pytest

from predict_events.config import Config


def make(**over):
    base = dict(source="x.csv", timestamp_col="ts", event_col="ev", window="7d")
    base.update(over)
    return Config(**base)


def test_defaults_and_window_seconds():
    cfg = make()
    assert cfg.horizon == 0
    assert cfg.max_antecedent_size == 2
    assert cfg.aggregation == "noisy_or"
    assert cfg.window_seconds() == 7 * 86400


def test_rejects_bad_window():
    with pytest.raises(ValueError):
        make(window="nonsense")


def test_rejects_negative_horizon():
    with pytest.raises(ValueError):
        make(horizon=-1)


def test_rejects_small_antecedent_size():
    with pytest.raises(ValueError):
        make(max_antecedent_size=0)


@pytest.mark.parametrize("field", ["min_support", "min_confidence"])
def test_rejects_out_of_range_thresholds(field):
    with pytest.raises(ValueError):
        make(**{field: 1.5})


def test_rejects_unknown_aggregation():
    with pytest.raises(ValueError):
        make(aggregation="median")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.config'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/config.py`:
```python
"""Configuration for an event-prediction run."""

from dataclasses import dataclass

from predict_events.duration import parse_duration

VALID_AGGREGATIONS = ("noisy_or", "max", "best_lift")


@dataclass
class Config:
    source: str
    timestamp_col: str
    event_col: str
    window: str
    horizon: int = 0
    max_antecedent_size: int = 2
    min_support: float = 0.0
    min_confidence: float = 0.0
    aggregation: str = "noisy_or"
    where: str | None = None

    def __post_init__(self) -> None:
        parse_duration(self.window)  # raises ValueError if malformed
        if self.horizon < 0:
            raise ValueError("horizon must be >= 0")
        if self.max_antecedent_size < 1:
            raise ValueError("max_antecedent_size must be >= 1")
        for name in ("min_support", "min_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {VALID_AGGREGATIONS}"
            )

    def window_seconds(self) -> int:
        return parse_duration(self.window)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/config.py tests/test_config.py
git commit -m "feat: add Config with validation"
```

---

### Task 4: Loader (normalized events view)

**Files:**
- Create: `src/predict_events/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `Config`.
- Produces:
  - `build_source_expr(source: str) -> str` — quotes file-ish sources (`'...'`), leaves table names bare.
  - `register_events(con, cfg: Config) -> None` — creates a view `events(ts TIMESTAMP, event VARCHAR)` from the source applying optional `where`. Raises `ValueError` if zero rows load.

- [ ] **Step 1: Write the failing test**

`tests/test_loader.py`:
```python
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
    cols = [c[1] for c in con.execute("DESCRIBE events").fetchall()]
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.loader'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/loader.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/loader.py tests/test_loader.py
git commit -m "feat: add source loader and normalized events view"
```

---

### Task 5: Windowing

**Files:**
- Create: `src/predict_events/windowing.py`
- Test: `tests/test_windowing.py`

**Interfaces:**
- Consumes: `Config`; a DuckDB connection with an `events(ts, event)` view present.
- Produces:
  - `WindowInfo` dataclass: `lo: int`, `hi: int`, `n_windows: int`.
  - `assign_windows(con, cfg: Config) -> WindowInfo` — creates view `baskets(window_id BIGINT, event VARCHAR)` (DISTINCT). Computes anchor = `min(epoch(ts))`. `n_windows = (hi - lo + 1) - cfg.horizon`. Raises `ValueError` if `n_windows < 1`.

- [ ] **Step 1: Write the failing test**

`tests/test_windowing.py`:
```python
import duckdb
import pytest

from predict_events.config import Config
from predict_events.windowing import assign_windows, WindowInfo


def seed(con, rows):
    con.execute("CREATE TABLE raw(ts TIMESTAMP, ev VARCHAR)")
    con.executemany("INSERT INTO raw VALUES (?, ?)", rows)
    con.execute(
        "CREATE VIEW events AS SELECT CAST(ts AS TIMESTAMP) ts, "
        "CAST(ev AS VARCHAR) event FROM raw"
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_windowing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.windowing'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/windowing.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_windowing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/windowing.py tests/test_windowing.py
git commit -m "feat: add tumbling window assignment"
```

---

### Task 6: Frequent itemset presence

**Files:**
- Create: `src/predict_events/rules.py`
- Test: `tests/test_itemsets.py`

**Interfaces:**
- Consumes: `Config`, `WindowInfo`; a connection with `baskets` present.
- Produces (first half of `rules.py`):
  - `build_present_itemsets(con, cfg: Config, info: WindowInfo) -> None` — creates:
    - view `ant_baskets(window_id, event)` = baskets restricted to valid antecedent windows (`window_id <= hi - horizon`).
    - table `ant_present(window_id, items VARCHAR[])` = every present itemset of size `1..max_antecedent_size` per valid antecedent window, built only from items that individually meet `min_support` (items sorted ascending within each itemset).
    - table `antecedents(items VARCHAR[], ant_count BIGINT)` = itemsets whose support meets `min_support` (i.e. `ant_count >= min_support * n_windows`).
  - Internal helper `_kway_present_sql(k: int) -> str`.

- [ ] **Step 1: Write the failing test**

`tests/test_itemsets.py`:
```python
import duckdb

from predict_events.config import Config
from predict_events.windowing import WindowInfo
from predict_events.rules import build_present_itemsets


def seed_baskets(con, rows):
    # rows: list of (window_id, event)
    con.execute("CREATE TABLE baskets(window_id BIGINT, event VARCHAR)")
    con.executemany("INSERT INTO baskets VALUES (?, ?)", rows)


def test_present_itemsets_size_1_and_2():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=2)
    info = WindowInfo(lo=0, hi=2, n_windows=3)
    build_present_itemsets(con, cfg, info)
    present = con.execute(
        "SELECT window_id, items FROM ant_present ORDER BY window_id, items"
    ).fetchall()
    assert present == [
        (0, ["a"]), (0, ["a", "b"]), (0, ["b"]),
        (1, ["a"]), (1, ["a", "b"]), (1, ["b"]),
        (2, ["a"]),
    ]
    ant = con.execute(
        "SELECT items, ant_count FROM antecedents ORDER BY items"
    ).fetchall()
    assert ant == [(["a"], 3), (["a", "b"], 2), (["b"], 2)]


def test_min_support_prunes_rare_items():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (1, "a"), (2, "a"), (3, "a"),
        (0, "rare"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=2, min_support=0.5)
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    build_present_itemsets(con, cfg, info)
    # "rare" appears in 1/4 windows < 0.5 -> excluded entirely, no pairs
    items = con.execute("SELECT DISTINCT items FROM ant_present").fetchall()
    assert items == [(["a"],)]
    ant = con.execute("SELECT items, ant_count FROM antecedents").fetchall()
    assert ant == [(["a"], 4)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_itemsets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.rules'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/rules.py`:
```python
"""Mine association rules across a horizon using DuckDB SQL."""

from predict_events.config import Config
from predict_events.windowing import WindowInfo


def _kway_present_sql(k: int) -> str:
    """SQL selecting present k-itemsets per window from the `fb` view.

    Items within an itemset are ordered ascending via the join condition,
    so each present subset is enumerated exactly once.
    """
    joins = ["fb AS b1"]
    for i in range(2, k + 1):
        joins.append(
            f"JOIN fb AS b{i} ON b{i}.window_id = b1.window_id "
            f"AND b{i - 1}.event < b{i}.event"
        )
    items = ", ".join(f"b{i}.event" for i in range(1, k + 1))
    return (
        f"SELECT b1.window_id, [{items}] AS items FROM " + " ".join(joins)
    )


def build_present_itemsets(con, cfg: Config, info: WindowInfo) -> None:
    min_count = cfg.min_support * info.n_windows
    valid_hi = info.hi - cfg.horizon

    con.execute(
        f"""
        CREATE OR REPLACE VIEW ant_baskets AS
        SELECT window_id, event FROM baskets WHERE window_id <= {valid_hi}
        """
    )
    # frequent single items (support pruning on level 1)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE freq_items AS
        SELECT event FROM ant_baskets
        GROUP BY event
        HAVING count(*) >= {min_count}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW fb AS
        SELECT a.window_id, a.event
        FROM ant_baskets a JOIN freq_items f USING (event)
        """
    )
    levels = [
        _kway_present_sql(k) for k in range(1, cfg.max_antecedent_size + 1)
    ]
    con.execute(
        "CREATE OR REPLACE TABLE ant_present AS " + " UNION ALL ".join(levels)
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE antecedents AS
        SELECT items, count(*) AS ant_count
        FROM ant_present
        GROUP BY items
        HAVING count(*) >= {min_count}
        """
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_itemsets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/rules.py tests/test_itemsets.py
git commit -m "feat: build present itemsets with support pruning"
```

---

### Task 7: Rule metrics and baserates

**Files:**
- Modify: `src/predict_events/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: `Config`, `WindowInfo`; a connection where `build_present_itemsets` has run (so `ant_present`, `antecedents`, `baskets` exist).
- Produces (second half of `rules.py`):
  - `generate_rules(con, cfg: Config, info: WindowInfo) -> None` — creates:
    - table `baserates(event VARCHAR, baserate DOUBLE)` = P(event present in window `w+horizon`) over valid antecedent windows.
    - table `rules(antecedent VARCHAR[], consequent VARCHAR, support DOUBLE, confidence DOUBLE, lift DOUBLE)` for every `(itemset -> target)` passing `min_support` and `min_confidence`.

- [ ] **Step 1: Write the failing test**

`tests/test_rules.py`:
```python
import duckdb

from predict_events.config import Config
from predict_events.windowing import WindowInfo
from predict_events.rules import build_present_itemsets, generate_rules


def seed_baskets(con, rows):
    con.execute("CREATE TABLE baskets(window_id BIGINT, event VARCHAR)")
    con.executemany("INSERT INTO baskets VALUES (?, ?)", rows)


def test_same_window_rule_metrics():
    # 4 windows. 'a' in all 4; 'b' in windows where 'a' is, 3 of 4.
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"), (2, "b"),
        (3, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1)
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)

    base = dict(con.execute("SELECT event, baserate FROM baserates").fetchall())
    assert base["a"] == 1.0           # 4/4
    assert base["b"] == 0.75          # 3/4

    row = con.execute(
        "SELECT support, confidence, lift FROM rules "
        "WHERE antecedent = ['a'] AND consequent = 'b'"
    ).fetchone()
    support, confidence, lift = row
    assert support == 0.75            # 3/4 windows have a&b
    assert confidence == 0.75         # 3/4 windows with a also have b
    assert lift == 1.0                # 0.75 / baserate(b)=0.75


def test_horizon_one_rule():
    # 'a' in window w predicts 'b' in window w+1.
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (1, "b"),
        (1, "a"), (2, "b"),
        (2, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", horizon=1, max_antecedent_size=1)
    # valid antecedent windows: 0,1,2 with hi=2 -> [0 .. 2-1=1]; n=(2-0+1)-1=2
    info = WindowInfo(lo=0, hi=2, n_windows=2)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)
    row = con.execute(
        "SELECT support, confidence FROM rules "
        "WHERE antecedent = ['a'] AND consequent = 'b'"
    ).fetchone()
    # antecedent windows with 'a': 0 and 1. Both have 'b' next window. 2/2.
    assert row == (1.0, 1.0)


def test_min_confidence_prunes():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"),
        (2, "a"),
        (3, "a"),
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1, min_confidence=0.5)
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)
    # confidence(a->b) = 1/4 = 0.25 < 0.5 -> pruned
    rows = con.execute(
        "SELECT count(*) FROM rules WHERE consequent = 'b'"
    ).fetchone()[0]
    assert rows == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rules.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_rules'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/predict_events/rules.py`:
```python
def generate_rules(con, cfg: Config, info: WindowInfo) -> None:
    n = info.n_windows
    horizon = cfg.horizon
    lo = info.lo
    valid_hi = info.hi - horizon

    # consequents indexed by the antecedent window they are predicted from
    con.execute(
        f"""
        CREATE OR REPLACE VIEW cons AS
        SELECT window_id - {horizon} AS ant_wid, event AS t
        FROM baskets
        WHERE window_id - {horizon} BETWEEN {lo} AND {valid_hi}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE baserates AS
        SELECT t AS event, count(*)::DOUBLE / {n} AS baserate
        FROM cons
        GROUP BY t
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE rules AS
        WITH joint AS (
            SELECT ap.items, c.t, count(*) AS joint_cnt
            FROM ant_present ap
            JOIN cons c ON ap.window_id = c.ant_wid
            GROUP BY ap.items, c.t
        )
        SELECT
            a.items AS antecedent,
            j.t AS consequent,
            j.joint_cnt::DOUBLE / {n} AS support,
            j.joint_cnt::DOUBLE / a.ant_count AS confidence,
            (j.joint_cnt::DOUBLE / a.ant_count) / br.baserate AS lift
        FROM joint j
        JOIN antecedents a ON a.items = j.items
        JOIN baserates br ON br.event = j.t
        WHERE j.joint_cnt::DOUBLE / {n} >= {cfg.min_support}
          AND j.joint_cnt::DOUBLE / a.ant_count >= {cfg.min_confidence}
        """
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/rules.py tests/test_rules.py
git commit -m "feat: compute rule support/confidence/lift and baserates"
```

---

### Task 8: Aggregation expressions

**Files:**
- Create: `src/predict_events/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: nothing (pure string + DuckDB validation).
- Produces: `aggregation_expr(method: str) -> str` returning a DuckDB aggregate expression over columns named `confidence` and `lift`, producing a probability. Raises `ValueError` on unknown method.
  - `noisy_or` → `1 - product(1 - confidence)`
  - `max` → `max(confidence)`
  - `best_lift` → `arg_max(confidence, lift)`

- [ ] **Step 1: Write the failing test**

`tests/test_aggregate.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.aggregate'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/aggregate.py`:
```python
"""DuckDB aggregate expressions that fold matched rules into a probability."""

_EXPRESSIONS = {
    "noisy_or": "1 - product(1 - confidence)",
    "max": "max(confidence)",
    "best_lift": "arg_max(confidence, lift)",
}


def aggregation_expr(method: str) -> str:
    """Return the DuckDB aggregate expression for the given method."""
    try:
        return _EXPRESSIONS[method]
    except KeyError:
        raise ValueError(f"unknown aggregation method {method!r}") from None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aggregate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/aggregate.py tests/test_aggregate.py
git commit -m "feat: add rule aggregation expressions"
```

---

### Task 9: Prediction

**Files:**
- Create: `src/predict_events/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `Config`, `WindowInfo`, `aggregation_expr`; a connection where `generate_rules` has run (so `rules`, `baserates`, `baskets` exist).
- Produces:
  - `Prediction` dataclass: `event: str`, `probability: float`, `n_rules: int`, `fallback: bool`.
  - `SupportingRule` dataclass: `antecedent: list[str]`, `confidence: float`, `lift: float`, `support: float`.
  - `current_basket(con, info: WindowInfo) -> list[str]` — events in the most recent window (`window_id == hi`).
  - `predict_all(con, cfg, info) -> list[Prediction]` — every candidate consequent ranked by probability desc. `fallback=False` for all (rule-backed).
  - `predict_target(con, cfg, info, target: str) -> tuple[Prediction, list[SupportingRule]]` — if no rule matches, `Prediction` uses baserate with `fallback=True` and an empty supporting list.

- [ ] **Step 1: Write the failing test**

`tests/test_predict.py`:
```python
import duckdb
import pytest

from predict_events.config import Config
from predict_events.windowing import WindowInfo
from predict_events.rules import build_present_itemsets, generate_rules
from predict_events.predict import (
    current_basket, predict_all, predict_target,
)


def seed_baskets(con, rows):
    con.execute("CREATE TABLE baskets(window_id BIGINT, event VARCHAR)")
    con.executemany("INSERT INTO baskets VALUES (?, ?)", rows)


def prepare(con, cfg, info):
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)


def test_current_basket_is_latest_window():
    con = duckdb.connect()
    seed_baskets(con, [(0, "a"), (1, "b"), (1, "c")])
    info = WindowInfo(lo=0, hi=1, n_windows=2)
    assert sorted(current_basket(con, info)) == ["b", "c"]


def test_predict_all_ranks_consequents():
    # Build history where 'a' strongly co-occurs with 'b'; latest window has 'a'.
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"), (2, "b"),
        (3, "a"),  # latest window: basket = {a}
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1, aggregation="max")
    info = WindowInfo(lo=0, hi=3, n_windows=4)
    prepare(con, cfg, info)
    preds = predict_all(con, cfg, info)
    by_event = {p.event: p for p in preds}
    # a->b confidence = 3/4 = 0.75 (a present in 0,1,2,3; b in 0,1,2)
    assert by_event["b"].probability == pytest.approx(0.75)
    assert by_event["b"].n_rules == 1
    assert by_event["b"].fallback is False
    # results are sorted by probability desc
    assert [p.probability for p in preds] == sorted(
        (p.probability for p in preds), reverse=True
    )


def test_predict_target_with_supporting_rules():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "b"),
        (1, "a"), (1, "b"),
        (2, "a"),  # latest basket = {a}
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1, aggregation="noisy_or")
    info = WindowInfo(lo=0, hi=2, n_windows=3)
    prepare(con, cfg, info)
    pred, supporting = predict_target(con, cfg, info, "b")
    assert pred.event == "b"
    assert pred.fallback is False
    assert pred.probability == pytest.approx(2 / 3)  # single rule conf 2/3
    assert len(supporting) == 1
    assert supporting[0].antecedent == ["a"]


def test_predict_target_falls_back_to_baserate():
    con = duckdb.connect()
    seed_baskets(con, [
        (0, "a"), (0, "z"),
        (1, "a"),
        (2, "q"),  # latest basket = {q}, which matches no antecedent for z
    ])
    cfg = Config(source="x", timestamp_col="t", event_col="e",
                 window="1d", max_antecedent_size=1)
    info = WindowInfo(lo=0, hi=2, n_windows=3)
    prepare(con, cfg, info)
    pred, supporting = predict_target(con, cfg, info, "z")
    assert pred.fallback is True
    assert pred.probability == pytest.approx(1 / 3)  # baserate(z) = 1/3
    assert supporting == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.predict'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/predict.py`:
```python
"""Match the current basket against rules and aggregate into probabilities."""

from dataclasses import dataclass

from predict_events.aggregate import aggregation_expr
from predict_events.config import Config
from predict_events.windowing import WindowInfo


@dataclass
class Prediction:
    event: str
    probability: float
    n_rules: int
    fallback: bool


@dataclass
class SupportingRule:
    antecedent: list[str]
    confidence: float
    lift: float
    support: float


def current_basket(con, info: WindowInfo) -> list[str]:
    rows = con.execute(
        f"SELECT event FROM baskets WHERE window_id = {info.hi}"
    ).fetchall()
    return [r[0] for r in rows]


def _create_matched_view(con, info: WindowInfo) -> None:
    """Rules whose antecedent is fully contained in the current basket."""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW basket AS
        SELECT event FROM baskets WHERE window_id = {info.hi}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW matched AS
        SELECT antecedent, consequent, support, confidence, lift
        FROM rules
        WHERE list_has_all((SELECT list(event) FROM basket), antecedent)
        """
    )


def predict_all(con, cfg: Config, info: WindowInfo) -> list[Prediction]:
    _create_matched_view(con, info)
    expr = aggregation_expr(cfg.aggregation)
    rows = con.execute(
        f"""
        SELECT consequent, {expr} AS probability, count(*) AS n_rules
        FROM matched
        GROUP BY consequent
        ORDER BY probability DESC, consequent
        """
    ).fetchall()
    return [
        Prediction(event=e, probability=p, n_rules=n, fallback=False)
        for (e, p, n) in rows
    ]


def predict_target(
    con, cfg: Config, info: WindowInfo, target: str
) -> tuple[Prediction, list[SupportingRule]]:
    _create_matched_view(con, info)
    expr = aggregation_expr(cfg.aggregation)
    row = con.execute(
        f"""
        SELECT {expr} AS probability, count(*) AS n_rules
        FROM matched
        WHERE consequent = ?
        """,
        [target],
    ).fetchone()

    if row is None or row[1] == 0:
        base = con.execute(
            "SELECT baserate FROM baserates WHERE event = ?", [target]
        ).fetchone()
        probability = base[0] if base is not None else 0.0
        return (
            Prediction(event=target, probability=probability, n_rules=0, fallback=True),
            [],
        )

    probability, n_rules = row
    supporting_rows = con.execute(
        """
        SELECT antecedent, confidence, lift, support
        FROM matched
        WHERE consequent = ?
        ORDER BY lift DESC, confidence DESC
        """,
        [target],
    ).fetchall()
    supporting = [
        SupportingRule(antecedent=a, confidence=c, lift=l, support=s)
        for (a, c, l, s) in supporting_rows
    ]
    return (
        Prediction(event=target, probability=probability, n_rules=n_rules, fallback=False),
        supporting,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_predict.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/predict.py tests/test_predict.py
git commit -m "feat: add basket matching, aggregation, and ranked prediction"
```

---

### Task 10: Orchestration API

**Files:**
- Create: `src/predict_events/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Config`, `register_events`, `assign_windows`, `build_present_itemsets`, `generate_rules`, `predict_all`, `predict_target`.
- Produces:
  - `Result` dataclass: `basket: list[str]`, `predictions: list[Prediction]`, `supporting: list[SupportingRule]` (supporting non-empty only for targeted runs).
  - `analyze(cfg: Config, target: str | None = None, con=None) -> Result` — runs the full pipeline on an in-memory DuckDB connection (or a supplied one) and returns a `Result`. When `target` is `None`, `predictions` holds the ranked list and `supporting` is empty; when `target` is given, `predictions` holds a single `Prediction` and `supporting` holds its rules.

- [ ] **Step 1: Write the failing test**

`tests/test_api.py`:
```python
from predict_events.api import analyze
from predict_events.config import Config


def write_log(path):
    # 'a' precedes/co-occurs with 'b' repeatedly; latest window holds 'a'.
    lines = ["when,kind"]
    for day in range(1, 4):  # days 1..3: both a and b
        lines.append(f"2024-01-0{day} 00:00:00,a")
        lines.append(f"2024-01-0{day} 02:00:00,b")
    lines.append("2024-01-04 00:00:00,a")  # latest window: {a}
    path.write_text("\n".join(lines) + "\n")


def test_analyze_ranked(tmp_path):
    csv = tmp_path / "log.csv"
    write_log(csv)
    cfg = Config(source=str(csv), timestamp_col="when", event_col="kind",
                 window="1d", max_antecedent_size=1, aggregation="max")
    result = analyze(cfg)
    assert result.basket == ["a"]
    events = {p.event for p in result.predictions}
    assert "b" in events
    assert result.supporting == []


def test_analyze_targeted(tmp_path):
    csv = tmp_path / "log.csv"
    write_log(csv)
    cfg = Config(source=str(csv), timestamp_col="when", event_col="kind",
                 window="1d", max_antecedent_size=1)
    result = analyze(cfg, target="b")
    assert len(result.predictions) == 1
    assert result.predictions[0].event == "b"
    assert result.predictions[0].probability > 0.0
    assert result.supporting[0].antecedent == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.api'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/api.py`:
```python
"""High-level orchestration: source -> windows -> rules -> prediction."""

from dataclasses import dataclass

import duckdb

from predict_events.config import Config
from predict_events.loader import register_events
from predict_events.predict import (
    Prediction,
    SupportingRule,
    current_basket,
    predict_all,
    predict_target,
)
from predict_events.rules import build_present_itemsets, generate_rules
from predict_events.windowing import assign_windows


@dataclass
class Result:
    basket: list[str]
    predictions: list[Prediction]
    supporting: list[SupportingRule]


def analyze(cfg: Config, target: str | None = None, con=None) -> Result:
    if con is None:
        con = duckdb.connect()
    register_events(con, cfg)
    info = assign_windows(con, cfg)
    build_present_itemsets(con, cfg, info)
    generate_rules(con, cfg, info)

    basket = current_basket(con, info)
    if target is None:
        return Result(basket=basket, predictions=predict_all(con, cfg, info),
                      supporting=[])
    prediction, supporting = predict_target(con, cfg, info, target)
    return Result(basket=basket, predictions=[prediction], supporting=supporting)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict_events/api.py tests/test_api.py
git commit -m "feat: add analyze() orchestration entrypoint"
```

---

### Task 11: CLI

**Files:**
- Create: `src/predict_events/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Config`, `analyze`.
- Produces:
  - `build_parser() -> argparse.ArgumentParser`.
  - `config_from_args(args) -> Config`.
  - `format_result(result, target) -> str` — human-readable text.
  - `main(argv: list[str] | None = None) -> int` — parses args, runs `analyze`, prints the formatted result, returns exit code 0. Flags: `--source` (required), `--timestamp-col` (required), `--event-col` (required), `--window` (required), `--horizon`, `--max-antecedent-size`, `--min-support`, `--min-confidence`, `--aggregation`, `--where`, `--target`, `--top` (limit ranked rows, default 20).

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from predict_events.cli import build_parser, config_from_args, format_result, main
from predict_events.predict import Prediction, SupportingRule
from predict_events.api import Result


def write_log(path):
    lines = ["when,kind"]
    for day in range(1, 4):
        lines.append(f"2024-01-0{day} 00:00:00,a")
        lines.append(f"2024-01-0{day} 02:00:00,b")
    lines.append("2024-01-04 00:00:00,a")
    path.write_text("\n".join(lines) + "\n")


def test_config_from_args_maps_flags():
    parser = build_parser()
    args = parser.parse_args([
        "--source", "x.csv", "--timestamp-col", "ts", "--event-col", "ev",
        "--window", "7d", "--horizon", "2", "--aggregation", "max",
        "--min-support", "0.1",
    ])
    cfg = config_from_args(args)
    assert cfg.source == "x.csv"
    assert cfg.horizon == 2
    assert cfg.aggregation == "max"
    assert cfg.min_support == 0.1


def test_format_targeted_mentions_probability_and_evidence():
    result = Result(
        basket=["a"],
        predictions=[Prediction(event="b", probability=0.75, n_rules=1, fallback=False)],
        supporting=[SupportingRule(antecedent=["a"], confidence=0.75, lift=1.0, support=0.6)],
    )
    text = format_result(result, target="b")
    assert "b" in text
    assert "75" in text  # probability rendered as percentage
    assert "a" in text   # supporting antecedent shown


def test_main_runs_end_to_end(tmp_path, capsys):
    csv = tmp_path / "log.csv"
    write_log(csv)
    code = main([
        "--source", str(csv), "--timestamp-col", "when", "--event-col", "kind",
        "--window", "1d", "--target", "b", "--max-antecedent-size", "1",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "b" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_events.cli'`

- [ ] **Step 3: Write minimal implementation**

`src/predict_events/cli.py`:
```python
"""Command-line interface for predict-events."""

import argparse

from predict_events.api import analyze, Result
from predict_events.config import VALID_AGGREGATIONS, Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict-events",
        description="Predict event likelihood via market basket analysis.",
    )
    p.add_argument("--source", required=True, help="DuckDB-readable path/URL/table")
    p.add_argument("--timestamp-col", required=True)
    p.add_argument("--event-col", required=True)
    p.add_argument("--window", required=True, help="window size, e.g. 7d, 1h, 30m")
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--max-antecedent-size", type=int, default=2)
    p.add_argument("--min-support", type=float, default=0.0)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--aggregation", choices=VALID_AGGREGATIONS, default="noisy_or")
    p.add_argument("--where", default=None)
    p.add_argument("--target", default=None, help="predict a specific event")
    p.add_argument("--top", type=int, default=20, help="rows to show in ranked mode")
    return p


def config_from_args(args) -> Config:
    return Config(
        source=args.source,
        timestamp_col=args.timestamp_col,
        event_col=args.event_col,
        window=args.window,
        horizon=args.horizon,
        max_antecedent_size=args.max_antecedent_size,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        aggregation=args.aggregation,
        where=args.where,
    )


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_result(result: Result, target: str | None) -> str:
    lines = [f"Current basket: {{{', '.join(result.basket) or '(empty)'}}}", ""]
    if target is not None:
        pred = result.predictions[0]
        note = " (no matching rules; base rate)" if pred.fallback else ""
        lines.append(f"{pred.event}: {_pct(pred.probability)}{note}")
        if result.supporting:
            lines.append("because:")
            for r in result.supporting:
                lines.append(
                    f"  {{{', '.join(r.antecedent)}}} -> {pred.event}  "
                    f"conf={_pct(r.confidence)} lift={r.lift:.2f}"
                )
    else:
        lines.append(f"{'event':<24}{'probability':>12}{'rules':>8}")
        for pred in result.predictions:
            lines.append(f"{pred.event:<24}{_pct(pred.probability):>12}{pred.n_rules:>8}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    result = analyze(cfg, target=args.target)
    if args.target is None:
        result.predictions = result.predictions[: args.top]
    print(format_result(result, args.target))
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and the installed CLI**

Run: `pytest -v`
Expected: ALL tests pass.

Run: `pip install -e . && predict-events --help`
Expected: usage text prints, exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/predict_events/cli.py tests/test_cli.py
git commit -m "feat: add command-line interface"
```

---

## Self-Review Notes

- **Spec coverage:** generic DuckDB input (Task 4), configurable window/horizon (Tasks 3, 5), rule mining with configurable max antecedent size (Tasks 6, 7), support/confidence/lift + baserates (Task 7), noisy_or/max/best_lift aggregation (Task 8), targeted + ranked output with explainability and no-match fallback (Tasks 9, 11), Python CLI + library (Tasks 10, 11). All spec sections map to a task.
- **No-match fallback** lives in `predict_target` (Task 9) and is surfaced by the CLI note (Task 11).
- **Type consistency:** `WindowInfo`, `Prediction`, `SupportingRule`, `Result`, and the SQL relation/column names (`baskets`, `ant_present`, `antecedents`, `rules`, `baserates`, `matched`; columns `antecedent`, `consequent`, `support`, `confidence`, `lift`) are used identically across tasks.
- **Note on itemset scaling:** pruning is at the individually-frequent-item level before k-way joins, not full apriori per-level antimonotone pruning. Correct for any `min_support`; for large `max_antecedent_size` on dense data this is heavier than full apriori. Acceptable given default `max_antecedent_size=2`. Revisit only if profiling shows a problem (YAGNI).
