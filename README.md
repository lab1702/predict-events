# predict-events

`predict-events` predicts the likelihood of future events using market basket analysis on DuckDB. It partitions an event log into time windows, treats each window as a "basket", mines association rules from historical windows, and matches the most recent window (the current basket) against those rules to rank or score candidate events. A configurable `horizon` parameter controls whether predictions target events within the same window (horizon=0) or in a following window (horizon>=1).

## Install

```
pip install -e .
```

## CLI usage

**Targeted mode** — score a specific event:

```
predict-events \
  --source events.parquet \
  --timestamp-col ts \
  --event-col event \
  --window 7d \
  --target Checkout
```

**Ranked mode** — rank all likely next events:

```
predict-events \
  --source events.parquet \
  --timestamp-col ts \
  --event-col event \
  --window 7d
```

## Flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--window` | required | Window size, e.g. `7d`, `1h`, `30m` |
| `--horizon` | `0` | Offset (in windows) between antecedent and consequent windows |
| `--max-antecedent-size` | `2` | Maximum items in a rule antecedent |
| `--min-support` | `0.0` | Minimum rule support threshold (0–1) |
| `--min-confidence` | `0.0` | Minimum rule confidence threshold (0–1) |
| `--aggregation` | `noisy_or` | How to combine multiple matching rules: `noisy_or`, `max`, `best_lift` |
| `--where` | none | Optional SQL filter applied to the raw event table |
| `--target` | none | Specific event to score (targeted mode); omit for ranked mode |
| `--top` | `20` | Number of predictions to show in ranked mode |
| `--output` | none | Write predictions to a file; format inferred from extension (`.csv`, `.parquet`, `.json`) |

## Library usage

```python
from predict_events import analyze, Config

cfg = Config(
    source="events.parquet",
    timestamp_col="ts",
    event_col="event",
    window="7d",
    target="Checkout",
)
result = analyze(cfg, target="Checkout")
print(result.predictions[0].probability)
```
