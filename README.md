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

**Forecasting with `--horizon`** — predict the *next* window instead of the current one:

```
predict-events \
  --source events.parquet \
  --timestamp-col ts \
  --event-col event \
  --window 1d \
  --horizon 1
```

`--horizon 0` (the default) asks "given the events already in the current
window, what *else* belongs in it?" — events already present are excluded from
ranked output as trivial self-matches. `--horizon 1` instead asks "given the
current window, what happens in the *following* window?", so a rule means
`antecedent in window w` ⇒ `consequent in window w+1`, and recurring events are
kept (an event that reliably repeats next window is a genuine forecast). For
example, the same `{browse, login}` basket might rank `checkout` at 90.8% for
same-window co-occurrence (`horizon=0`) but 85.7% as a next-window forecast
(`horizon=1`). Use larger horizons to forecast further ahead.

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
| `--output` | none | Write predictions to a file; format inferred from extension (`.csv`, `.parquet`, `.json`). The file contains columns `event`, `probability`, and `n_rules`; the terminal "evidence"/top-rule column is not included. |

## Library usage

```python
from predict_events import analyze, Config

cfg = Config(
    source="events.parquet",
    timestamp_col="ts",
    event_col="event",
    window="7d",
    horizon=1,  # forecast the next window; 0 = same-window co-occurrence
)
result = analyze(cfg, target="Checkout")  # target is an analyze() argument
print(result.predictions[0].probability)
```

## License

MIT — see [LICENSE](LICENSE).
