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
| `--aggregation` | `max` | How to score a consequent from its matching rules: `max`, `most_specific`, `best_lift`, `noisy_or` (see Methodology notes) |
| `--where` | none | Optional SQL filter applied to the raw event table |
| `--target` | none | Specific event to score (targeted mode); omit for ranked mode |
| `--top` | `20` | Number of predictions to show in the terminal table (ranked mode); does not affect `--output` |
| `--output` | none | Write the **full** ranked prediction set to a file (not truncated by `--top`); format inferred from extension (`.csv`, `.parquet`, `.json`). The file contains columns `event`, `probability`, and `n_rules`; the terminal "evidence"/top-rule column is not included. |

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

## Methodology notes

**Aggregation and calibration.** At prediction time every rule whose antecedent
is a subset of the current basket matches. `max` (default), `most_specific`,
and `best_lift` each report the confidence of a *single* rule, so the result is
a genuine conditional probability:

- `max` — the highest-confidence matching rule.
- `most_specific` — the rule with the largest antecedent (conditioned on the
  most of the basket).
- `best_lift` — the rule with the highest lift over its base rate.

`noisy_or` instead combines all matching rules as `1 - ∏(1 - confidenceᵢ)`,
assuming they are *independent* evidence. They usually are not — they are
nested/overlapping subsets of the same basket — so `noisy_or` systematically
**overestimates** (e.g. three rules each at 57% can report ~91%). Treat it as
an uncalibrated ranking score, not a probability; the CLI labels it as such.

**Reliability / sample size.** Each rule's confidence and lift are estimated
from a finite number of historical windows. The output surfaces a support
count (`n=…`) per rule and the total window count (`Model: N window(s)`) so you
can judge reliability — a rule with `n=1` is a single-window coincidence, not
evidence. Raise `--min-support` to prune low-count rules. The package does not
report confidence intervals or correct for the many comparisons made in ranked
mode, so treat thin-data predictions cautiously.

**Counting windows.** Windows are tumbling and contiguous from the first to the
last event. Support and base rates are computed over *all* windows in that
span, **including empty ones** (periods with no events). On a sparse or bursty
timeline this lowers support and base rates and raises lift; choose a `--window`
size matched to your event cadence so most windows are non-empty.

**Same-window self-matches.** At `horizon=0`, rules whose consequent is one of
their own antecedent items are tautological (confidence 1.0 by construction)
and are dropped; events already in the current basket are also excluded from
ranked output. At `horizon>=1` a recurring event is a legitimate forecast, so
these are kept.

**`--where` is raw SQL (trust boundary).** The `--where` value is injected
verbatim into the query against your source — it is *not* parameterized or
sandboxed. Pass only filter expressions you trust; never forward untrusted
input straight into `--where`. The `--source`, `--timestamp-col`, and
`--event-col` values are escaped (quoted identifiers / string literals), and
predict-events' own working tables are namespaced (`_pe_*`) so a source table
named `events`, `baskets`, or `rules` does not collide.

## License

MIT — see [LICENSE](LICENSE).
