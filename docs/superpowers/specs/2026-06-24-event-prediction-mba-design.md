# Generic Event Prediction via Market Basket Analysis — Design

**Date:** 2026-06-24
**Status:** Approved (brainstorm), pending implementation plan

## Summary

A generic, domain-agnostic event-prediction tool. It treats the most recent
time window as an in-progress "basket" of events, mines association rules from
similar-sized historical windows, and reports how likely a target event is —
together with the supporting evidence ("EventA is X% likely because B and C
happened recently, and historically when B and C happen, EventA also happens").

The analytical heavy lifting runs as SQL inside **DuckDB**, so the system reads
any DuckDB-supported source (CSV, Parquet, JSON, DuckDB tables, remote URLs)
and scales to larger-than-memory data with no special configuration. A thin
**Python** layer provides both a **CLI** and an importable **library** API.

## Goals

- Generic: no assumptions about the event domain.
- Format-agnostic input via DuckDB.
- Fast, larger-than-memory computation by pushing set/aggregation math into SQL.
- Explainable predictions: always surface the rules behind a probability.
- Configurable temporal semantics (same-window co-occurrence through
  multi-window forecasting).

## Non-Goals (YAGNI)

- No sliding/overlapping windows (tumbling only).
- No sequence/deep-learning models — this is association-rule based.
- No streaming/online updates; batch analysis over a static source.
- No GUI; CLI + library only.

## Inputs / Configuration

User provides:

- **Data source** — any DuckDB-readable path/URL/table.
- **`timestamp` column** name.
- **`event` column** name.
- Optional **filter** (WHERE clause) to scope rows.
- **`window`** size — duration string (e.g. `7d`, `1h`, `30m`).
- **`horizon`** H — integer windows ahead: `0` = same window (co-occurrence),
  `1` = next window, etc.
- **`max_antecedent_size`** N — largest itemset on a rule's left-hand side
  (default `2`).
- **`min_support`**, **`min_confidence`** — pruning thresholds.
- **`aggregation`** — `noisy_or` (default), `max`, or `best_lift`.

## Data Flow

The heavy math runs as SQL inside DuckDB.

1. **Load & normalize** — read the source into a relation `(ts, event)`,
   applying the optional filter. Validate that columns exist and the timestamp
   parses.
2. **Window assignment** — tumbling, non-overlapping windows:
   `window_id = floor((ts − anchor) / size)`, where `anchor` aligns windows
   (default: `min(ts)`). Produce distinct `(window_id, event)` baskets.
3. **Horizon labeling** — pair an antecedent observed in window `w` with
   consequents observed in window `w + H`. For `H = 0`, antecedent and
   consequent share the window (co-occurrence). This join is what makes the
   system predictive.
4. **Rule generation** — frequent itemsets of size `1..N` built via SQL
   self-joins, pruned early by `min_support`. For each rule
   `antecedent_itemset I → target T`, computed over windows:
   - `support  = count(I in w AND T in w+H) / total_windows`
   - `confidence = count(I in w AND T in w+H) / count(I in w)`
   - `lift = confidence / baserate(T)`, where
     `baserate(T) = P(T present in any window at the horizon)`
   - Keep rules passing `min_support` and `min_confidence` in a rules relation
     `(antecedent_items[], consequent, support, confidence, lift,
     antecedent_count)`.
5. **Predict** — let basket `B` = events in the most recent window. Match every
   rule whose antecedent is a subset of `B`. Aggregate matched rules per target
   into a single probability and rank.

## Aggregation

When basket `B` matches several rules pointing at the same target `T`, fold
them into one score:

- **`noisy_or`** (default): `P(T) = 1 − Π(1 − confidence_i)` over matched rules.
  Bounded `[0,1]`, rewards multiple independent supporting rules, intuitive.
- **`max`**: highest matching `confidence` — most conservative.
- **`best_lift`**: confidence of the single highest-lift matching rule.

Regardless of method, the contributing rules are retained for explainability.

## Output

- **Targeted** (`--target EventA`): the probability plus the ranked list of
  supporting rules.
- **Ranked** (no target): every candidate event ranked by predicted
  probability, each annotated with its top supporting rule.
- Rendered as a table to stdout, or written via DuckDB to CSV / Parquet / JSON.
- **No-match fallback**: if no rule matches the basket, fall back to base rates
  and state that explicitly.

## Components

Each unit has one purpose; SQL lives in templated query strings.

- **`loader.py`** — source spec → normalized `(ts, event)` relation; validates
  columns and timestamps.
- **`windowing.py`** — window assignment + horizon antecedent/consequent
  mapping.
- **`rules.py`** — frequent-itemset and rule-metric generation → rules relation.
- **`predict.py`** — subset-match basket against rules, aggregate, rank.
- **`aggregate.py`** — `noisy_or` / `max` / `best_lift` functions.
- **`config.py`** — parameters dataclass.
- **`cli.py`** — argument parsing and orchestration.

## Error Handling

- Validate columns exist and timestamps parse; clear errors otherwise.
- Guard against insufficient history (too few windows) with a warning.
- `min_support` prunes noise from rare itemsets.
- No-match prediction falls back to base rates rather than failing.

## Testing (TDD)

- Small hand-built synthetic event logs where support/confidence/lift are
  computable by hand.
- Window alignment and empty-window cases.
- Horizon offset correctness (`H = 0`, `H = 1`).
- Subset matching at prediction time.
- Each aggregation method (`noisy_or`, `max`, `best_lift`).
- No-match base-rate fallback.

## Open Questions

None blocking. Anchor alignment defaults to `min(ts)`; could later support a
fixed epoch anchor if cross-dataset window alignment is needed.
