"""Command-line interface for predict-events."""

import argparse
import sys

import duckdb

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
    p.add_argument("--aggregation", choices=VALID_AGGREGATIONS, default="max")
    p.add_argument("--where", default=None)
    p.add_argument("--target", default=None, help="predict a specific event")
    p.add_argument("--top", type=int, default=20, help="rows to show in ranked mode")
    p.add_argument("--output", default=None,
                   help="write predictions to a file (.csv/.parquet/.json; "
                        "format inferred from extension)")
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


def write_output(con, predictions, path):
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _out("
        "event VARCHAR, probability DOUBLE, n_rules BIGINT)"
    )
    con.executemany(
        "INSERT INTO _out VALUES (?, ?, ?)",
        [(p.event, p.probability, p.n_rules) for p in predictions],
    )
    con.execute(f"COPY _out TO '{path}'")


def format_result(result: Result, target: str | None,
                  aggregation: str = "max") -> str:
    lines = [
        f"Current basket: {{{', '.join(result.basket) or '(empty)'}}}",
        f"Model: {result.n_windows} window(s), aggregation={aggregation}",
        "",
    ]
    if target is not None:
        pred = result.predictions[0]
        note = " (no matching rules; base rate)" if pred.fallback else ""
        lines.append(f"{pred.event}: {_pct(pred.probability)}{note}")
        if result.supporting:
            lines.append("because:")
            for r in result.supporting:
                lines.append(
                    f"  {{{', '.join(r.antecedent)}}} -> {pred.event}  "
                    f"conf={_pct(r.confidence)} lift={r.lift:.2f} "
                    f"(n={r.support_count})"
                )
    else:
        lines.append(f"{'event':<24}{'probability':>12}{'rules':>7}  evidence")
        for pred in result.predictions:
            if pred.top_rule is not None:
                ev = (f"{{{', '.join(pred.top_rule.antecedent)}}} "
                      f"(conf {_pct(pred.top_rule.confidence)}, "
                      f"n={pred.top_rule.support_count})")
            else:
                ev = ""
            lines.append(
                f"{pred.event:<24}{_pct(pred.probability):>12}"
                f"{pred.n_rules:>7}  {ev}"
            )
    if aggregation == "noisy_or":
        lines.append("")
        lines.append(
            "note: noisy_or is an uncalibrated heuristic score "
            "(it overestimates when rules overlap), not a probability."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
        con = duckdb.connect()
        result = analyze(cfg, target=args.target, con=con)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.target is None:
        result.predictions = result.predictions[: args.top]
    if args.output:
        write_output(con, result.predictions, args.output)
    print(format_result(result, args.target, args.aggregation))
    return 0
