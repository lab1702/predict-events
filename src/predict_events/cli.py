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
