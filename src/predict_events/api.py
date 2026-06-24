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


def analyze(
    cfg: Config,
    target: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> Result:
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
