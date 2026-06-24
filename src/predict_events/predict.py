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
