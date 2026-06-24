"""Match the current basket against rules and aggregate into probabilities."""

from dataclasses import dataclass

from predict_events.aggregate import aggregation_expr, top_rule_order
from predict_events.config import Config
from predict_events.windowing import WindowInfo


@dataclass
class SupportingRule:
    antecedent: list[str]
    confidence: float
    lift: float
    support: float
    support_count: int


@dataclass
class Prediction:
    event: str
    probability: float
    n_rules: int
    fallback: bool
    top_rule: "SupportingRule | None" = None


def current_basket(con, info: WindowInfo) -> list[str]:
    rows = con.execute(
        f"SELECT event FROM _pe_baskets WHERE window_id = {info.hi} ORDER BY event"
    ).fetchall()
    return [r[0] for r in rows]


def _create_matched_view(con, info: WindowInfo) -> None:
    """Rules whose antecedent is fully contained in the current basket."""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW _pe_basket AS
        SELECT event FROM _pe_baskets WHERE window_id = {info.hi}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW _pe_matched AS
        SELECT antecedent, consequent, support, confidence, lift, support_count
        FROM _pe_rules
        WHERE list_has_all((SELECT list(event) FROM _pe_basket), antecedent)
        """
    )


def predict_all(con, cfg: Config, info: WindowInfo) -> list[Prediction]:
    _create_matched_view(con, info)
    expr = aggregation_expr(cfg.aggregation)
    # When horizon=0 a present event trivially "predicts" itself at 100%;
    # exclude consequents already in the current basket so ranked output
    # shows only NEW likely events. For horizon>=1 a recurring event is a
    # legitimate prediction, so no exclusion.
    exclude = ""
    if cfg.horizon == 0:
        exclude = "WHERE consequent NOT IN (SELECT event FROM _pe_basket)"
    rows = con.execute(
        f"""
        SELECT consequent, {expr} AS probability, count(*) AS n_rules
        FROM _pe_matched
        {exclude}
        GROUP BY consequent
        ORDER BY probability DESC, consequent
        """
    ).fetchall()
    order = top_rule_order(cfg.aggregation)
    top_rows = con.execute(
        f"""
        SELECT consequent, antecedent, confidence, lift, support, support_count
        FROM _pe_matched
        {exclude}
        QUALIFY row_number() OVER (
            PARTITION BY consequent ORDER BY {order}
        ) = 1
        """
    ).fetchall()
    top = {
        c: SupportingRule(antecedent=a, confidence=cf, lift=l, support=s,
                          support_count=sc)
        for (c, a, cf, l, s, sc) in top_rows
    }
    return [
        Prediction(event=e, probability=p, n_rules=n, fallback=False,
                   top_rule=top.get(e))
        for (e, p, n) in rows
    ]


def predict_target(
    con, cfg: Config, info: WindowInfo, target: str
) -> tuple[Prediction, list[SupportingRule]]:
    _create_matched_view(con, info)
    expr = aggregation_expr(cfg.aggregation)
    # An aggregate query always returns exactly one row, so the no-match case
    # is signalled by a zero rule count, not a missing row.
    row = con.execute(
        f"""
        SELECT {expr} AS probability, count(*) AS n_rules
        FROM _pe_matched
        WHERE consequent = ?
        """,
        [target],
    ).fetchone()

    if row[1] == 0:
        base = con.execute(
            "SELECT baserate FROM _pe_baserates WHERE event = ?", [target]
        ).fetchone()
        probability = base[0] if base is not None else 0.0
        return (
            Prediction(event=target, probability=probability, n_rules=0, fallback=True),
            [],
        )

    probability, n_rules = row
    order = top_rule_order(cfg.aggregation)
    supporting_rows = con.execute(
        f"""
        SELECT antecedent, confidence, lift, support, support_count
        FROM _pe_matched
        WHERE consequent = ?
        ORDER BY {order}
        """,
        [target],
    ).fetchall()
    supporting = [
        SupportingRule(antecedent=a, confidence=c, lift=l, support=s,
                       support_count=sc)
        for (a, c, l, s, sc) in supporting_rows
    ]
    return (
        Prediction(event=target, probability=probability, n_rules=n_rules, fallback=False),
        supporting,
    )
