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
