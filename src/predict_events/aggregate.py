"""Fold the rules matching the current basket into one score per consequent.

`max`, `most_specific`, and `best_lift` each report the confidence of a single
rule, so the result is a genuine conditional probability P(consequent | that
rule's antecedent):

- ``max``           - highest-confidence matching rule (the default).
- ``most_specific`` - confidence of the rule with the largest antecedent, i.e.
                      the one conditioned on the most of the current basket.
- ``best_lift``     - confidence of the rule with the highest lift.

``noisy_or`` instead combines every matching rule as if they were independent
evidence (``1 - prod(1 - confidence)``). Matching rules are usually nested or
overlapping subsets of the same basket, so that independence assumption is
violated and ``noisy_or`` systematically OVERESTIMATES. Treat its output as an
uncalibrated ranking score, not a probability.

The single-rule expressions order ties with a homogeneous DOUBLE list so the
chosen confidence is deterministic.
"""

_EXPRESSIONS = {
    "max": "max(confidence)",
    "most_specific": (
        "arg_max(confidence, [len(antecedent)::DOUBLE, confidence, lift])"
    ),
    "best_lift": "arg_max(confidence, [lift, confidence, len(antecedent)::DOUBLE])",
    "noisy_or": "1 - product(1 - confidence)",
}

# ORDER BY clause selecting the single rule that best represents a consequent's
# score, so the displayed "top rule" is the one that actually produced the
# probability (for noisy_or there is no single rule, so show the highest-lift
# one as a representative).
_TOP_RULE_ORDER = {
    "max": "confidence DESC, lift DESC, len(antecedent) DESC, antecedent",
    "most_specific": "len(antecedent) DESC, confidence DESC, lift DESC, antecedent",
    "best_lift": "lift DESC, confidence DESC, len(antecedent) DESC, antecedent",
    "noisy_or": "lift DESC, confidence DESC, len(antecedent) DESC, antecedent",
}


def aggregation_expr(method: str) -> str:
    """Return the DuckDB aggregate expression for the given method."""
    try:
        return _EXPRESSIONS[method]
    except KeyError:
        raise ValueError(f"unknown aggregation method {method!r}") from None


def top_rule_order(method: str) -> str:
    """Return the ORDER BY clause picking a consequent's representative rule."""
    try:
        return _TOP_RULE_ORDER[method]
    except KeyError:
        raise ValueError(f"unknown aggregation method {method!r}") from None
