"""DuckDB aggregate expressions that fold matched rules into a probability."""

_EXPRESSIONS = {
    "noisy_or": "1 - product(1 - confidence)",
    "max": "max(confidence)",
    "best_lift": "arg_max(confidence, lift)",
}


def aggregation_expr(method: str) -> str:
    """Return the DuckDB aggregate expression for the given method."""
    try:
        return _EXPRESSIONS[method]
    except KeyError:
        raise ValueError(f"unknown aggregation method {method!r}") from None
