"""Parse human duration strings like '7d' into seconds."""

import re

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_PATTERN = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$")


def parse_duration(text: str) -> int:
    """Return the number of seconds for a duration string like '7d' or '30m'."""
    match = _PATTERN.match(text)
    if match is None:
        raise ValueError(
            f"invalid duration {text!r}; expected <int><unit> with unit in s,m,h,d,w"
        )
    value, unit = match.groups()
    return int(value) * _UNITS[unit]
