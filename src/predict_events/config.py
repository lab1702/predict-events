"""Configuration for an event-prediction run."""

from dataclasses import dataclass

from predict_events.duration import parse_duration

VALID_AGGREGATIONS = ("max", "most_specific", "best_lift", "noisy_or")


@dataclass
class Config:
    source: str
    timestamp_col: str
    event_col: str
    window: str
    horizon: int = 0
    max_antecedent_size: int = 2
    min_support: float = 0.0
    min_confidence: float = 0.0
    aggregation: str = "max"
    where: str | None = None

    def __post_init__(self) -> None:
        parse_duration(self.window)  # raises ValueError if malformed
        if self.horizon < 0:
            raise ValueError("horizon must be >= 0")
        if self.max_antecedent_size < 1:
            raise ValueError("max_antecedent_size must be >= 1")
        for name in ("min_support", "min_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {VALID_AGGREGATIONS}"
            )

    def window_seconds(self) -> int:
        return parse_duration(self.window)
