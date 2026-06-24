import pytest

from predict_events.config import Config


def make(**over):
    base = dict(source="x.csv", timestamp_col="ts", event_col="ev", window="7d")
    base.update(over)
    return Config(**base)


def test_defaults_and_window_seconds():
    cfg = make()
    assert cfg.horizon == 0
    assert cfg.max_antecedent_size == 2
    assert cfg.aggregation == "max"
    assert cfg.window_seconds() == 7 * 86400


def test_rejects_bad_window():
    with pytest.raises(ValueError):
        make(window="nonsense")


def test_rejects_negative_horizon():
    with pytest.raises(ValueError):
        make(horizon=-1)


@pytest.mark.parametrize("size", [0, -5])
def test_rejects_small_antecedent_size(size):
    with pytest.raises(ValueError):
        make(max_antecedent_size=size)


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_support", 1.5),
        ("min_support", -0.1),
        ("min_confidence", 1.5),
        ("min_confidence", -0.1),
    ],
)
def test_rejects_out_of_range_thresholds(field, value):
    with pytest.raises(ValueError):
        make(**{field: value})


def test_rejects_unknown_aggregation():
    with pytest.raises(ValueError):
        make(aggregation="median")
