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
    assert cfg.aggregation == "noisy_or"
    assert cfg.window_seconds() == 7 * 86400


def test_rejects_bad_window():
    with pytest.raises(ValueError):
        make(window="nonsense")


def test_rejects_negative_horizon():
    with pytest.raises(ValueError):
        make(horizon=-1)


def test_rejects_small_antecedent_size():
    with pytest.raises(ValueError):
        make(max_antecedent_size=0)


@pytest.mark.parametrize("field", ["min_support", "min_confidence"])
def test_rejects_out_of_range_thresholds(field):
    with pytest.raises(ValueError):
        make(**{field: 1.5})


def test_rejects_unknown_aggregation():
    with pytest.raises(ValueError):
        make(aggregation="median")
