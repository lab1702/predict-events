import pytest

from predict_events.duration import parse_duration


def test_parses_each_unit():
    assert parse_duration("30s") == 30
    assert parse_duration("5m") == 300
    assert parse_duration("2h") == 7200
    assert parse_duration("7d") == 7 * 86400
    assert parse_duration("1w") == 604800


def test_allows_internal_whitespace():
    assert parse_duration(" 7 d ") == 7 * 86400


@pytest.mark.parametrize("bad", ["", "7", "d", "7x", "-3d", "1.5h"])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)
