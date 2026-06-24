from predict_events.api import analyze
from predict_events.config import Config


def write_log(path):
    # 'a' precedes/co-occurs with 'b' repeatedly; latest window holds 'a'.
    lines = ["when,kind"]
    for day in range(1, 4):  # days 1..3: both a and b
        lines.append(f"2024-01-0{day} 00:00:00,a")
        lines.append(f"2024-01-0{day} 02:00:00,b")
    lines.append("2024-01-04 00:00:00,a")  # latest window: {a}
    path.write_text("\n".join(lines) + "\n")


def test_analyze_ranked(tmp_path):
    csv = tmp_path / "log.csv"
    write_log(csv)
    cfg = Config(source=str(csv), timestamp_col="when", event_col="kind",
                 window="1d", max_antecedent_size=1, aggregation="max")
    result = analyze(cfg)
    assert result.basket == ["a"]
    events = {p.event for p in result.predictions}
    assert "b" in events
    assert result.supporting == []


def test_analyze_targeted(tmp_path):
    csv = tmp_path / "log.csv"
    write_log(csv)
    cfg = Config(source=str(csv), timestamp_col="when", event_col="kind",
                 window="1d", max_antecedent_size=1)
    result = analyze(cfg, target="b")
    assert len(result.predictions) == 1
    assert result.predictions[0].event == "b"
    assert result.predictions[0].probability > 0.0
    assert result.supporting  # targeted prediction with matching rules has evidence
    assert result.supporting[0].antecedent == ["a"]
