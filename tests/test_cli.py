from predict_events.cli import build_parser, config_from_args, format_result, main
from predict_events.predict import Prediction, SupportingRule
from predict_events.api import Result


def write_log(path):
    lines = ["when,kind"]
    for day in range(1, 4):
        lines.append(f"2024-01-0{day} 00:00:00,a")
        lines.append(f"2024-01-0{day} 02:00:00,b")
    lines.append("2024-01-04 00:00:00,a")
    path.write_text("\n".join(lines) + "\n")


def test_config_from_args_maps_flags():
    parser = build_parser()
    args = parser.parse_args([
        "--source", "x.csv", "--timestamp-col", "ts", "--event-col", "ev",
        "--window", "7d", "--horizon", "2", "--aggregation", "max",
        "--min-support", "0.1",
    ])
    cfg = config_from_args(args)
    assert cfg.source == "x.csv"
    assert cfg.horizon == 2
    assert cfg.aggregation == "max"
    assert cfg.min_support == 0.1


def test_format_targeted_mentions_probability_and_evidence():
    result = Result(
        basket=["a"],
        predictions=[Prediction(event="b", probability=0.75, n_rules=1, fallback=False)],
        supporting=[SupportingRule(antecedent=["a"], confidence=0.75, lift=1.0, support=0.6)],
    )
    text = format_result(result, target="b")
    assert "b" in text
    assert "75" in text  # probability rendered as percentage
    assert "a" in text   # supporting antecedent shown


def test_main_runs_end_to_end(tmp_path, capsys):
    csv = tmp_path / "log.csv"
    write_log(csv)
    code = main([
        "--source", str(csv), "--timestamp-col", "when", "--event-col", "kind",
        "--window", "1d", "--target", "b", "--max-antecedent-size", "1",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "b" in out
