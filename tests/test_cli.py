import duckdb

from predict_events.cli import build_parser, config_from_args, format_result, main, write_output
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
        "--min-support", "0.1", "--min-confidence", "0.2",
        "--max-antecedent-size", "3", "--where", "ev <> 'noise'",
    ])
    cfg = config_from_args(args)
    assert cfg.source == "x.csv"
    assert cfg.timestamp_col == "ts"
    assert cfg.event_col == "ev"
    assert cfg.window == "7d"
    assert cfg.horizon == 2
    assert cfg.aggregation == "max"
    assert cfg.min_support == 0.1
    assert cfg.min_confidence == 0.2
    assert cfg.max_antecedent_size == 3
    assert cfg.where == "ev <> 'noise'"


def test_format_targeted_mentions_probability_and_evidence():
    result = Result(
        basket=["a"],
        predictions=[Prediction(event="b", probability=0.75, n_rules=1, fallback=False)],
        supporting=[SupportingRule(antecedent=["a"], confidence=0.75, lift=1.0,
                                   support=0.6, support_count=3)],
        n_windows=5,
    )
    text = format_result(result, target="b")
    assert "b" in text
    assert "75" in text  # probability rendered as percentage
    assert "a" in text   # supporting antecedent shown
    assert "n=3" in text  # support count surfaced for reliability


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


def test_format_ranked_shows_table():
    result = Result(
        basket=["a"],
        predictions=[
            Prediction(event="b", probability=0.75, n_rules=2, fallback=False),
            Prediction(event="c", probability=0.40, n_rules=1, fallback=False),
        ],
        supporting=[],
        n_windows=5,
    )
    text = format_result(result, target=None)
    assert "event" in text        # header label
    assert "probability" in text  # header label
    assert "b" in text and "c" in text
    assert "75.0%" in text
    assert "40.0%" in text


def test_format_ranked_shows_evidence():
    result = Result(
        basket=["a"],
        predictions=[
            Prediction(event="b", probability=0.75, n_rules=2, fallback=False,
                       top_rule=SupportingRule(antecedent=["a"], confidence=0.6,
                                               lift=1.2, support=0.5,
                                               support_count=4)),
        ],
        supporting=[],
        n_windows=5,
    )
    text = format_result(result, target=None)
    assert "evidence" in text
    assert "a" in text
    assert "60.0%" in text  # top rule confidence rendered
    assert "n=4" in text    # support count surfaced


def test_main_writes_output_file(tmp_path):
    log = tmp_path / "log.csv"
    write_log(log)
    out = tmp_path / "preds.csv"
    code = main([
        "--source", str(log), "--timestamp-col", "when", "--event-col", "kind",
        "--window", "1d", "--max-antecedent-size", "1", "--output", str(out),
    ])
    assert code == 0
    assert out.exists()
    con = duckdb.connect()
    n = con.execute(f"SELECT count(*) FROM read_csv_auto('{out.as_posix()}')").fetchone()[0]
    assert n >= 1  # at least one predicted event written


def test_main_returns_1_on_bad_input(capsys):
    code = main([
        "--source", "nope.csv", "--timestamp-col", "t", "--event-col", "e",
        "--window", "not-a-duration",
    ])
    assert code == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_format_flags_noisy_or_as_heuristic():
    result = Result(
        basket=["a"],
        predictions=[Prediction(event="b", probability=0.9, n_rules=3, fallback=False)],
        supporting=[SupportingRule(antecedent=["a"], confidence=0.6, lift=1.1,
                                   support=0.5, support_count=3)],
        n_windows=5,
    )
    text = format_result(result, target="b", aggregation="noisy_or")
    assert "noisy_or" in text
    assert "not a probability" in text


def test_format_reports_training_window_count():
    result = Result(basket=["a"], predictions=[], supporting=[], n_windows=42)
    text = format_result(result, target=None, aggregation="max")
    assert "42" in text  # model training size surfaced
