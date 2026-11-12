from engine_health_flow.cli import main
from engine_health_flow.data import HEALTH_SHORT
from engine_health_flow.report import indicator_table, load, main_table, paired_table


def fake_model(rmse, posterior=True):
    test = {"rmse": [rmse] * 10, "pooled": {"rmse": rmse, "smape": 0.1, "pearson": 0.98}}
    if posterior:
        scores = {
            "crps_mean": 0.0004,
            "crps": [0.0004] * 10,
            "intervals": {"0.9": {"coverage": [0.85] * 10, "width": [0.002] * 10}},
        }
        test["posterior"] = test["posterior_rescaled"] = scores
    return {"fit_seconds": 3.0, "test": test}


def fake_result(ridge, flow):
    models = {"ridge": fake_model(ridge, posterior=False), "flow": fake_model(flow)}
    return {"models": models}


def rows_of(table):
    return {line.split("|")[1].strip(): line for line in table.splitlines()[2:]}


def test_main_table_averages_over_folds_and_marks_missing_scores():
    table = main_table([fake_result(0.001, 0.002), fake_result(0.002, 0.002)])
    rows = rows_of(table)
    assert "1.50 ± 0.71" in rows["ridge"]  # rmse x1e3 over two folds
    assert "2.00 ± 0.00" in rows["flow"]
    ridge_cells = [c.strip() for c in rows["ridge"].strip("|").split("|")]
    flow_cells = [c.strip() for c in rows["flow"].strip("|").split("|")]
    assert ridge_cells[4:8] == ["-", "-", "-", "-"]  # no posterior, no CRPS or coverage
    assert flow_cells[4] == "0.400 ± 0.000" and flow_cells[5] == "0.85 ± 0.00"


def test_main_table_of_one_fold_has_no_spread():
    table = main_table([fake_result(0.001, 0.002)])
    assert "±" not in table
    assert table.splitlines()[0].startswith("| model | RMSE x1e3 |")


def test_indicator_table_has_one_column_per_indicator():
    lines = indicator_table([fake_result(0.001, 0.002)]).splitlines()
    assert lines[0].split("|")[2:-1] == [f" {name} " for name in HEALTH_SHORT]
    assert lines[2].split("|")[2:4] == [" 1.00 ", " 1.00 "]


def test_paired_table_counts_the_folds_a_model_wins():
    # per fold the flow beats the ridge by 0.5, 0.1 and loses by 0.2 (x1e-3)
    folds = [fake_result(0.0020, 0.0015), fake_result(0.0013, 0.0012), fake_result(0.0010, 0.0012)]
    table = paired_table(folds, pairs=(("flow", "ridge"),))
    row = table.splitlines()[2]
    assert row.split("|")[1].strip() == "flow - ridge"
    assert "-0.13 (lower in 2 of 3)" in row
    # the ridge has no CRPS, so there is nothing to compare there
    assert row.split("|")[3].strip() == "-"


def test_paired_table_skips_models_that_were_not_run():
    table = paired_table([fake_result(0.001, 0.002)], pairs=(("hybrid", "ukf"), ("flow", "ridge")))
    assert [line.split("|")[1].strip() for line in table.splitlines()[2:]] == ["flow - ridge"]


def test_table_command_reads_the_files_that_run_wrote(synthetic_csv, tmp_path, capsys):
    out = tmp_path / "fold0.json"
    main(["run", "--csv", str(synthetic_csv), "--models", "ridge", "--out", str(out)])
    capsys.readouterr()
    main(["table", str(out)])
    printed = capsys.readouterr().out
    assert "| ridge |" in printed
    assert len(load([out])) == 1
