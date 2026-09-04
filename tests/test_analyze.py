import pandas as pd

from analyze import run


def test_run_writes_all_three_outputs(tmp_path, data_dir):
    summary = run(data_dir, tmp_path)

    thresholds = pd.read_csv(tmp_path / "solar_thresholds.csv")
    windows = pd.read_csv(tmp_path / "solar_windows.csv")
    assert len(thresholds) == 366
    assert len(windows) == len(pd.date_range("2020-01-01", "2026-12-31"))
    assert (tmp_path / "report.html").read_text().count("<svg") == 5

    assert summary["n_days_observed"] == 2047
    assert 0 < summary["shortest_night_h"] < summary["longest_night_h"] < 24


def test_windows_csv_timestamps_are_unambiguous(tmp_path, data_dir):
    run(data_dir, tmp_path)
    w = pd.read_csv(tmp_path / "solar_windows.csv")
    assert w["solar_start_local"].str.contains(r"\+0[12]:00").all()
    assert w["solar_start_utc"].str.endswith("+00:00").all()
