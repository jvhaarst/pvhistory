import pandas as pd
import pytest

from analyze_night import run


def test_run_writes_all_three_outputs(tmp_path, data_dir):
    s = run(data_dir, tmp_path)

    summary = pd.read_csv(tmp_path / "night_summary.csv")
    sweep = pd.read_csv(tmp_path / "battery_sweep.csv")
    assert len(summary) > 2000
    assert set(sweep.power_kw.unique()) == {2.5, 3.0, 3.7}
    assert (tmp_path / "night_report.html").read_text().count("<svg") == 8

    assert s["n_ev_nights"] == 72
    assert s["median_night_kwh"] == pytest.approx(4.85, abs=0.05)
    assert 0 < s["recommended_kwh"] <= 30


def test_power_cap_sensitivity_is_measured_not_assumed(tmp_path, data_dir):
    """Spec fact 9 says the 3 kW charge cap is *expected* not to bind, and
    that the simulation must confirm it rather than the spec asserting it.
    This pins that the figure is actually computed and is a sane percentage;
    the report states the value, whatever it turns out to be."""
    s = run(data_dir, tmp_path)
    gain = s["pct_gain_from_3p7kw_inverter"]
    assert gain == gain          # not NaN
    assert 0.0 <= gain < 100.0


def test_summary_csv_carries_the_ev_flag_and_coverage(tmp_path, data_dir):
    run(data_dir, tmp_path)
    d = pd.read_csv(tmp_path / "night_summary.csv")
    for col in ["date", "night_wh", "peak_w", "hours_above_2kw", "is_ev", "coverage"]:
        assert col in d.columns
    assert d["is_ev"].sum() == 72
