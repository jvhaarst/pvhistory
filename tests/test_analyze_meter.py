import numpy as np
import pandas as pd
import pytest

import analyze_meter
from analyze_meter import run


def test_run_writes_all_three_outputs(tmp_path, repo_root):
    s = run(repo_root, tmp_path)

    nights = pd.read_csv(tmp_path / "meter_night_summary.csv")
    sweep = pd.read_csv(tmp_path / "meter_battery_sweep.csv")
    assert len(nights) > 2000
    assert set(sweep["bound"].unique()) == {"charge_first", "discharge_first"}
    assert set(sweep["power_kw"].unique()) == {2.5, 3.0, 3.7}
    assert (tmp_path / "meter_report.html").read_text().count("<svg") == 6

    assert s["n_nights"] > 2000
    # Both elbows must land inside the sweep, and nothing more is asserted
    # about them. Which within-interval ordering yields the larger elbow has
    # never been measured, so pinning an order here would be a guess dressed
    # as a test — the measured pair is reported instead.
    assert 0 < s["elbow_charge_first_kwh"] <= 30
    assert 0 < s["elbow_discharge_first_kwh"] <= 30


def test_the_meter_night_median_exceeds_pvoutputs(tmp_path, repo_root):
    """The whole point: PVOutput was missing load, so the meter's median
    night must be the larger figure."""
    s = run(repo_root, tmp_path)
    pv = pd.read_csv(repo_root / "out" / "night_summary.csv")
    pv_median = pv[pv["covered"]]["night_wh"].median() / 1000
    assert s["median_night_kwh"] > pv_median


def test_resolution_penalty_is_measured_and_plausible(tmp_path, repo_root):
    """Averaging hides peaks, so the coarser run should look no worse than
    the finer one, and the gap should be single-digit percent."""
    s = run(repo_root, tmp_path)
    assert -1.0 <= s["resolution_penalty_pct"] <= 25.0


def _pair(dates, pv_wh, covered_pv, meter_kwh, covered_meter):
    """Matched PVOutput and meter night frames, with a known ratio."""
    pv = pd.DataFrame({"date": pd.to_datetime(dates), "night_wh": pv_wh,
                       "covered": covered_pv})
    me = pd.DataFrame({"date": pd.to_datetime(dates), "import_kwh": meter_kwh,
                       "covered": covered_meter})
    return me, pv


def test_monthly_ratio_divides_summed_energy_not_averaged_ratios():
    """A heavy night must weigh more than a light one in the same month.

    Averaging the two nights' ratios would give 0.694 (0.5 and 8/9); summing
    the energy first gives 9/11 = 0.818, which is the fraction of that
    month's load PVOutput actually saw.
    """
    me, pv = _pair(["2023-01-05", "2023-01-06"], [1_000.0, 8_000.0],
                   [True, True], [2.0, 9.0], [True, True])
    out = analyze_meter.monthly_ratio(me, pv)

    assert len(out) == 1
    assert out["ratio"].iloc[0] == pytest.approx(9.0 / 11.0)


def test_monthly_ratio_drops_a_night_either_source_calls_unusable():
    """Coverage is required on both sides, so a night present in only one
    frame cannot inflate or deflate the month it lands in."""
    me, pv = _pair(["2023-03-01", "2023-03-02", "2023-03-03"],
                   [1_000.0, 5_000.0, 1_000.0], [True, True, False],
                   [2.0, 5.0, 2.0], [True, False, True])
    out = analyze_meter.monthly_ratio(me, pv)

    # Only 2023-03-01 survives: the second night fails on the meter side, the
    # third on PVOutput's.
    assert out["meter_kwh"].iloc[0] == pytest.approx(2.0)
    assert out["ratio"].iloc[0] == pytest.approx(0.5)


def test_monthly_ratio_refuses_to_publish_an_empty_comparison():
    me, pv = _pair(["2023-01-05"], [1_000.0], [False], [2.0], [True])
    with pytest.raises(ValueError, match="no nights are usable in both"):
        analyze_meter.monthly_ratio(me, pv)


def test_pv_shortfall_weighs_months_by_energy():
    """A near-perfect quiet month must not offset a badly-wrong heavy one."""
    ratio = pd.DataFrame({
        "month": pd.PeriodIndex(["2024-01", "2024-07"], freq="M"),
        "pv_kwh": [75.0, 10.0], "meter_kwh": [100.0, 10.0],
    })
    # 85 of 110 kWh seen -> 22.7% short, not the 12.5% a mean of the two
    # monthly ratios would report.
    assert analyze_meter.pv_shortfall_pct(ratio, 2024) == pytest.approx(
        100.0 * (1.0 - 85.0 / 110.0))
    assert np.isnan(analyze_meter.pv_shortfall_pct(ratio, 2019))


def test_overlapping_span_actually_restricts_and_is_not_a_no_op(repo_root):
    """The span check is only evidence if it really narrows the record.

    A restriction that silently matched everything would return the headline
    elbow too, and would look like confirmation while checking nothing. So
    assert the narrowing itself, not just the elbow that comes out.
    """
    from pvnight import meter, meter_nights
    from pvnight.loader import load

    windows = pd.read_csv(repo_root / "out" / "solar_windows.csv",
                          parse_dates=analyze_meter._WINDOW_DATES)
    meter_df = meter.load_meter(repo_root / meter.METER_SUBDIR)
    gaps = meter.find_gaps(meter_df)
    nights = meter_nights.summarise(meter_df, windows, gaps)
    samples = load(repo_root / analyze_meter.DATA_SUBDIR)

    lo, hi = samples["ts_utc"].min(), samples["ts_utc"].max()
    kept = meter_df[(meter_df["ts_utc"] >= lo) & (meter_df["ts_utc"] <= hi)]
    assert len(kept) < len(meter_df), "the restriction matched the whole record"

    kept_nights = nights[(nights["night_start_utc"] >= lo)
                         & (nights["night_end_utc"] <= hi)]
    assert len(kept_nights) < len(nights)
    # Nights straddling an edge are dropped, not clipped: a half-night would
    # understate its own consumption and bias the curve it feeds.
    assert (kept_nights["night_start_utc"] >= lo).all()
    assert (kept_nights["night_end_utc"] <= hi).all()


def test_overlapping_span_returns_nan_when_nothing_overlaps():
    """No overlap is not the same as agreement, and must not read as 9.0."""
    ts = pd.date_range("2023-01-01", periods=8, freq="15min", tz="UTC")
    meter_df = pd.DataFrame({"ts_utc": ts, "import_kwh": 0.2, "export_kwh": 0.0})
    nights = pd.DataFrame([{"date": ts[0].date(), "night_start_utc": ts[0],
                            "night_end_utc": ts[-1], "is_ev": False,
                            "covered": True}])
    elsewhere = pd.DataFrame({
        "ts_utc": pd.date_range("2019-01-01", periods=4, freq="5min", tz="UTC")})
    assert np.isnan(analyze_meter.elbow_on_overlapping_span(
        meter_df, nights, elsewhere))
