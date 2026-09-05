import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pvnight.meter import METER_SUBDIR, find_gaps, load_meter
from pvnight.meter_nights import ev_sensitivity, summarise


@pytest.fixture(scope="session")
def parts(repo_root):
    m = load_meter(repo_root / METER_SUBDIR)
    w = pd.read_csv(
        repo_root / "out" / "solar_windows.csv",
        parse_dates=["date", "night_start_utc", "night_end_utc"],
    )
    return m, w, find_gaps(m)


@pytest.fixture(scope="session")
def nights(parts):
    return summarise(*parts)


def _toy():
    ts = pd.date_range("2023-06-01T20:00", periods=12, freq="15min", tz="UTC")
    m = pd.DataFrame({"ts_utc": ts, "import_kwh": 0.25, "export_kwh": 0.0})
    w = pd.DataFrame([{
        "date": pd.Timestamp("2023-06-01"),
        "night_start_utc": ts[0], "night_end_utc": ts[-1],
    }])
    return m, w, pd.DataFrame(columns=["gap_start_utc", "gap_end_utc", "missing_intervals"])


def test_night_import_is_the_sum_over_the_window():
    m, w, g = _toy()
    out = summarise(m, w, g)
    assert out.loc[0, "import_kwh"] == pytest.approx(0.25 * 12)


def test_peak_kw_converts_from_interval_energy():
    """0.25 kWh in a quarter hour is a 1 kW mean."""
    m, w, g = _toy()
    out = summarise(m, w, g)
    assert out.loc[0, "peak_kw"] == pytest.approx(1.0)


def test_a_night_touching_a_gap_is_not_covered():
    m, w, _ = _toy()
    gaps = pd.DataFrame([{
        "gap_start_utc": pd.Timestamp("2023-06-01T21:00Z"),
        "gap_end_utc": pd.Timestamp("2023-06-01T22:00Z"),
        "missing_intervals": 3,
    }])
    out = summarise(m, w, gaps)
    assert out.loc[0, "missing_intervals"] == 3
    assert not bool(out.loc[0, "covered"])


def test_real_data_night_totals_agree_with_pvoutput_before_the_fault(nights, repo_root):
    """Spec §7 test 1. For 2020-2021 the two independent sources agree to
    within 3%. This is the evidence the night-extraction method is sound, and
    it must keep passing."""
    pv = pd.read_csv(repo_root / "out" / "night_summary.csv", parse_dates=["date"])
    pv = pv[pv["covered"]].set_index("date")["night_wh"] / 1000
    me = nights[nights["covered"]].set_index("date")["import_kwh"]
    j = pd.concat({"pv": pv, "me": me}, axis=1).dropna()
    j = j[j.index.year.isin([2020, 2021])]
    ratio = j["pv"].sum() / j["me"].sum()
    assert 0.97 <= ratio <= 1.03, f"2020-21 agreement drifted: {ratio:.3f}"


def test_real_data_shows_the_known_divergence_after_the_fault(nights, repo_root):
    """Spec §7 test 2. 2023/2024/2025 shortfalls measured at 24.9/27.6/30.8%.
    Assert 20-35% per year — wide enough that 24.9 is not on a cliff edge,
    narrow enough that the fault must still be there."""
    pv = pd.read_csv(repo_root / "out" / "night_summary.csv", parse_dates=["date"])
    pv = pv[pv["covered"]].set_index("date")["night_wh"] / 1000
    me = nights[nights["covered"]].set_index("date")["import_kwh"]
    j = pd.concat({"pv": pv, "me": me}, axis=1).dropna()
    for year in (2023, 2024, 2025):
        y = j[j.index.year == year]
        short = 100 * (1 - y["pv"].sum() / y["me"].sum())
        assert 20.0 <= short <= 35.0, f"{year} shortfall {short:.1f}% outside 20-35"


def test_the_december_2022_step_is_visible(nights, repo_root):
    """Spec §7 test 3. November 2022 >= 0.95, December 2022 <= 0.85."""
    pv = pd.read_csv(repo_root / "out" / "night_summary.csv", parse_dates=["date"])
    pv = pv[pv["covered"]].set_index("date")["night_wh"] / 1000
    me = nights[nights["covered"]].set_index("date")["import_kwh"]
    j = pd.concat({"pv": pv, "me": me}, axis=1).dropna()
    j = j[j["me"] > 0.5]
    nov = (j[j.index.to_period("M") == "2022-11"]["pv"]
           / j[j.index.to_period("M") == "2022-11"]["me"]).median()
    dec = (j[j.index.to_period("M") == "2022-12"]["pv"]
           / j[j.index.to_period("M") == "2022-12"]["me"]).median()
    assert nov >= 0.95
    assert dec <= 0.85


def test_january_2024_outage_removes_nights_rather_than_shrinking_them(nights):
    """Five gaps remove most of 8-19 January 2024. Those nights must be
    excluded, not counted as unusually quiet midwinter nights."""
    jan = nights[(nights["date"] >= "2024-01-08") & (nights["date"] <= "2024-01-19")]
    assert len(jan) > 0
    assert not jan["covered"].all(), "the outage nights should not all be covered"


def test_ev_sensitivity_grid_has_a_row_per_combination(parts):
    g = ev_sensitivity(*parts)
    assert len(g) == 12
    assert set(g.columns) == {"power_kw", "hours", "n_ev", "median_ev_kwh", "median_rest_kwh"}


def test_the_meter_sees_more_ev_nights_than_pvoutput_did(nights):
    """Phase 2 found 72 EV nights from a signal missing much of the car. The
    meter sees the whole load, so at the same threshold it should find at
    least as many."""
    assert int(nights.loc[nights["covered"], "is_ev"].sum()) >= 72
