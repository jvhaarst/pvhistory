"""The winter wall, measured at the meter.

Phase 2 found five months whose median daytime surplus was negative and only
48.2% of nights coverable with infinite storage, and called that the real
bound on any battery answer. Those figures came from the consumption channel
that was later measured faulty, so phase 3 declined to repeat them. These
tests cover re-deriving them from the meter.
"""

import numpy as np
import pandas as pd
import pytest

from pvnight.meter_wall import (
    coverable_fraction,
    daily_surplus,
    monthly_discharge,
    monthly_wall,
    negative_surplus_months,
)


def _windows(dates, day_hours=(9, 17), night_hours=(20, 28)):
    """A phase-1-shaped window table: a solar window and the night after it."""
    rows = []
    for d in dates:
        base = pd.Timestamp(d, tz="UTC")
        rows.append({
            "date": base.normalize().tz_localize(None),
            "solar_start_utc": base + pd.Timedelta(hours=day_hours[0]),
            "solar_end_utc": base + pd.Timedelta(hours=day_hours[1]),
            "night_start_utc": base + pd.Timedelta(hours=night_hours[0]),
            "night_end_utc": base + pd.Timedelta(hours=night_hours[1]),
        })
    return pd.DataFrame(rows)


def _meter(dates, day_export, day_import, night_import):
    """Quarter-hourly meter rows with a chosen day and night profile.

    One continuous timeline written once, then each date's own windows filled
    in. Building a frame per date and concatenating does not work: a night
    runs past midnight, so consecutive dates overlap and the later date's
    profile is dropped as a duplicate — which silently zeroed three of four
    days the first time this was written.
    """
    first = pd.Timestamp(min(dates), tz="UTC")
    last = pd.Timestamp(max(dates), tz="UTC") + pd.Timedelta(days=2)
    ts = pd.date_range(first, last, freq="15min", tz="UTC")
    imp = np.zeros(len(ts))
    exp = np.zeros(len(ts))

    for d, de, di, ni in zip(dates, day_export, day_import, night_import):
        base = pd.Timestamp(d, tz="UTC")
        h = (ts - base) / pd.Timedelta(hours=1)
        day = (h >= 9) & (h < 17)
        night = (h >= 20) & (h < 28)
        exp[day] = de / int(day.sum())
        imp[day] = di / int(day.sum())
        imp[night] = ni / int(night.sum())

    return pd.DataFrame({"ts_utc": ts, "import_kwh": imp, "export_kwh": exp})


def _nights(dates, night_import, covered=None):
    rows = []
    covered = [True] * len(dates) if covered is None else covered
    for d, ni, c in zip(dates, night_import, covered):
        base = pd.Timestamp(d, tz="UTC")
        rows.append({
            "date": base.normalize().tz_localize(None),
            "night_start_utc": base + pd.Timedelta(hours=20),
            "night_end_utc": base + pd.Timedelta(hours=28),
            "import_kwh": ni, "is_ev": False, "covered": c,
        })
    return pd.DataFrame(rows)


NO_GAPS = pd.DataFrame(columns=["gap_start_utc", "gap_end_utc",
                                "missing_intervals"])


def test_surplus_is_the_net_daytime_position_and_can_be_negative():
    """A winter day imports through its own solar window.

    Surplus is export minus import across the window, so a day that draws
    more than it sends reads negative — which is what phase 2's "negative
    surplus month" meant, and what makes the two phases comparable.
    """
    dates = ["2024-06-01", "2024-12-01"]
    m = _meter(dates, day_export=[20.0, 0.9], day_import=[1.0, 4.2],
               night_import=[4.0, 8.0])
    d = daily_surplus(m, _windows(dates), NO_GAPS)

    summer = d[d["date"] == pd.Timestamp("2024-06-01")].iloc[0]
    winter = d[d["date"] == pd.Timestamp("2024-12-01")].iloc[0]
    assert summer["surplus_kwh"] == pytest.approx(19.0)
    assert winter["surplus_kwh"] == pytest.approx(-3.3)
    assert winter["surplus_kwh"] < 0


def test_a_day_whose_solar_window_overlaps_a_gap_is_excluded():
    """An outage must not be read as a day with nothing to export.

    Left in, a gap looks exactly like a house that generated nothing, and it
    would fabricate a wall in the month that already dominates the answer.
    """
    dates = ["2024-01-08", "2024-01-09"]
    m = _meter(dates, day_export=[6.0, 6.0], day_import=[1.0, 1.0],
               night_import=[8.0, 8.0])
    gaps = pd.DataFrame([{
        "gap_start_utc": pd.Timestamp("2024-01-08 10:00", tz="UTC"),
        "gap_end_utc": pd.Timestamp("2024-01-08 14:00", tz="UTC"),
        "missing_intervals": 16,
    }])
    d = daily_surplus(m, _windows(dates), gaps)

    hit = d[d["date"] == pd.Timestamp("2024-01-08")].iloc[0]
    clean = d[d["date"] == pd.Timestamp("2024-01-09")].iloc[0]
    assert not hit["covered"], "a day overlapping a gap must not be usable"
    assert clean["covered"]


def test_coverable_fraction_pairs_a_day_with_the_night_that_follows_it():
    """The ceiling is same-day surplus against that night's need.

    Pairing a day with the wrong night would still produce a plausible
    percentage, so the test makes the two days disagree: the first day's
    surplus covers its own night and the second's does not.
    """
    dates = ["2024-05-01", "2024-05-02"]
    m = _meter(dates, day_export=[20.0, 3.0], day_import=[1.0, 1.0],
               night_import=[5.0, 9.0])
    d = daily_surplus(m, _windows(dates), NO_GAPS)
    n = _nights(dates, night_import=[5.0, 9.0])

    # day 1: surplus 19.0 >= need 5.0. day 2: surplus 2.0 < need 9.0.
    assert coverable_fraction(d, n) == pytest.approx(0.5)


def test_coverable_fraction_ignores_nights_the_meter_could_not_measure():
    dates = ["2024-05-01", "2024-05-02"]
    m = _meter(dates, day_export=[20.0, 3.0], day_import=[1.0, 1.0],
               night_import=[5.0, 9.0])
    d = daily_surplus(m, _windows(dates), NO_GAPS)
    n = _nights(dates, night_import=[5.0, 9.0], covered=[True, False])

    # Only the covered night remains, and its surplus covers it.
    assert coverable_fraction(d, n) == pytest.approx(1.0)


def test_coverable_fraction_refuses_an_empty_comparison():
    """Zero is a real answer here, so an empty join must not return it."""
    d = daily_surplus(_meter(["2024-05-01"], [5.0], [1.0], [4.0]),
                      _windows(["2024-05-01"]), NO_GAPS)
    n = _nights(["2019-01-01"], night_import=[4.0])
    with pytest.raises(ValueError, match="no dates in common"):
        coverable_fraction(d, n)


def test_monthly_wall_reports_medians_and_names_the_negative_months():
    dates = pd.date_range("2024-12-01", periods=4, freq="D").strftime("%Y-%m-%d")
    m = _meter(dates, day_export=[1.0, 1.0, 1.0, 1.0],
               day_import=[3.0, 3.0, 5.0, 5.0],
               night_import=[8.0, 8.0, 8.0, 8.0])
    d = daily_surplus(m, _windows(dates), NO_GAPS)
    n = _nights(dates, night_import=[8.0] * 4)
    w = monthly_wall(d, n)

    row = w[w["month"] == 12].iloc[0]
    assert row["surplus_kwh"] == pytest.approx(-3.0)
    assert row["night_kwh"] == pytest.approx(8.0)
    assert row["n_days"] == 4
    assert negative_surplus_months(w) == [12]


def test_negative_surplus_months_is_empty_when_no_month_is_negative():
    dates = ["2024-06-01", "2024-06-02"]
    m = _meter(dates, day_export=[20.0, 20.0], day_import=[1.0, 1.0],
               night_import=[4.0, 4.0])
    d = daily_surplus(m, _windows(dates), NO_GAPS)
    n = _nights(dates, night_import=[4.0, 4.0])
    assert negative_surplus_months(monthly_wall(d, n)) == []


def test_monthly_discharge_is_zero_in_a_month_with_nothing_to_charge_from():
    """The wall, expressed as the battery's own behaviour.

    A month that never exports cannot charge a battery, so its discharge must
    be zero however large the battery is — which is the claim the winter card
    makes and the reason capacity is not the binding constraint.
    """
    dates = pd.date_range("2024-12-01", periods=3, freq="D").strftime("%Y-%m-%d")
    m = _meter(dates, day_export=[0.0, 0.0, 0.0], day_import=[3.0, 3.0, 3.0],
               night_import=[8.0, 8.0, 8.0])
    n = _nights(dates, night_import=[8.0] * 3)
    out = monthly_discharge(m, n, capacity_kwh=9.0)

    assert out.loc[out["month"] == 12, "discharge_kwh"].iloc[0] == pytest.approx(0.0)


def test_monthly_discharge_is_positive_when_the_day_exports():
    dates = pd.date_range("2024-06-01", periods=3, freq="D").strftime("%Y-%m-%d")
    m = _meter(dates, day_export=[20.0, 20.0, 20.0], day_import=[0.0, 0.0, 0.0],
               night_import=[6.0, 6.0, 6.0])
    n = _nights(dates, night_import=[6.0] * 3)
    out = monthly_discharge(m, n, capacity_kwh=9.0)

    assert out.loc[out["month"] == 6, "discharge_kwh"].iloc[0] > 0.0
