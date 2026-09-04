import datetime as dt

import numpy as np
import pandas as pd

from pvnight import solar
from pvnight.config import SUNRISE_ELEVATION_DEG


def test_day_grid_length_reflects_true_local_day_length():
    """Built in UTC, the grid spans the real day: 23h, 24h or 25h."""
    assert len(solar.day_grid_utc(dt.date(2023, 3, 26))) == 1380
    assert len(solar.day_grid_utc(dt.date(2023, 6, 15))) == 1440
    assert len(solar.day_grid_utc(dt.date(2023, 10, 29))) == 1500


def test_day_grid_is_utc_and_strictly_increasing():
    grid = solar.day_grid_utc(dt.date(2023, 10, 29))
    assert str(grid.tz) == "UTC"
    assert grid.is_monotonic_increasing
    assert not grid.duplicated().any()


def test_sun_rise_set_returns_the_requested_day_not_the_previous_one():
    """Regression guard for spec fact 10: pvlib given local midnight returns
    the previous day's sunrise. sun_rise_set must pass local noon."""
    dates = [dt.date(2023, 6, 21), dt.date(2023, 12, 21)]
    rs = solar.sun_rise_set(dates)
    for d in dates:
        assert rs.loc[d, "sunrise_utc"].tz_convert("Europe/Amsterdam").date() == d
        assert rs.loc[d, "sunset_utc"].tz_convert("Europe/Amsterdam").date() == d


def test_sunrise_is_before_sunset():
    rs = solar.sun_rise_set([dt.date(2023, 1, 15), dt.date(2023, 7, 15)])
    assert (rs["sunrise_utc"] < rs["sunset_utc"]).all()


def test_crossings_agree_with_pvlib_sunrise_sunset():
    """Cross-validation: the elevation-crossing solver and pvlib's SPA
    rise/set routine are independent code paths and must give one answer.

    The grid has one-minute resolution, so the first minute at or above the
    threshold normally falls in [sunrise, sunrise + 60s). The small negative
    tolerance covers a sub-second disagreement between two different pvlib
    algorithms: sun_rise_set_transit_spa uses SPA section 3.8, which is
    deliberately reduced-precision, while get_solarposition uses the full
    algorithm. Measured across all 365 days of 2023, the worst excursion
    below zero is -0.182s; -1s gives five-fold margin while still failing
    any real solver bug, which would show tens of seconds.
    """
    dates = sorted({dt.date(2023, 1, 1) + dt.timedelta(days=n)
                    for n in range(0, 365, 7)}
                   | {dt.date(2023, 3, 26), dt.date(2023, 10, 29),
                      dt.date(2023, 6, 21), dt.date(2023, 12, 21)})
    rs = solar.sun_rise_set(dates)

    for d in dates:
        start, end = solar.crossings(d, SUNRISE_ELEVATION_DEG, SUNRISE_ELEVATION_DEG)
        assert start is not pd.NaT and end is not pd.NaT
        lead = (start - rs.loc[d, "sunrise_utc"]).total_seconds()
        lag = (rs.loc[d, "sunset_utc"] - end).total_seconds()
        assert -1 <= lead < 60, f"{d}: start {lead}s from sunrise"
        assert -1 <= lag < 60, f"{d}: end {lag}s from sunset"


def test_crossings_return_nat_when_threshold_is_never_reached():
    start, end = solar.crossings(dt.date(2023, 12, 21), 80.0, 80.0)
    assert start is pd.NaT and end is pd.NaT


def test_elevation_is_geometric_not_apparent():
    """Below the horizon pvlib applies no refraction, so the two agree; above
    it they diverge. This pins that we return the geometric column."""
    import pvlib

    from pvnight.config import LATITUDE, LONGITUDE

    idx = pd.DatetimeIndex(["2023-06-21T04:00", "2023-06-21T10:00"], tz="UTC")
    sp = pvlib.solarposition.get_solarposition(idx, LATITUDE, LONGITUDE)
    got = solar.elevation(idx)
    assert np.allclose(got.to_numpy(), sp["elevation"].to_numpy())
    assert not np.allclose(got.to_numpy(), sp["apparent_elevation"].to_numpy())


def test_elevation_peaks_around_solar_noon():
    grid = solar.day_grid_utc(dt.date(2023, 6, 21))
    el = solar.elevation(grid).to_numpy()
    peak = grid[int(np.argmax(el))].tz_convert("Europe/Amsterdam")
    assert 13 <= peak.hour <= 14
