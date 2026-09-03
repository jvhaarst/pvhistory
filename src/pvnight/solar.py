"""All contact with pvlib lives here. Geometric elevation throughout."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pvlib

from .config import LATITUDE, LONGITUDE, SITE_TZ


def _position(ts_utc: pd.DatetimeIndex) -> pd.DataFrame:
    return pvlib.solarposition.get_solarposition(ts_utc, LATITUDE, LONGITUDE)


def elevation(ts_utc: pd.DatetimeIndex) -> pd.Series:
    """Geometric solar elevation in degrees.

    Deliberately not ``apparent_elevation``: pvlib applies no refraction
    correction below the horizon, which puts a kink in the apparent curve at
    0 degrees, and a third of our events sit below it (spec fact 11).
    """
    return _position(pd.DatetimeIndex(ts_utc))["elevation"]


def azimuth(ts_utc: pd.DatetimeIndex) -> pd.Series:
    """Solar azimuth in degrees clockwise from north."""
    return _position(pd.DatetimeIndex(ts_utc))["azimuth"]


def sun_rise_set(dates: Sequence[dt.date]) -> pd.DataFrame:
    """Sunrise and sunset in UTC for each local date.

    pvlib is given local *noon*. Given local midnight it converts to UTC
    first, lands on the previous UTC day, and returns the previous day's
    sunrise — a silent one-day shift (spec fact 10).

    Duplicate dates in the input are collapsed (order-preserving): the
    result is indexed by date, and a date can only appear once in an index
    without ``.loc`` lookups degrading from a scalar to a Series.
    """
    dates = list(dict.fromkeys(dates))
    noon = pd.DatetimeIndex(
        [pd.Timestamp(d) + pd.Timedelta(hours=12) for d in dates]
    ).tz_localize(SITE_TZ)
    rst = pvlib.solarposition.sun_rise_set_transit_spa(noon, LATITUDE, LONGITUDE)
    return pd.DataFrame(
        {
            "sunrise_utc": pd.DatetimeIndex(rst["sunrise"]).tz_convert("UTC"),
            "sunset_utc": pd.DatetimeIndex(rst["sunset"]).tz_convert("UTC"),
        },
        index=pd.Index(dates, name="date"),
    )


def day_grid_utc(local_date: dt.date, freq: str = "1min") -> pd.DatetimeIndex:
    """UTC instants spanning one local day.

    Built in UTC so DST transitions simply make the day shorter or longer
    (1380 / 1440 / 1500 minutes) instead of producing a gap or duplicates.
    """
    start = pd.Timestamp(local_date).tz_localize(SITE_TZ)
    end = (pd.Timestamp(local_date) + pd.Timedelta(days=1)).tz_localize(SITE_TZ)
    return pd.date_range(
        start.tz_convert("UTC"), end.tz_convert("UTC"),
        freq=freq, inclusive="left", tz="UTC",
    )


def crossings(
    local_date: dt.date, theta_start_deg: float, theta_end_deg: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last minute of the local day at or above the thresholds.

    Elevation rises to a single midday peak and falls, so the first and last
    samples above a threshold are exactly the two crossings.
    """
    grid = day_grid_utc(local_date)
    el = elevation(grid).to_numpy()

    above_start = el >= theta_start_deg
    start = grid[int(np.argmax(above_start))] if above_start.any() else pd.NaT

    above_end = el >= theta_end_deg
    end = (
        grid[len(grid) - 1 - int(np.argmax(above_end[::-1]))]
        if above_end.any()
        else pd.NaT
    )
    return start, end
