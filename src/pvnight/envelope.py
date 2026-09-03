"""Fit the seasonal elevation threshold at which generation starts and stops."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import solar
from .config import N_HARMONICS, PERCENTILE, POOL_HALF_WIDTH_DAYS, SITE_TZ

DAYS_PER_YEAR = 365.25


def year_angle(ts: pd.DatetimeIndex) -> np.ndarray:
    """Position within the calendar year, in radians.

    Uses the fraction of the actual year elapsed rather than an integer day
    number, so leap years need no special case for 29 February.
    """
    ts = pd.DatetimeIndex(ts)
    year_start = pd.to_datetime(
        pd.Index(ts.year).astype(str) + "-01-01", utc=True
    )
    next_year = pd.to_datetime(
        pd.Index(ts.year + 1).astype(str) + "-01-01", utc=True
    )
    frac = (ts - year_start) / (next_year - year_start)
    return 2 * np.pi * np.asarray(frac, dtype=float)


def attach_solar(events: pd.DataFrame) -> pd.DataFrame:
    """Add solar elevation and year angle for both ends of each day."""
    ev = events.dropna(subset=["first_light_utc", "last_light_utc"]).copy()
    first = pd.DatetimeIndex(ev["first_light_utc"])
    last = pd.DatetimeIndex(ev["last_light_utc"])
    ev["el_first"] = solar.elevation(first).to_numpy()
    ev["el_last"] = solar.elevation(last).to_numpy()
    ev["phi_first"] = year_angle(first)
    ev["phi_last"] = year_angle(last)
    return ev


def pooled_percentile(
    phi_events: np.ndarray,
    values: np.ndarray,
    phi_targets: np.ndarray,
    half_width_days: int = POOL_HALF_WIDTH_DAYS,
    q: float = PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile of ``values`` pooled over a circular window in the year.

    The window wraps the year boundary, so 1 January pools with late
    December. Circular distance is taken via the complex argument, which
    handles the wrap without any modular arithmetic of our own.
    """
    half = 2 * np.pi * half_width_days / DAYS_PER_YEAR
    out = np.full(len(phi_targets), np.nan)
    counts = np.zeros(len(phi_targets), dtype=int)

    for i, target in enumerate(phi_targets):
        distance = np.abs(np.angle(np.exp(1j * (phi_events - target))))
        selected = distance <= half
        counts[i] = int(selected.sum())
        if counts[i]:
            out[i] = np.percentile(values[selected], q)
    return out, counts


def pooled_year_count(
    phi_events: np.ndarray,
    years: np.ndarray,
    phi_targets: np.ndarray,
    half_width_days: int = POOL_HALF_WIDTH_DAYS,
) -> np.ndarray:
    """Distinct calendar years contributing to each target.

    Spec section 7 requires this alongside the raw sample count: 120 samples
    drawn from one year is a very different claim from 120 drawn from six.
    """
    half = 2 * np.pi * half_width_days / DAYS_PER_YEAR
    out = np.zeros(len(phi_targets), dtype=int)
    for i, target in enumerate(phi_targets):
        distance = np.abs(np.angle(np.exp(1j * (phi_events - target))))
        out[i] = len(np.unique(years[distance <= half]))
    return out


def _design(phi: np.ndarray, n_harmonics: int) -> np.ndarray:
    columns = [np.ones_like(phi)]
    for k in range(1, n_harmonics + 1):
        columns.append(np.cos(k * phi))
        columns.append(np.sin(k * phi))
    return np.column_stack(columns)


def fit_fourier(
    phi: np.ndarray, values: np.ndarray, weights: np.ndarray, n_harmonics: int
) -> np.ndarray:
    """Weighted least-squares truncated Fourier fit. Periodic by construction."""
    finite = np.isfinite(values) & (weights > 0)
    design = _design(phi[finite], n_harmonics)
    root_w = np.sqrt(weights[finite])
    coef, *_ = np.linalg.lstsq(
        design * root_w[:, None], values[finite] * root_w, rcond=None
    )
    return coef


def eval_fourier(coef: np.ndarray, phi: np.ndarray, n_harmonics: int) -> np.ndarray:
    return _design(np.asarray(phi, dtype=float), n_harmonics) @ coef


@dataclass
class Envelope:
    """The fitted seasonal thresholds, plus the raw values behind them."""

    coef_start: np.ndarray
    coef_end: np.ndarray
    n_harmonics: int
    grid_phi: np.ndarray
    raw_start: np.ndarray
    raw_end: np.ndarray
    n_start: np.ndarray
    n_end: np.ndarray
    n_years_start: np.ndarray
    n_years_end: np.ndarray

    def theta_start(self, phi) -> np.ndarray:
        return eval_fourier(self.coef_start, np.atleast_1d(phi), self.n_harmonics)

    def theta_end(self, phi) -> np.ndarray:
        return eval_fourier(self.coef_end, np.atleast_1d(phi), self.n_harmonics)

    def _nearest(self, phi) -> int:
        return int(np.argmin(np.abs(np.angle(np.exp(1j * (self.grid_phi - phi))))))

    def samples_near(self, phi) -> tuple[int, int]:
        """Pooled sample counts at the nearest grid point to ``phi``."""
        i = self._nearest(phi)
        return int(self.n_start[i]), int(self.n_end[i])

    def years_near(self, phi) -> tuple[int, int]:
        """Distinct contributing years at the nearest grid point to ``phi``."""
        i = self._nearest(phi)
        return int(self.n_years_start[i]), int(self.n_years_end[i])


def fit(
    events_with_solar: pd.DataFrame,
    half_width_days: int = POOL_HALF_WIDTH_DAYS,
    q: float = PERCENTILE,
    n_harmonics: int = N_HARMONICS,
) -> Envelope:
    """Fit theta_start and theta_end over a 366-point year grid."""
    grid = 2 * np.pi * np.arange(366) / 366

    raw_start, n_start = pooled_percentile(
        events_with_solar["phi_first"].to_numpy(),
        events_with_solar["el_first"].to_numpy(),
        grid, half_width_days, q,
    )
    raw_end, n_end = pooled_percentile(
        events_with_solar["phi_last"].to_numpy(),
        events_with_solar["el_last"].to_numpy(),
        grid, half_width_days, q,
    )

    years = pd.to_datetime(events_with_solar["solar_date"]).dt.year.to_numpy()
    n_years_start = pooled_year_count(
        events_with_solar["phi_first"].to_numpy(), years, grid, half_width_days
    )
    n_years_end = pooled_year_count(
        events_with_solar["phi_last"].to_numpy(), years, grid, half_width_days
    )

    return Envelope(
        coef_start=fit_fourier(grid, raw_start, n_start.astype(float), n_harmonics),
        coef_end=fit_fourier(grid, raw_end, n_end.astype(float), n_harmonics),
        n_harmonics=n_harmonics,
        grid_phi=grid,
        raw_start=raw_start,
        raw_end=raw_end,
        n_start=n_start,
        n_end=n_end,
        n_years_start=n_years_start,
        n_years_end=n_years_end,
    )


def thresholds_table(model: Envelope) -> pd.DataFrame:
    """The model itself, one row per day of a leap reference year."""
    return pd.DataFrame(
        {
            "doy": np.arange(1, 367),
            "theta_start_deg": model.theta_start(model.grid_phi),
            "theta_end_deg": model.theta_end(model.grid_phi),
            "theta_start_raw_deg": model.raw_start,
            "theta_end_raw_deg": model.raw_end,
            "n_samples_start": model.n_start,
            "n_samples_end": model.n_end,
            "n_years": np.minimum(model.n_years_start, model.n_years_end),
        }
    )


def _local_iso(ts: pd.Timestamp) -> str | None:
    """ISO-8601 in site-local time with an explicit UTC offset."""
    if pd.isna(ts):
        return None
    return ts.tz_convert(SITE_TZ).isoformat()


def build_windows(
    model: Envelope,
    start_date: dt.date,
    end_date: dt.date,
    observed_dates: set[dt.date],
) -> pd.DataFrame:
    """Apply the fitted model to every date in the range."""
    dates = [d.date() for d in pd.date_range(start_date, end_date, freq="D")]
    rise_set = solar.sun_rise_set(dates)

    rows = []
    for d in dates:
        phi = year_angle(
            pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=12)])
        )[0]
        theta_start = float(model.theta_start(phi)[0])
        theta_end = float(model.theta_end(phi)[0])
        start, end = solar.crossings(d, theta_start, theta_end)
        n_start, n_end = model.samples_near(phi)
        years_start, years_end = model.years_near(phi)

        # A 25-hour local day is a fall-back day: the local clock repeats
        # an hour, and PVOutput stored only 288 slots for it, so one hour
        # of data is absent.
        grid_len = len(solar.day_grid_utc(d))

        rows.append(
            {
                "date": d,
                "solar_start_utc": start,
                "solar_end_utc": end,
                "solar_start_local": _local_iso(start),
                "solar_end_local": _local_iso(end),
                "sunrise_utc": rise_set.loc[d, "sunrise_utc"],
                "sunset_utc": rise_set.loc[d, "sunset_utc"],
                "theta_start_deg": theta_start,
                "theta_end_deg": theta_end,
                "n_samples": min(n_start, n_end),
                "n_years": min(years_start, years_end),
                "dst_hour_missing": grid_len == 1500,
                "extrapolated": d not in observed_dates,
                # Spec section 8: not expected at this latitude, but flagged
                # rather than assumed away.
                "window_undefined": pd.isna(start) or pd.isna(end),
            }
        )

    w = pd.DataFrame(rows)
    w["start_offset_min"] = (
        w["solar_start_utc"] - w["sunrise_utc"]
    ).dt.total_seconds() / 60
    w["end_offset_min"] = (
        w["solar_end_utc"] - w["sunset_utc"]
    ).dt.total_seconds() / 60
    w["night_start_utc"] = w["solar_end_utc"]
    w["night_end_utc"] = w["solar_start_utc"].shift(-1)
    w["night_duration_h"] = (
        w["night_end_utc"] - w["night_start_utc"]
    ).dt.total_seconds() / 3600
    return w
