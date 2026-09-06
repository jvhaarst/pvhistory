"""The winter wall, measured at the meter.

Phase 2 established that capacity is not what bounds this answer: in five
months the median daytime surplus was negative, so no battery of any size
received a charge, and across six years only 48.2% of nights could have been
covered even with infinite storage. Those figures came from PVOutput's
consumption channel, which was later measured to have lost roughly a quarter
of the household load from December 2022, so phase 3 declined to repeat them
and left the wall unmeasured.

This module re-derives it from the meter. The meter sees only the grid
connection, so daytime surplus is the **net daytime position** — export minus
import summed across the phase-1 solar window. That is the same quantity
phase 2 computed as generation minus daytime consumption, since a house's
generation minus its load *is* what crosses the meter, and it goes negative in
exactly the months phase 2 called negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import BatterySpec, simulate
from .meter import DT_HOURS
from .meter_battery import _masks, bound_signals

# The wall is a ceiling, so it is drawn with the most optimistic battery that
# could exist: infinite capacity, no inverter limit, no round-trip loss. A
# realistic battery does worse. That is the point — a bound nothing can beat.
REPORT_POWER_KW = 3.0


def _window_totals(meter_df: pd.DataFrame, starts, ends) -> tuple:
    """Import and export summed between each pair of timestamps."""
    m = meter_df.sort_values("ts_utc")
    ts = m["ts_utc"].to_numpy()
    imp = np.concatenate([[0.0], np.cumsum(m["import_kwh"].to_numpy())])
    exp = np.concatenate([[0.0], np.cumsum(m["export_kwh"].to_numpy())])
    i = np.searchsorted(ts, np.asarray(starts), "left")
    j = np.searchsorted(ts, np.asarray(ends), "right")
    return imp[j] - imp[i], exp[j] - exp[i]


def _overlaps_a_gap(starts, ends, gaps: pd.DataFrame) -> np.ndarray:
    """Whether each window contains a missing meter interval.

    A day whose solar window overlaps an outage looks exactly like a day that
    generated nothing, and would fabricate a wall in whatever month the outage
    fell in. Nights are already excluded on this rule; days must be too.
    """
    out = np.zeros(len(starts), dtype=bool)
    if gaps.empty:
        return out
    gs = gaps["gap_start_utc"].to_numpy()
    ge = gaps["gap_end_utc"].to_numpy()
    for k, (a, b) in enumerate(zip(starts, ends)):
        out[k] = bool(((gs < b) & (ge > a)).any())
    return out


def daily_surplus(meter_df: pd.DataFrame, windows: pd.DataFrame,
                  gaps: pd.DataFrame) -> pd.DataFrame:
    """Per date: what the daytime left over, measured at the meter.

    ``surplus_kwh`` is export minus import across the solar window, so it is
    negative on a day the house drew more than it sent — a winter day whose
    own daytime load exceeds its generation. ``covered`` is False when the
    window overlaps a meter gap.
    """
    w = windows.dropna(subset=["solar_start_utc", "solar_end_utc"]).copy()
    starts = w["solar_start_utc"].to_numpy()
    ends = w["solar_end_utc"].to_numpy()
    imp, exp = _window_totals(meter_df, starts, ends)

    lo = meter_df["ts_utc"].min()
    hi = meter_df["ts_utc"].max()
    in_span = (w["solar_start_utc"] >= lo) & (w["solar_end_utc"] <= hi)

    return pd.DataFrame({
        "date": pd.to_datetime(w["date"].values),
        "day_import_kwh": imp,
        "day_export_kwh": exp,
        "surplus_kwh": exp - imp,
        "covered": in_span.to_numpy() & ~_overlaps_a_gap(starts, ends, gaps),
    })


def _paired(daily: pd.DataFrame, nights_df: pd.DataFrame) -> pd.DataFrame:
    """Each usable day beside the night that follows it.

    Pairing on ``date`` is what makes the ceiling comparable to phase 2's:
    a day's surplus is judged against its own night, not against an average
    one. Both sides must be usable, so an excluded night drops its day too.
    """
    d = daily[daily["covered"]][["date", "surplus_kwh"]]
    n = nights_df[nights_df["covered"]].copy()
    n["date"] = pd.to_datetime(n["date"])
    return d.merge(n[["date", "import_kwh", "is_ev"]], on="date")


def coverable_fraction(daily: pd.DataFrame, nights_df: pd.DataFrame) -> float:
    """Fraction of nights that day's surplus could have supplied.

    Infinite battery, no power limit, no round-trip loss — the ceiling no
    capacity can beat, computed rather than quoted so the report's headline
    caveat cannot drift from the data it describes.
    """
    m = _paired(daily, nights_df)
    if m.empty:
        raise ValueError(
            "coverable_fraction: no dates in common between the daily surplus "
            "and the covered nights — refusing to silently return 0.0"
        )
    return float((m["surplus_kwh"] >= m["import_kwh"]).mean())


def monthly_wall(daily: pd.DataFrame, nights_df: pd.DataFrame) -> pd.DataFrame:
    """Median daytime surplus against median night need, per calendar month.

    Medians rather than means: one EV night or one exceptional day should not
    decide whether a month reads as walled.
    """
    m = _paired(daily, nights_df)
    if m.empty:
        return pd.DataFrame(columns=["month", "surplus_kwh", "night_kwh",
                                     "n_days"])
    m = m.assign(month=m["date"].dt.month)
    out = (m.groupby("month")
             .agg(surplus_kwh=("surplus_kwh", "median"),
                  night_kwh=("import_kwh", "median"),
                  n_days=("date", "size"))
             .reset_index())
    return out.sort_values("month").reset_index(drop=True)


def negative_surplus_months(monthly: pd.DataFrame) -> list[int]:
    """The months where the median day does not cover its own daytime load."""
    if monthly.empty:
        return []
    return [int(x) for x in
            monthly.loc[monthly["surplus_kwh"] < 0, "month"].tolist()]


def monthly_discharge(meter_df: pd.DataFrame, nights_df: pd.DataFrame,
                      capacity_kwh: float,
                      power_kw: float = REPORT_POWER_KW,
                      round_trip: float = 0.90,
                      usable_fraction: float = 0.90) -> pd.DataFrame:
    """How much a battery of this size actually delivers, per month.

    The wall expressed as the battery's own behaviour rather than as a
    surplus: a month with nothing to charge from discharges nothing, however
    large the battery. Run on the charge-first ordering, which agrees with
    discharge-first at the published capacity to within a fraction of a
    percent.
    """
    m = meter_df.sort_values("ts_utc")
    night_i, nonev_i, month_i = _masks(m, nights_df)
    sig, src = bound_signals(m)["charge_first"]
    spec = BatterySpec(np.array([float(capacity_kwh)]), power_kw,
                       round_trip, usable_fraction)
    r = simulate(sig, night_i[src], nonev_i[src], month_i[src], spec,
                 dt_hours=DT_HOURS)
    return pd.DataFrame({
        "month": np.arange(1, 13),
        "discharge_kwh": r.monthly_discharge_wh[:, 0] / 1000.0,
    })
