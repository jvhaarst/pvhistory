"""Per-night figures taken from the meter rather than from PVOutput.

At night there is no generation, so meter import *is* household consumption.
That equivalence is what lets this module answer the same question phase 2
answered, from a source that sees the whole house.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .meter import DT_HOURS

EV_POWER_KW = 2.0
EV_HOURS = 2.0


def _overlaps_a_gap(starts, ends, gaps: pd.DataFrame) -> np.ndarray:
    """Missing intervals inside each night. Zero when the frame is clean."""
    out = np.zeros(len(starts), dtype=int)
    if gaps.empty:
        return out
    gs = gaps["gap_start_utc"].to_numpy()
    ge = gaps["gap_end_utc"].to_numpy()
    n = gaps["missing_intervals"].to_numpy()
    for k, (a, b) in enumerate(zip(starts, ends)):
        hit = (gs < b) & (ge > a)
        out[k] = int(n[hit].sum())
    return out


def summarise(
    meter_df: pd.DataFrame,
    windows: pd.DataFrame,
    gaps: pd.DataFrame,
    ev_power_kw: float = EV_POWER_KW,
    ev_hours: float = EV_HOURS,
) -> pd.DataFrame:
    """One row per night, from the meter.

    `covered` means the night sits inside the meter's span and contains no
    missing intervals. There is no partial-coverage ratio as in phase 2: the
    meter either recorded an interval or it did not.
    """
    m = meter_df.sort_values("ts_utc").reset_index(drop=True)
    ts = m["ts_utc"].to_numpy()
    imp = m["import_kwh"].to_numpy()
    cum_i = np.concatenate([[0.0], imp.cumsum()])
    cum_e = np.concatenate([[0.0], m["export_kwh"].to_numpy().cumsum()])

    w = windows.dropna(subset=["night_start_utc", "night_end_utc"]).reset_index(drop=True)
    starts = w["night_start_utc"].to_numpy()
    ends = w["night_end_utc"].to_numpy()
    lo = np.searchsorted(ts, starts, side="left")
    hi = np.searchsorted(ts, ends, side="right")

    peak = np.zeros(len(w))
    hours_hi = np.zeros(len(w))
    for k, (a, b) in enumerate(zip(lo, hi)):
        seg = imp[a:b]
        if len(seg):
            peak[k] = seg.max() / DT_HOURS
            hours_hi[k] = (seg / DT_HOURS > ev_power_kw).sum() * DT_HOURS

    missing = _overlaps_a_gap(starts, ends, gaps)
    inside = (starts >= ts[0]) & (ends <= ts[-1])

    out = pd.DataFrame({
        "date": w["date"],
        "night_start_utc": w["night_start_utc"],
        "night_end_utc": w["night_end_utc"],
        "import_kwh": cum_i[hi] - cum_i[lo],
        "export_kwh": cum_e[hi] - cum_e[lo],
        "peak_kw": peak,
        "hours_above_2kw": hours_hi,
        "missing_intervals": missing,
    })
    out["is_ev"] = out["hours_above_2kw"] >= ev_hours
    out["covered"] = inside & (missing == 0) & (hi > lo)
    return out


def ev_sensitivity(
    meter_df: pd.DataFrame,
    windows: pd.DataFrame,
    gaps: pd.DataFrame,
    powers: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0),
    hours: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """How the EV split moves as the threshold moves.

    Republished because phase 2's rule was fitted to a signal missing much of
    the car; the threshold that separated a partial signal is not necessarily
    the one that separates a complete one.
    """
    rows = []
    for p in powers:
        base = summarise(meter_df, windows, gaps, ev_power_kw=p, ev_hours=min(hours))
        base = base[base["covered"]].rename(columns={"hours_above_2kw": "hours_above_p"})
        for h in hours:
            is_ev = base["hours_above_p"] >= h
            rows.append({
                "power_kw": p,
                "hours": h,
                "n_ev": int(is_ev.sum()),
                "median_ev_kwh": base.loc[is_ev, "import_kwh"].median(),
                "median_rest_kwh": base.loc[~is_ev, "import_kwh"].median(),
            })
    return pd.DataFrame(rows)
