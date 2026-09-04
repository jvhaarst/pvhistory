"""Per-night consumption, aggregated against the phase-1 solar window."""

from __future__ import annotations

import numpy as np
import pandas as pd

DT_HOURS = 5.0 / 60.0
EV_POWER_W = 2000.0
EV_HOURS = 2.0
MIN_COVERAGE = 0.95


def consumption_increments(samples: pd.DataFrame) -> pd.Series:
    """Wh consumed in each 5-minute sample.

    `energy_cons_wh` is cumulative within a local day and resets at midnight,
    so the diff is grouped by `solar_date`. Only the first row of each day is
    zero-filled, because its NaN is an artefact of `diff()` having no
    predecessor. A NaN anywhere else is a genuine unrecorded reading and is
    left as NaN: filling it would invent a zero, and would also discard the
    real increment on the row after the gap. Callers must treat NaN as
    "unknown", never as "no consumption".

    Input must be ordered by `ts_utc` within each `solar_date`.
    """
    delta = samples.groupby("solar_date")["energy_cons_wh"].diff()
    first_of_day = samples.groupby("solar_date").cumcount() == 0
    return delta.mask(first_of_day & delta.isna(), 0.0).rename("cons_delta_wh")


def summarise_nights(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    ev_power_w: float = EV_POWER_W,
    ev_hours: float = EV_HOURS,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """One row per night, with energy, peak, EV flag and sample coverage."""
    s = samples.sort_values("ts_utc").reset_index(drop=True)
    delta = consumption_increments(s).to_numpy()
    cum = np.concatenate([[0.0], np.nancumsum(delta)])
    gap_cum = np.concatenate([[0], np.isnan(delta).cumsum()])
    ts = s["ts_utc"].to_numpy()
    power = s["power_cons_w"].to_numpy()
    energy = s["energy_cons_wh"].to_numpy()

    w = windows.dropna(subset=["night_start_utc", "night_end_utc"]).reset_index(drop=True)
    lo = np.searchsorted(ts, w["night_start_utc"].to_numpy(), side="left")
    hi = np.searchsorted(ts, w["night_end_utc"].to_numpy(), side="right")

    duration_h = (
        w["night_end_utc"] - w["night_start_utc"]
    ).dt.total_seconds().to_numpy() / 3600.0
    expected = duration_h / DT_HOURS

    peak = np.zeros(len(w))
    hours_hi = np.zeros(len(w))
    present = np.zeros(len(w))
    for k, (a, b) in enumerate(zip(lo, hi)):
        seg_power = power[a:b]
        seg_energy = energy[a:b]
        both_present = ~np.isnan(seg_power) & ~np.isnan(seg_energy)
        present[k] = both_present.sum()
        seg = seg_power[~np.isnan(seg_power)]
        if len(seg):
            peak[k] = seg.max()
            hours_hi[k] = (seg > ev_power_w).sum() * DT_HOURS

    out = pd.DataFrame({
        "date": w["date"],
        "night_start_utc": w["night_start_utc"],
        "night_end_utc": w["night_end_utc"],
        "night_wh": cum[hi] - cum[lo],
        "peak_w": peak,
        "hours_above_2kw": hours_hi,
        "coverage": np.where(expected > 0, present / expected, 0.0),
        "missing_increments": gap_cum[hi] - gap_cum[lo],
        "dst_hour_missing": w["dst_hour_missing"] if "dst_hour_missing" in w else False,
    })
    out["is_ev"] = out["hours_above_2kw"] >= ev_hours
    out["covered"] = (out["coverage"] >= min_coverage) & (out["missing_increments"] == 0)
    return out


def ev_sensitivity(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    powers: tuple[float, ...] = (1500.0, 2000.0, 2500.0, 3000.0),
    hours: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """How the EV/non-EV split moves as the threshold moves.

    Published in the report because the classification is a heuristic: a split
    that swings wildly across this grid is an artefact of where the line was
    drawn, and the reader deserves to see that.
    """
    rows = []
    for p in powers:
        base = summarise_nights(samples, windows, ev_power_w=p, ev_hours=min(hours))
        base = base[base["covered"]]
        # Rename immediately: `summarise_nights` always calls this column
        # `hours_above_2kw`, but on this pass it holds hours above `p`, which
        # varies across the grid. Renaming here stops the misleading name
        # from escaping into the next reader's hands.
        base = base.rename(columns={"hours_above_2kw": "hours_above_p"})
        for h in hours:
            is_ev = base["hours_above_p"] >= h
            rows.append({
                "power_w": p,
                "hours": h,
                "n_ev": int(is_ev.sum()),
                "median_ev_kwh": base.loc[is_ev, "night_wh"].median() / 1000,
                "median_rest_kwh": base.loc[~is_ev, "night_wh"].median() / 1000,
            })
    return pd.DataFrame(rows)
