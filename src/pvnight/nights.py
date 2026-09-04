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
    so the diff is grouped by `solar_date` — a global diff would go sharply
    negative at every midnight. Mirrors phase 1's treatment of generation.
    """
    return (
        samples.groupby("solar_date")["energy_cons_wh"].diff().fillna(0.0)
    ).rename("cons_delta_wh")


def summarise_nights(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    ev_power_w: float = EV_POWER_W,
    ev_hours: float = EV_HOURS,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """One row per night, with energy, peak, EV flag and sample coverage."""
    s = samples.sort_values("ts_utc").reset_index(drop=True)
    cum = np.concatenate([[0.0], consumption_increments(s).to_numpy().cumsum()])
    ts = s["ts_utc"].to_numpy()
    power = s["power_cons_w"].to_numpy()

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
        seg = power[a:b]
        seg = seg[~np.isnan(seg)]
        present[k] = len(seg)
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
        "dst_hour_missing": w["dst_hour_missing"] if "dst_hour_missing" in w else False,
    })
    out["is_ev"] = out["hours_above_2kw"] >= ev_hours
    out["covered"] = out["coverage"] >= min_coverage
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
        for h in hours:
            # NOTE: the column is named for the default 2 kW threshold, but it
            # holds hours above `p` on this pass. Do not read it as literally
            # 2 kW here.
            is_ev = base["hours_above_2kw"] >= h
            rows.append({
                "power_w": p,
                "hours": h,
                "n_ev": int(is_ev.sum()),
                "median_ev_kwh": base.loc[is_ev, "night_wh"].median() / 1000,
                "median_rest_kwh": base.loc[~is_ev, "night_wh"].median() / 1000,
            })
    return pd.DataFrame(rows)
