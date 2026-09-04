"""Battery simulation driven by the meter, at both bounds of the
within-interval ambiguity.

A 15-minute interval can record both import and export, and 13.5% of them do.
Collapsing them to a net figure hides export the battery could have stored;
treating them as fully separable credits it with more than it could certainly
have done. Both are run, and the pair is the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import BatterySpec, simulate
from .meter import DT_HOURS


def bound_signals(meter_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(net_wh, gross_wh)` per interval, in Wh.

    `net` is export minus import, the conservative reading: the two flows are
    collapsed, so an interval that carried both looks like whichever won.
    `gross` always adds back the smaller, cancelling flow on top of `net`,
    in either direction — as if the battery had captured it before the
    larger flow arrived. `both = min(import, export) >= 0`, so `gross` is
    never worse than `net` in a single interval, which is what guarantees
    the two bounds stay correctly ordered after the full simulation too.
    """
    m = meter_df.sort_values("ts_utc")
    imp = m["import_kwh"].to_numpy(dtype=float) * 1000.0
    exp = m["export_kwh"].to_numpy(dtype=float) * 1000.0
    net = exp - imp
    both = np.minimum(imp, exp)
    gross = net + both
    return net, gross


def _masks(meter_df: pd.DataFrame, nights_df: pd.DataFrame):
    m = meter_df.sort_values("ts_utc")
    ts = m["ts_utc"].to_numpy()
    night = np.zeros(len(m), bool)
    nonev = np.zeros(len(m), bool)
    n = nights_df.dropna(subset=["night_start_utc", "night_end_utc"])
    lo = np.searchsorted(ts, n["night_start_utc"].to_numpy(), "left")
    hi = np.searchsorted(ts, n["night_end_utc"].to_numpy(), "right")
    for a, b, is_ev, covered in zip(lo, hi, n["is_ev"], n["covered"]):
        if not covered:
            continue
        night[a:b] = True
        if not is_ev:
            nonev[a:b] = True
    return night, nonev, pd.DatetimeIndex(m["ts_utc"]).month.to_numpy() - 1


def sweep_bounds(
    meter_df: pd.DataFrame,
    nights_df: pd.DataFrame,
    capacities_kwh: np.ndarray,
    power_kws: tuple[float, ...] = (2.5, 3.0, 3.7),
    round_trip: float = 0.90,
    usable_fraction: float = 0.90,
) -> pd.DataFrame:
    """Phase 2's sweep metrics, once per bound."""
    m = meter_df.sort_values("ts_utc")
    night, nonev, month = _masks(m, nights_df)
    span = (m["ts_utc"].iloc[-1] - m["ts_utc"].iloc[0]).total_seconds()
    years = span / (365.25 * 24 * 3600)
    signals = dict(zip(("net", "gross"), bound_signals(m)))

    frames = []
    for name, sig in signals.items():
        night_load = -sig[night & (sig < 0)].sum()
        nonev_load = -sig[nonev & (sig < 0)].sum()
        night_load = night_load if night_load > 0 else np.nan
        nonev_load = nonev_load if nonev_load > 0 else np.nan
        for p in power_kws:
            spec = BatterySpec(np.asarray(capacities_kwh, float), p,
                               round_trip, usable_fraction)
            r = simulate(sig, night, nonev, month, spec, dt_hours=DT_HOURS)
            usable = np.where(spec.usable_wh > 0, spec.usable_wh, np.nan)
            df = pd.DataFrame({
                "capacity_kwh": r.capacities_kwh,
                "power_kw": p,
                "bound": name,
                "grid_import_kwh_yr": r.grid_import_wh / 1000 / years,
                "pv_export_kwh_yr": r.export_wh / 1000 / years,
                "night_grid_import_kwh_yr": r.night_grid_import_wh / 1000 / years,
                "nonev_night_grid_import_kwh_yr": r.nonev_grid_import_wh / 1000 / years,
                "night_self_sufficiency_pct": 100 * (1 - r.night_grid_import_wh / night_load),
                "nonev_night_self_sufficiency_pct": 100 * (1 - r.nonev_grid_import_wh / nonev_load),
                "cycles_per_yr": r.discharge_wh / usable / years,
            }).sort_values("capacity_kwh").reset_index(drop=True)
            step = df["capacity_kwh"].diff()
            df["marginal_kwh_per_kwh"] = -df["grid_import_kwh_yr"].diff() / step
            df["nonev_marginal_kwh_per_kwh"] = (
                -df["nonev_night_grid_import_kwh_yr"].diff() / step)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
