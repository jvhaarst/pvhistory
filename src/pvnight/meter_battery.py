"""Battery simulation driven by the meter, at both within-interval orderings.

A 15-minute interval can record both import and export, and 13.5% of them do.
The meter cannot say which came first, and that ordering decides how much of
the import a battery could have bridged with the export. `charge_first` and
`discharge_first` are run as a bracket; neither is the answer on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import BatterySpec, simulate
from .meter import DT_HOURS


def bound_signals(meter_df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """`{name: (signal_wh, source_idx)}` for the two within-interval orderings.

    The meter cannot say whether the export or the import came first inside a
    quarter hour, and that ordering decides how much a battery could bridge.
    `charge_first` stores the export then spends it on the import;
    `discharge_first` meets the import before the export exists. Both
    reproduce the measured grid import exactly at zero capacity, which is what
    makes them a genuine bracket rather than two different questions.
    """
    m = meter_df.sort_values("ts_utc")
    imp = m["import_kwh"].to_numpy(dtype=float) * 1000.0
    exp = m["export_kwh"].to_numpy(dtype=float) * 1000.0
    n = len(imp)
    src = np.repeat(np.arange(n), 2)

    charge_first = np.empty(2 * n, dtype=float)
    charge_first[0::2] = exp
    charge_first[1::2] = -imp

    discharge_first = np.empty(2 * n, dtype=float)
    discharge_first[0::2] = -imp
    discharge_first[1::2] = exp

    return {"charge_first": (charge_first, src),
            "discharge_first": (discharge_first, src)}


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
    night_i, nonev_i, month_i = _masks(m, nights_df)
    span = (m["ts_utc"].iloc[-1] - m["ts_utc"].iloc[0]).total_seconds()
    years = span / (365.25 * 24 * 3600)

    frames = []
    for name, (sig, src) in bound_signals(m).items():
        night = night_i[src]
        nonev = nonev_i[src]
        month = month_i[src]
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
