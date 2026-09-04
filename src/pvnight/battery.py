"""Chronological battery simulation, vectorised across candidate capacities.

The simulation is inherently sequential in time — each step's state depends on
the last — but every candidate capacity can advance together as a numpy vector,
so the whole sweep is one pass over the data rather than one pass per capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DT_HOURS = 5.0 / 60.0


@dataclass(frozen=True)
class BatterySpec:
    """Candidate capacities and the inverter they sit behind.

    Capacities are nameplate — the number a product is sold under. Usable
    energy at the terminal is `capacity * usable_fraction`, and the round trip
    is split symmetrically so charging and discharging each cost sqrt(eta).
    """

    capacities_kwh: np.ndarray
    power_kw: float
    round_trip: float = 0.90
    usable_fraction: float = 0.90

    @property
    def usable_wh(self) -> np.ndarray:
        return np.asarray(self.capacities_kwh, dtype=float) * 1000.0 * self.usable_fraction

    @property
    def eta(self) -> float:
        """One-way efficiency: sqrt of the round trip, applied on each side."""
        return float(np.sqrt(self.round_trip))


@dataclass
class SimResult:
    capacities_kwh: np.ndarray
    grid_import_wh: np.ndarray
    export_wh: np.ndarray
    charge_wh: np.ndarray
    discharge_wh: np.ndarray
    night_grid_import_wh: np.ndarray
    nonev_grid_import_wh: np.ndarray
    monthly_discharge_wh: np.ndarray
    final_stored_wh: np.ndarray


def simulate(
    net_wh: np.ndarray,
    night_mask: np.ndarray,
    nonev_mask: np.ndarray,
    month_idx: np.ndarray,
    spec: BatterySpec,
) -> SimResult:
    """Run the whole timeline once for every capacity in `spec`.

    `net_wh` is generation minus load per interval. Positive charges the
    battery and exports the rest; negative discharges and imports the rest.
    `night_mask` and `nonev_mask` select which intervals contribute to the
    night-specific accumulators — the timeline itself is never broken up, so
    the battery's state always reflects every night that actually happened.
    """
    caps = spec.usable_wh
    n = len(caps)
    eta = spec.eta
    limit_wh = spec.power_kw * 1000.0 * DT_HOURS

    stored = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    grid = np.zeros(n)
    export = np.zeros(n)
    night_grid = np.zeros(n)
    nonev_grid = np.zeros(n)
    monthly = np.zeros((12, n))

    for t in range(len(net_wh)):
        e = float(net_wh[t])
        if e > 0.0:
            accept = np.minimum(np.minimum(e, limit_wh), (caps - stored) / eta)
            stored += accept * eta
            charge += accept
            export += e - accept
        elif e < 0.0:
            need = -e
            deliver = np.minimum(np.minimum(need, limit_wh), stored * eta)
            stored -= deliver / eta
            discharge += deliver
            shortfall = need - deliver
            grid += shortfall
            if night_mask[t]:
                night_grid += shortfall
            if nonev_mask[t]:
                nonev_grid += shortfall
            monthly[month_idx[t]] += deliver

    return SimResult(
        capacities_kwh=np.asarray(spec.capacities_kwh, dtype=float),
        grid_import_wh=grid,
        export_wh=export,
        charge_wh=charge,
        discharge_wh=discharge,
        night_grid_import_wh=night_grid,
        nonev_grid_import_wh=nonev_grid,
        monthly_discharge_wh=monthly,
        final_stored_wh=stored,
    )


def net_wh(samples: pd.DataFrame) -> np.ndarray:
    """Generation minus load per interval, in Wh.

    Intervals with no consumption reading become zero rather than
    generation-only: we cannot see that load, and pretending it was zero
    would invent free energy and understate grid import.
    """
    s = samples.sort_values("ts_utc")
    gen = s["power_gen_w"].to_numpy(dtype=float)
    load = s["power_cons_w"].to_numpy(dtype=float)
    net = (gen - load) * DT_HOURS
    return np.where(np.isnan(load), 0.0, net)


def build_masks(
    samples: pd.DataFrame, nights_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-interval flags: inside any night, inside a non-EV night, and month."""
    s = samples.sort_values("ts_utc")
    ts = s["ts_utc"].to_numpy()
    night = np.zeros(len(s), dtype=bool)
    nonev = np.zeros(len(s), dtype=bool)

    n = nights_df.dropna(subset=["night_start_utc", "night_end_utc"])
    lo = np.searchsorted(ts, n["night_start_utc"].to_numpy(), side="left")
    hi = np.searchsorted(ts, n["night_end_utc"].to_numpy(), side="right")
    for a, b, is_ev, covered in zip(lo, hi, n["is_ev"], n["covered"]):
        if not covered:
            continue
        night[a:b] = True
        if not is_ev:
            nonev[a:b] = True

    month_idx = pd.DatetimeIndex(s["ts_utc"]).month.to_numpy() - 1
    return night, nonev, month_idx


def sweep(
    samples: pd.DataFrame,
    nights_df: pd.DataFrame,
    capacities_kwh: np.ndarray,
    power_kws: tuple[float, ...] = (2.5, 3.0, 3.7),
    round_trip: float = 0.90,
    usable_fraction: float = 0.90,
) -> pd.DataFrame:
    """Simulate every capacity against every inverter power, once each."""
    s = samples.sort_values("ts_utc")
    net = net_wh(s)
    night, nonev, month = build_masks(s, nights_df)
    span = (s["ts_utc"].iloc[-1] - s["ts_utc"].iloc[0]).total_seconds()
    years = span / (365.25 * 24 * 3600)

    night_load = -net[night & (net < 0)].sum()
    nonev_load = -net[nonev & (net < 0)].sum()
    # Zero night-time deficit — no covered nights, or none in the mask.
    # Make the resulting NaN deliberate rather than an incidental 0/0 with a
    # RuntimeWarning, matching the np.where guard used for `usable` below.
    night_load = night_load if night_load > 0 else np.nan
    nonev_load = nonev_load if nonev_load > 0 else np.nan

    frames = []
    for p in power_kws:
        spec = BatterySpec(np.asarray(capacities_kwh, dtype=float), p, round_trip, usable_fraction)
        r = simulate(net, night, nonev, month, spec)
        usable = np.where(spec.usable_wh > 0, spec.usable_wh, np.nan)
        df = pd.DataFrame({
            "capacity_kwh": r.capacities_kwh,
            "power_kw": p,
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
        df["nonev_marginal_kwh_per_kwh"] = -df["nonev_night_grid_import_kwh_yr"].diff() / step
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def recommend_capacity(
    sweep_df: pd.DataFrame,
    power_kw: float = 3.0,
    threshold_kwh_per_kwh: float = 50.0,
    scenario: str = "nonev",
) -> float:
    """Smallest capacity whose marginal return has fallen below the threshold.

    "Where the curve flattens" is not implementable, so the rule is explicit:
    an extra kWh of battery earning less than `threshold_kwh_per_kwh` per year
    is cycling under about once a week, which is hard to justify buying. The
    threshold is a stated judgement, not a derived constant — the report prints
    the whole curve so a reader can choose differently.
    """
    col = "marginal_kwh_per_kwh" if scenario == "all" else "nonev_marginal_kwh_per_kwh"
    d = sweep_df[sweep_df["power_kw"] == power_kw].sort_values("capacity_kwh")
    below = d[d[col].notna() & (d[col] < threshold_kwh_per_kwh)]
    if below.empty:
        return float(d["capacity_kwh"].max())
    return float(below["capacity_kwh"].iloc[0])
