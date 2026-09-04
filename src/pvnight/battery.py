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
    dt_hours: float = DT_HOURS,
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
    # The interval length is a parameter because the meter data is 15-minute
    # where PVOutput is 5-minute. The power cap converts to an energy limit
    # per interval, so it is the one place resolution genuinely bites.
    limit_wh = spec.power_kw * 1000.0 * dt_hours

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
    """Simulate every capacity against every inverter power, once each.

    ``cycles_per_yr`` divides AC-side delivered energy (``discharge_wh``,
    what actually reached the house) by DC-side usable capacity
    (``usable_wh``, before the discharge-side efficiency loss). That
    understates true full-equivalent cycle throughput by the discharge
    efficiency factor (sqrt(round_trip) ~= 94.9% at the default 90% round
    trip), i.e. by about 5% — conservative, not misleading in the dangerous
    direction, but worth knowing when comparing this figure to a spec sheet
    or to "cycles roughly once a week" reasoning.
    """
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
    """Smallest capacity beyond which the marginal return never recovers.

    The marginal-return curve is NOT monotonic. A very small battery is
    exhausted within minutes of sunset, so its first half-kWh buys little;
    marginal value climbs to a peak near 2.5 kWh before saturating. A
    "first capacity below the threshold" rule therefore fires on the
    leading edge of that climb and returns a degenerate answer.

    So: find the largest capacity still at or above the threshold, and
    recommend the next step up.

    NOT USED BY THE REPORT. The default 50 kWh/yr threshold was a judgement
    call — reasoned from "an extra kWh cycling under once a week is hard to
    justify", but never derived from a battery price, a tariff, or any
    measurement. The report therefore uses `elbow_capacity` instead, which
    needs no such choice. This function is kept because the rule is sound
    once the threshold is a real number: divide battery cost per kWh by your
    electricity rate over the warranty period to get the kWh/yr a marginal
    kWh must return, and pass that as `threshold_kwh_per_kwh`.
    """
    col = "marginal_kwh_per_kwh" if scenario == "all" else "nonev_marginal_kwh_per_kwh"
    d_power = sweep_df[sweep_df["power_kw"] == power_kw].sort_values("capacity_kwh")
    d = d_power[d_power[col].notna()]
    if d.empty:
        return float(d_power["capacity_kwh"].max())

    at_or_above = d[d[col] >= threshold_kwh_per_kwh]
    if at_or_above.empty:
        return float(d["capacity_kwh"].iloc[0])

    last = at_or_above["capacity_kwh"].iloc[-1]
    beyond = d[d["capacity_kwh"] > last]
    if beyond.empty:
        return float(last)
    return float(beyond["capacity_kwh"].iloc[0])


def _import_column(scenario: str) -> str:
    return ("night_grid_import_kwh_yr" if scenario == "all"
            else "nonev_night_grid_import_kwh_yr")


def benefit_share_pct(
    sweep_df: pd.DataFrame, power_kw: float = 3.0, scenario: str = "nonev"
) -> pd.Series:
    """Share of the achievable benefit each capacity captures, 0 to 100.

    The denominator is the reduction in night grid import between the
    smallest and largest simulated capacity. At the top of this project's
    sweep the marginal return is about 1.2 kWh/yr per kWh, i.e. effectively
    flat, so that endpoint stands in for an infinitely large battery.

    This is the threshold-free way to read the curve: it needs no judgement
    about what an extra kWh must earn, only the curve's own endpoints. It
    answers "how much of what is possible does this size get me", which is
    the question a buyer actually has.
    """
    col = _import_column(scenario)
    d = sweep_df[sweep_df["power_kw"] == power_kw].sort_values("capacity_kwh")
    y = d[col].to_numpy(dtype=float)
    achievable = y[0] - y[-1]
    if not achievable > 0:
        return pd.Series(np.full(len(y), np.nan), index=d.index, name="benefit_share_pct")
    return pd.Series(100.0 * (y[0] - y) / achievable, index=d.index,
                     name="benefit_share_pct")


def elbow_capacity(
    sweep_df: pd.DataFrame, power_kw: float = 3.0, scenario: str = "nonev"
) -> float:
    """The bend in the import-versus-capacity curve, with no threshold at all.

    Takes the point of maximum perpendicular distance from the straight chord
    joining the curve's first and last points — the standard parameter-free
    elbow method. Where `recommend_capacity` needs someone to decide what an
    extra kWh must earn, this needs nothing: the curve's own geometry picks
    the point where it stops bending and starts merely drifting.

    Note this is NOT the inflection point. The inflection sits at the peak of
    the marginal-return curve (about 2.5 kWh here), where diminishing returns
    *begin*. The elbow is further out, where they have largely finished.

    A perfectly straight curve has no bend; return the largest capacity
    rather than an arbitrary interior point.
    """
    col = _import_column(scenario)
    d = sweep_df[sweep_df["power_kw"] == power_kw].sort_values("capacity_kwh")
    x = d["capacity_kwh"].to_numpy(dtype=float)
    y = d[col].to_numpy(dtype=float)
    if len(x) < 3:
        return float(x[-1])

    span = np.hypot(x[-1] - x[0], y[-1] - y[0])
    if not span > 0:
        return float(x[-1])
    distance = np.abs(
        (y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0]
    ) / span
    if np.allclose(distance, 0.0):
        return float(x[-1])
    return float(x[int(np.argmax(distance))])


def elbow_stability(
    sweep_df: pd.DataFrame,
    power_kw: float = 3.0,
    tops: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0),
    scenario: str = "nonev",
) -> pd.DataFrame:
    """How the elbow moves as the sweep is truncated at different capacities.

    The elbow is often described as parameter-free, and it needs no threshold
    — but it is not assumption-free: it depends on where the curve ends,
    because the chord it measures against is drawn to that endpoint. On this
    installation it reads 6.0 kWh from a 0-10 kWh sweep and settles at 8.0
    once the sweep reaches 25 kWh and the top end has gone flat.

    Publishing this table is the honest alternative to claiming an
    independence the method does not have. A reader can see for themselves
    whether the sweep was carried far enough for the answer to have settled.
    """
    rows = []
    d_all = sweep_df[sweep_df["power_kw"] == power_kw].sort_values("capacity_kwh")
    col = _import_column(scenario)
    marg = ("marginal_kwh_per_kwh" if scenario == "all"
            else "nonev_marginal_kwh_per_kwh")
    for top in tops:
        sub = sweep_df[sweep_df["capacity_kwh"] <= top]
        d = d_all[d_all["capacity_kwh"] <= top]
        if len(d) < 3:
            continue
        tail = d[marg].to_numpy(dtype=float)[-1] if marg in d else np.nan
        rows.append({
            "sweep_top_kwh": float(top),
            "elbow_kwh": elbow_capacity(sub, power_kw=power_kw, scenario=scenario),
            "top_end_marginal": float(tail),
        })
    return pd.DataFrame(rows, columns=["sweep_top_kwh", "elbow_kwh", "top_end_marginal"])


def convergence_readings(
    sweep_df: pd.DataFrame, power_kw: float = 3.0, scenario: str = "nonev"
) -> pd.DataFrame:
    """Every threshold-free reading of the capacity curve, side by side.

    None of these is authoritative, and that is the point of showing them
    together. The curve has no sharp corner, so each method's apparent
    precision comes from its own assumptions rather than from the data. What
    the data supports is the range they span, not any one of them.

    The inflection of the import curve is included for orientation but is not
    a candidate size: it marks where diminishing returns *begin*, and a
    battery that small captures well under a third of the achievable benefit.

    Note the two elbow rows differ only in where the measurement window
    starts. That the answer moves at all is the honest caveat on the method.
    """
    col = _import_column(scenario)
    marg = ("marginal_kwh_per_kwh" if scenario == "all"
            else "nonev_marginal_kwh_per_kwh")
    d = sweep_df[sweep_df["power_kw"] == power_kw].sort_values("capacity_kwh")
    x = d["capacity_kwh"].to_numpy(dtype=float)
    y = d[col].to_numpy(dtype=float)
    m = d[marg].to_numpy(dtype=float)

    inflection = float(x[int(np.nanargmax(m))])
    beyond = x > inflection
    dm = np.gradient(np.nan_to_num(m, nan=np.nanmean(m)), x)
    steepest = float(x[beyond][int(np.nanargmin(dm[beyond]))])
    marg_infl = float(x[beyond][int(np.nanargmax(np.gradient(dm, x)[beyond]))])

    from_inflection = sweep_df[sweep_df["capacity_kwh"] >= inflection]

    return pd.DataFrame(
        [
            ("inflection of the import curve", inflection,
             "where diminishing returns begin — not a size to buy"),
            ("steepest collapse of marginal return", steepest,
             "where each added kWh loses value fastest"),
            ("inflection of the marginal curve", marg_infl,
             "where that collapse stops accelerating"),
            ("elbow, full sweep", elbow_capacity(sweep_df, power_kw, scenario),
             "greatest distance from the chord across 0-30 kWh"),
            ("elbow, measured from the inflection", 
             elbow_capacity(from_inflection, power_kw, scenario),
             "the same method, started past the leading rise"),
        ],
        columns=["method", "capacity_kwh", "what_it_measures"],
    )
