"""Night consumption and battery sizing, measured at the smart meter.

Run: uv run python analyze_meter.py

Phase 2 answered this question from PVOutput's consumption channel, which
stopped seeing part of the house in December 2022. This entry point redoes it
against the meter, which measures what actually crosses the grid connection,
and publishes both side by side with phase 2 labelled superseded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pvnight import battery, compare_report, meter, meter_battery, meter_nights
from pvnight.config import DATA_SUBDIR
from pvnight.loader import load

CAPACITIES = np.arange(0.0, 30.01, 0.5)
POWER_KWS = (2.5, 3.0, 3.7)
REPORT_POWER_KW = 3.0

# PVOutput samples every five minutes; the meter every fifteen. Three of the
# former make one of the latter, which is what makes the resolution penalty
# measurable on real data rather than assumed.
SAMPLES_PER_METER_INTERVAL = 3

_WINDOW_DATES = ["date", "solar_start_utc", "solar_end_utc",
                 "night_start_utc", "night_end_utc"]


def _read_windows(repo_root: Path) -> pd.DataFrame:
    """The phase-1 window table.

    Resolved from ``repo_root``, never from a data directory: the window
    table is an OUTPUT of phase 1, and deriving its path from an input path
    is the mistake phase 2 recorded in the spec's section 5.
    """
    return pd.read_csv(repo_root / "out" / "solar_windows.csv",
                       parse_dates=_WINDOW_DATES)


def _read_pv_nights(repo_root: Path) -> pd.DataFrame:
    """Phase 2's night summary, read back for comparison only."""
    return pd.read_csv(repo_root / "out" / "night_summary.csv",
                       parse_dates=["date", "night_start_utc", "night_end_utc"])


def monthly_ratio(meter_nights_df: pd.DataFrame, pv_nights: pd.DataFrame) -> pd.DataFrame:
    """PVOutput night energy over meter night energy, per calendar month.

    This is the evidence behind the whole report, so it is computed on the
    nights both sources call usable and on nothing else: a month's ratio is
    the sum of PVOutput's night kWh over the sum of the meter's, across the
    nights they have in common. Months where the two disagree about coverage
    contribute to neither side, which is why the ratio measures the channel
    rather than the calendar.
    """
    pv = pv_nights[pv_nights["covered"]].set_index("date")["night_wh"] / 1000.0
    me = meter_nights_df[meter_nights_df["covered"]].set_index("date")["import_kwh"]
    # sort=True is explicit rather than defaulted: pandas is changing that
    # default, and a chronological join axis is what the monthly grouping
    # below reads best from.
    j = pd.concat({"pv_kwh": pv, "meter_kwh": me}, axis=1, sort=True).dropna()
    if j.empty:
        raise ValueError(
            "monthly_ratio: no nights are usable in both sources — refusing "
            "to publish an empty comparison"
        )
    j["month"] = j.index.to_period("M")
    out = j.groupby("month")[["pv_kwh", "meter_kwh"]].sum().reset_index()
    out["ratio"] = out["pv_kwh"] / out["meter_kwh"]
    return out


def pv_shortfall_pct(ratio_df: pd.DataFrame, year: int) -> float:
    """How much of that year's night load PVOutput failed to see, in percent.

    Summed over the year's months rather than averaged over their ratios, so
    a quiet month cannot carry the same weight as a heavy one.
    """
    y = ratio_df[ratio_df["month"].dt.year == year]
    if y.empty:
        return float("nan")
    return float(100.0 * (1.0 - y["pv_kwh"].sum() / y["meter_kwh"].sum()))


def resolution_penalty_pct(
    samples: pd.DataFrame,
    pv_nights: pd.DataFrame,
    pv_sweep: pd.DataFrame,
    capacities_kwh: np.ndarray = CAPACITIES,
    power_kw: float = REPORT_POWER_KW,
) -> float:
    """How much better a 15-minute simulation looks than a 5-minute one.

    Measured, not asserted. Phase 2's PVOutput data genuinely exists at both
    resolutions: its five-minute net signal is aggregated onto the meter's own
    quarter-hour boundaries, the same capacity sweep is run over each, and the
    two are read at the *five-minute* elbow — the finer run is the truth the
    coarser one is being scored against.

    Returned as a percentage of the fifteen-minute figure, because fifteen
    minutes is what the meter records: the number says how much the meter-
    based answer flatters itself.

    Grouping is by ``floor("15min")`` on the timestamp, not by reshaping the
    array in threes, and the difference is not cosmetic. PVOutput's record
    starts at 03:40, so a reshape from index 0 puts 192,984 of 196,477 groups
    on a boundary ten minutes off the clock the meter keeps, and additionally
    merges across the twenty places where PVOutput skipped samples — one such
    "quarter hour" spans 13h45. Measured at the elbow, the four phasings give
    +0.0800% (reshape from 0), -0.0096% (from 1), -0.0117% (from 2) and
    -0.0163% (clock-aligned). The spread is 0.10 pp and the sign flips, which
    is larger than the quantity being measured — so the phase is not a detail,
    and the only defensible choice is the meter's own boundaries.
    """
    s = samples.sort_values("ts_utc").reset_index(drop=True)
    # net_wh maps a NaN consumption reading to 0.0, so the sums below cannot
    # be NaN-poisoned the way phase 2's prefix sum was. That guarantee rests
    # on generation never being NaN (loader fillna(0.0)s it) — assert it here
    # rather than depend on it silently.
    assert not s["power_gen_w"].isna().any(), "generation must not be NaN"
    net = battery.net_wh(s)
    night, nonev, month = battery.build_masks(s, pv_nights)

    grouped = pd.DataFrame({
        "quarter": pd.DatetimeIndex(s["ts_utc"]).floor(
            f"{int(SAMPLES_PER_METER_INTERVAL * 5)}min"),
        "net": net, "night": night, "nonev": nonev, "month": month,
    }).groupby("quarter", sort=True).agg(
        # Majority vote for the masks: a coarse interval belongs to the night
        # that owns most of it. Only the interval at each night's edge can
        # differ from `any` or `all`, out of the ~65 in a night. `mean` also
        # does the right thing for the short groups at a data gap, where a
        # fixed divisor of three would not.
        net=("net", "sum"), night=("night", "mean"),
        nonev=("nonev", "mean"), month=("month", "first"),
    )
    net_15 = grouped["net"].to_numpy()
    night_15 = (grouped["night"] >= 0.5).to_numpy()
    nonev_15 = (grouped["nonev"] >= 0.5).to_numpy()
    month_15 = grouped["month"].to_numpy()

    span = (s["ts_utc"].iloc[-1] - s["ts_utc"].iloc[0]).total_seconds()
    years = span / (365.25 * 24 * 3600)

    spec = battery.BatterySpec(np.asarray(capacities_kwh, dtype=float), power_kw)
    coarse = battery.simulate(net_15, night_15, nonev_15, month_15, spec,
                              dt_hours=meter.DT_HOURS)
    coarse_kwh_yr = pd.Series(coarse.nonev_grid_import_wh / 1000.0 / years,
                              index=np.asarray(capacities_kwh, dtype=float))

    # The five-minute side of the comparison is phase 2's own sweep, run at
    # the same capacities and power — no second simulation of the same thing.
    fine = pv_sweep[pv_sweep["power_kw"] == power_kw].set_index("capacity_kwh")
    elbow_5min = battery.elbow_capacity(pv_sweep, power_kw=power_kw)

    at_coarse = float(coarse_kwh_yr.loc[elbow_5min])
    at_fine = float(fine.loc[elbow_5min, "nonev_night_grid_import_kwh_yr"])
    if not at_coarse > 0:
        return float("nan")
    return 100.0 * (at_fine - at_coarse) / at_coarse


def _excluded_in_worst_gap_month(meter_nights_df: pd.DataFrame,
                                 gaps: pd.DataFrame) -> tuple[int, str]:
    """Nights lost to the worst outage, and the month it fell in.

    The month is read from ``gaps`` via the report's own summary rather than
    recomputed here, so the dict key, the report prose and the stat tile
    cannot drift apart — an earlier round of this project spent a whole fix
    removing a hardcoded "January 2024" from the prose.
    """
    g = compare_report.gap_summary(gaps)
    if g is None:
        return 0, ""
    ym = pd.to_datetime(meter_nights_df["date"]).dt.to_period("M")
    in_month = ym == g["worst_month"]
    return int((in_month & ~meter_nights_df["covered"]).sum()), str(g["month_name"])


def run(repo_root: Path, out_dir: Path) -> dict:
    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = _read_windows(repo_root)

    meter_df = meter.load_meter(repo_root / meter.METER_SUBDIR)
    gaps = meter.find_gaps(meter_df)
    nights_df = meter_nights.summarise(meter_df, windows, gaps)
    sweep_df = meter_battery.sweep_bounds(meter_df, nights_df, CAPACITIES, POWER_KWS)

    # The elbow of each bound's curve, not a chosen cut-off. Neither bound is
    # the answer on its own; the pair is the range the report publishes.
    elbows = {
        b: battery.elbow_capacity(sweep_df[sweep_df["bound"] == b],
                                  power_kw=REPORT_POWER_KW)
        for b in ("charge_first", "discharge_first")
    }

    sensitivity = meter_nights.ev_sensitivity(meter_df, windows, gaps)

    samples = load(repo_root / DATA_SUBDIR)
    pv_nights = _read_pv_nights(repo_root)
    pv_sweep = battery.sweep(samples, pv_nights, CAPACITIES, POWER_KWS)

    ratio = monthly_ratio(nights_df, pv_nights)
    penalty = resolution_penalty_pct(samples, pv_nights, pv_sweep)
    excluded, worst_month_name = _excluded_in_worst_gap_month(nights_df, gaps)

    nights_df.to_csv(out_dir / "meter_night_summary.csv", index=False)
    sweep_df.to_csv(out_dir / "meter_battery_sweep.csv", index=False)
    (out_dir / "meter_report.html").write_text(
        compare_report.build_html(
            nights_df, pv_nights, sweep_df, pv_sweep, sensitivity,
            ratio[["month", "ratio"]], gaps,
            resolution_penalty_pct=penalty,
            excluded_nights=excluded,
        )
    )

    covered = nights_df[nights_df["covered"]]
    return {
        "n_nights": int(len(covered)),
        "n_ev_nights": int(covered["is_ev"].sum()),
        "median_night_kwh": float(covered["import_kwh"].median()),
        "elbow_charge_first_kwh": float(elbows["charge_first"]),
        "elbow_discharge_first_kwh": float(elbows["discharge_first"]),
        "resolution_penalty_pct": float(penalty),
        "excluded_nights_worst_month": int(excluded),
        "worst_gap_month": worst_month_name,
        "pv_shortfall_2025_pct": pv_shortfall_pct(ratio, 2025),
    }


if __name__ == "__main__":
    here = Path(__file__).parent
    for k, v in run(here, here / "out").items():
        print(f"{k}: {v}")
