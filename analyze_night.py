"""Night consumption and battery sizing. Run: uv run python analyze_night.py"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pvnight import battery, night_report, nights
from pvnight.loader import load

CAPACITIES = np.arange(0.0, 30.01, 0.5)
POWER_KWS = (2.5, 3.0, 3.7)
CAP_W = 3000.0


def _monthly(samples: pd.DataFrame, nights_df: pd.DataFrame,
             windows: pd.DataFrame, recommended_kwh: float) -> pd.DataFrame:
    """Per-month medians the charts need."""
    s = samples.sort_values("ts_utc").reset_index(drop=True)
    delta = nights.consumption_increments(s)
    cum = np.concatenate([[0.0], delta.to_numpy().cumsum()])
    ts = s["ts_utc"].to_numpy()

    gen = s.groupby("solar_date")["energy_gen_wh"].max().rename("gen_wh").reset_index()
    gen["date"] = pd.to_datetime(gen["solar_date"])
    ws = windows.dropna(subset=["solar_start_utc", "solar_end_utc"])
    i = np.searchsorted(ts, ws["solar_start_utc"].to_numpy(), "left")
    j = np.searchsorted(ts, ws["solar_end_utc"].to_numpy(), "right")
    day = pd.DataFrame({"date": ws["date"].values, "day_cons_wh": cum[j] - cum[i]})

    n = nights_df[nights_df["covered"]].copy()
    n["date"] = pd.to_datetime(n["date"])
    m = gen.merge(day, on="date").merge(n[["date", "night_wh", "is_ev"]], on="date")
    m["month"] = m["date"].dt.month
    m["gen_kwh"] = m["gen_wh"] / 1000
    m["day_cons_kwh"] = m["day_cons_wh"] / 1000
    m["surplus_kwh"] = m["gen_kwh"] - m["day_cons_kwh"]
    m["night_kwh"] = m["night_wh"] / 1000
    out = m.groupby("month")[["gen_kwh", "day_cons_kwh", "surplus_kwh", "night_kwh"]].median()

    # share of night energy above the inverter cap, by month and EV class
    lo = np.searchsorted(ts, n["night_start_utc"].to_numpy(), "left")
    hi = np.searchsorted(ts, n["night_end_utc"].to_numpy(), "right")
    p = s["power_cons_w"].to_numpy()
    tot = {True: np.zeros(13), False: np.zeros(13)}
    above = {True: np.zeros(13), False: np.zeros(13)}
    for a, b, mo, is_ev in zip(lo, hi, pd.to_datetime(n["date"]).dt.month, n["is_ev"]):
        seg = p[a:b]; seg = seg[~np.isnan(seg)]
        tot[bool(is_ev)][mo] += seg.sum()
        above[bool(is_ev)][mo] += np.clip(seg - CAP_W, 0, None).sum()
    out["above_cap_pct_ev"] = [
        100 * above[True][k] / tot[True][k] if tot[True][k] else 0.0 for k in out.index]
    out["above_cap_pct_nonev"] = [
        100 * above[False][k] / tot[False][k] if tot[False][k] else 0.0 for k in out.index]

    # battery discharge by month at the recommended capacity
    net = battery.net_wh(s)
    night, nonev, month = battery.build_masks(s, nights_df)
    spec = battery.BatterySpec(np.array([recommended_kwh]), 3.0)
    r = battery.simulate(net, night, nonev, month, spec)
    out["discharge_kwh"] = r.monthly_discharge_wh[:, 0] / 1000
    return out.reset_index()


def _coverable_fraction(samples: pd.DataFrame, nights_df: pd.DataFrame,
                        windows: pd.DataFrame) -> float:
    """Fraction of covered nights whose energy that day's daytime surplus
    could have supplied, with an infinite battery and no power limit.

    This is the ceiling the winter months impose: no capacity can beat it.
    Computed rather than quoted, so the report's headline caveat cannot drift
    away from the data it describes.
    """
    s = samples.sort_values("ts_utc").reset_index(drop=True)
    cum = np.concatenate([[0.0], nights.consumption_increments(s).to_numpy().cumsum()])
    ts = s["ts_utc"].to_numpy()
    gen = s.groupby("solar_date")["energy_gen_wh"].max().rename("gen_wh").reset_index()
    gen["date"] = pd.to_datetime(gen["solar_date"])
    ws = windows.dropna(subset=["solar_start_utc", "solar_end_utc"])
    i = np.searchsorted(ts, ws["solar_start_utc"].to_numpy(), "left")
    j = np.searchsorted(ts, ws["solar_end_utc"].to_numpy(), "right")
    day = pd.DataFrame({"date": ws["date"].values, "day_cons_wh": cum[j] - cum[i]})

    n = nights_df[nights_df["covered"]].copy()
    n["date"] = pd.to_datetime(n["date"])
    m = gen.merge(day, on="date").merge(n[["date", "night_wh"]], on="date")
    surplus = m["gen_wh"] - m["day_cons_wh"]
    return float((surplus >= m["night_wh"]).mean())


def run(data_dir: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load(Path(data_dir))
    windows = pd.read_csv(
        Path(data_dir) / "out" / "solar_windows.csv",
        parse_dates=["date", "solar_start_utc", "solar_end_utc",
                     "night_start_utc", "night_end_utc"],
    )
    nights_df = nights.summarise_nights(samples, windows)
    sweep_df = battery.sweep(samples, nights_df, CAPACITIES, POWER_KWS)
    recommended = battery.recommend_capacity(sweep_df, power_kw=3.0)
    sens = nights.ev_sensitivity(samples, windows)
    monthly = _monthly(samples, nights_df, windows, recommended)

    nights_df.to_csv(out_dir / "night_summary.csv", index=False)
    sweep_df.to_csv(out_dir / "battery_sweep.csv", index=False)
    # Derived rather than hardcoded: the winter prose must move with the data.
    covered_nights = nights_df[nights_df["covered"]]
    coverable_pct = 100.0 * _coverable_fraction(samples, nights_df, windows)
    negative_surplus_months = [
        int(m) for m in monthly.loc[monthly["surplus_kwh"] < 0, "month"]
    ]
    (out_dir / "night_report.html").write_text(
        night_report.build_html(
            nights_df, sweep_df, monthly, sens, recommended,
            coverable_pct=coverable_pct,
            negative_surplus_months=negative_surplus_months,
        )
    )

    covered = nights_df[nights_df["covered"]]
    at = sweep_df[(sweep_df.power_kw == 3.0) &
                  np.isclose(sweep_df.capacity_kwh, recommended)].iloc[0]

    # Spec fact 9 is stated as an expectation the simulation must CONFIRM, not
    # as a finding: the 3 kW charge cap clips 15.6% of instantaneous surplus
    # flow in June, but the battery should still fill long before that costs
    # anything. Measure it rather than assume it — how much does going to a
    # 3.7 kW inverter actually buy at the recommended capacity?
    at37 = sweep_df[(sweep_df.power_kw == 3.7) &
                    np.isclose(sweep_df.capacity_kwh, recommended)].iloc[0]
    base = float(at["nonev_night_grid_import_kwh_yr"])
    better = float(at37["nonev_night_grid_import_kwh_yr"])
    power_gain = 100.0 * (base - better) / base if base else 0.0

    return {
        "n_nights": int(len(covered)),
        "n_ev_nights": int(covered["is_ev"].sum()),
        "median_night_kwh": float(covered["night_wh"].median() / 1000),
        "p90_night_kwh": float(covered["night_wh"].quantile(0.9) / 1000),
        "recommended_kwh": float(recommended),
        "night_self_sufficiency_pct": float(at["nonev_night_self_sufficiency_pct"]),
        "cycles_per_yr": float(at["cycles_per_yr"]),
        "pct_gain_from_3p7kw_inverter": power_gain,
    }


if __name__ == "__main__":
    here = Path(__file__).parent
    for k, v in run(here, here / "out").items():
        print(f"{k}: {v}")
