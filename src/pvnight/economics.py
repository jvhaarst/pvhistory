"""The battery in euros.

Every capacity recommendation this project published before this module was
energy-only, which meant trading a stock (kWh bought) against a flow (kWh/yr
saved) with no exchange rate. Two attempts to supply one failed: an invented
50 kWh/yr threshold, removed at the user's instruction, and a geometric elbow
that turned out to drift with the sweep's truncation. A tariff supplies a
real rate, so this module is where the recommendation stops being arbitrary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import BatterySpec, simulate
from .meter import DT_HOURS
from .meter_battery import _masks, bound_signals
from .config import SITE_TZ
from .tariff import Tariff


def full_years(ts_utc, tolerance: float = 0.999) -> set[int]:
    """Calendar years the record covers end to end AND without holes.

    Two ways to not be a full year, and both have bitten this analysis:

    - **Span.** A first draft pinned the cutoff to a literal `2026-01-01`,
      which silently admitted 2019 — the record opens at 2019-12-31 23:15
      UTC, so that "year" holds three intervals and a EUR 0.03 bill, and
      averaging EUR/yr across it diluted every saving by a seventh.
    - **Holes.** Spanning a year is not covering it. 2024 spans fine but is
      missing 672 intervals — seven midwinter days from the January outage,
      in the season that dominates a battery answer. Counted as whole it
      understates that year's bill, and moves the published optimum by half
      a step.

    A year is complete when it holds at least `tolerance` of the intervals
    its length implies, at the record's own median spacing.
    """
    idx = pd.DatetimeIndex(ts_utc)
    lo, hi = idx.min(), idx.max()
    dt = pd.Series(idx).diff().median()
    if pd.isna(dt) or dt <= pd.Timedelta(0):
        return set()
    out = set()
    for y in range(int(lo.year), int(hi.year) + 1):
        start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        end = pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
        if not (lo <= start and hi >= end - dt):
            continue
        expected = (end - start) / dt
        got = int(((idx >= start) & (idx < end)).sum())
        if got >= tolerance * expected:
            out.add(y)
    return out


def year_band_index(ts_utc, tariff: Tariff) -> tuple[np.ndarray, list[int], int]:
    """One index encoding both the calendar year and the tariff band.

    Per-year figures could be had by simulating each year separately, but the
    battery's state of charge carries across a year boundary and phase 3's
    spec is explicit that the timeline is never broken up. Encoding
    ``year * n_bands + band`` into a single accumulator index gets per-year
    costs out of one continuous simulation instead.
    """
    idx = pd.DatetimeIndex(ts_utc)
    years = sorted({int(y) for y in idx.year})
    pos = {y: i for i, y in enumerate(years)}
    n_bands = len(tariff.bands)
    band = tariff.index_for(idx)
    year_pos = np.array([pos[int(y)] for y in idx.year])
    return year_pos * n_bands + band, years, n_bands


def price_capacities(meter_df: pd.DataFrame, nights_df: pd.DataFrame,
                     capacities_kwh: np.ndarray, tariff: Tariff,
                     power_kw: float = 3.0, round_trip: float = 0.90,
                     usable_fraction: float = 0.90) -> pd.DataFrame:
    """Cost of the variable electricity bill, per capacity and per year.

    Uses the charge-first ordering. Phase 3 measured the two within-interval
    orderings to differ by under 0.06% of baseline import and to agree on the
    elbow, so the choice does not move the answer.
    """
    m = meter_df.sort_values("ts_utc").reset_index(drop=True)
    night_i, nonev_i, month_i = _masks(m, nights_df)
    sig, src = bound_signals(m)["charge_first"]

    combo, years, n_bands = year_band_index(m["ts_utc"], tariff)
    complete = full_years(m["ts_utc"])
    spec = BatterySpec(np.asarray(capacities_kwh, float), power_kw,
                       round_trip, usable_fraction)
    r = simulate(sig, night_i[src], nonev_i[src], month_i[src], spec,
                 dt_hours=DT_HOURS, band_idx=combo[src],
                 n_bands=len(years) * n_bands)

    lev = tariff.bands["levering_eur_kwh"].to_numpy()
    net_exp = tariff.net_export_eur_kwh
    rows = []
    for yi, year in enumerate(years):
        sl = slice(yi * n_bands, (yi + 1) * n_bands)
        imp = r.band_grid_import_wh[sl] / 1000.0        # (n_bands, n_caps)
        exp = r.band_export_wh[sl] / 1000.0
        cost = ((imp * lev[:, None]).sum(axis=0)
                - (exp * net_exp[:, None]).sum(axis=0))
        for ci, cap in enumerate(spec.capacities_kwh):
            rows.append({
                "capacity_kwh": float(cap),
                "year": int(year),
                "is_full_year": int(year) in complete,
                "import_kwh": float(imp[:, ci].sum()),
                "export_kwh": float(exp[:, ci].sum()),
                "cost_eur": float(cost[ci]),
            })
    return pd.DataFrame(rows)


def savings(priced: pd.DataFrame) -> pd.DataFrame:
    """What each capacity saves against no battery, per year."""
    base = priced[priced["capacity_kwh"] == 0.0].set_index("year")["cost_eur"]
    if base.empty:
        raise ValueError(
            "savings: the sweep must include capacity 0.0 — it is the "
            "baseline every saving is measured against")
    out = priced.copy()
    out["saving_eur"] = out["year"].map(base).to_numpy() - out["cost_eur"]
    return out[["capacity_kwh", "year", "is_full_year", "cost_eur", "saving_eur"]]


def break_even_eur_per_kwh(saving_eur_per_yr: float, capacity_kwh: float,
                           years: float) -> float:
    """Installed cost per kWh at which this capacity exactly repays itself.

    Undiscounted, and degradation is not modelled — both stated on the page.
    """
    if capacity_kwh <= 0:
        return float("nan")
    return float(saving_eur_per_yr * years / capacity_kwh)


def payback_years(saving_eur_per_yr: float, capacity_kwh: float,
                  eur_per_kwh: float, fixed_cost_eur: float) -> float:
    if saving_eur_per_yr <= 0:
        return float("inf")
    return float((fixed_cost_eur + capacity_kwh * eur_per_kwh)
                 / saving_eur_per_yr)


def recommend_capacity_eur(annual: pd.DataFrame, eur_per_kwh: float,
                           fixed_cost_eur: float, years: float) -> float:
    """The capacity with the best undiscounted net position over ``years``.

    Not the largest saving: a bigger battery always saves more energy, so a
    rule that maximised saving would always recommend the largest capacity
    swept. Returns 0.0 when nothing pays back, which is a real answer.
    """
    a = annual.sort_values("capacity_kwh")
    net = (a["saving_eur"].to_numpy() * years
           - fixed_cost_eur - a["capacity_kwh"].to_numpy() * eur_per_kwh)
    if net.max() <= 0:
        return 0.0
    return float(a["capacity_kwh"].to_numpy()[int(np.argmax(net))])


def blended_value_eur_per_kwh(priced: pd.DataFrame, capacity_kwh: float) -> float:
    """Euros saved per kWh of grid import the battery actually displaced.

    Measured from the band mix the battery displaces rather than assumed,
    because displaced energy is not spread evenly across the bands.
    """
    s = savings(priced)
    row = s[s["capacity_kwh"] == capacity_kwh]
    base = priced[priced["capacity_kwh"] == 0.0]["import_kwh"].sum()
    here = priced[priced["capacity_kwh"] == capacity_kwh]["import_kwh"].sum()
    displaced = base - here
    if displaced <= 0:
        return float("nan")
    return float(row["saving_eur"].sum() / displaced)


def derived_threshold_kwh_per_kwh(eur_per_kwh: float, years: float,
                                  blended_value: float) -> float:
    """The kWh/yr a marginal kWh of capacity must return to be worth buying.

    This is the number ``battery.recommend_capacity`` was always missing. Its
    default of 50 was a judgement call the user had removed from the report;
    this replaces it with an arithmetic consequence of a real price.
    """
    if years <= 0 or blended_value <= 0:
        return float("nan")
    return float(eur_per_kwh / (years * blended_value))


def arbitrage_ceiling_eur_yr(meter_df: pd.DataFrame, tariff: Tariff,
                             capacity_kwh: float, power_kw: float = 3.0,
                             round_trip: float = 0.90,
                             usable_fraction: float = 0.90) -> dict:
    """An upper bound on what price-aware grid charging could add.

    Not a recommendation and not a dispatch policy. It answers one question:
    is a price-aware phase worth building at all?

    The bound is deliberately loose in three ways, all of which push it up,
    so a small number here is genuinely conclusive while a large one only
    means "worth investigating":

    - it assumes one full cycle every day at the day's widest band spread;
    - it ignores that the battery is already occupied storing solar surplus;
    - it uses perfect foresight, which no policy has.

    Hindsight is legitimate for a bound and illegitimate for a
    recommendation, so every caller must label it.
    """
    idx = pd.DatetimeIndex(meter_df["ts_utc"])
    band = tariff.index_for(idx)
    lev = tariff.bands["levering_eur_kwh"].to_numpy()[band]

    d = pd.DataFrame({"day": idx.tz_convert(SITE_TZ).normalize(), "price": lev})
    per_day = d.groupby("day")["price"].agg(["min", "max"])

    # Buying loses the round trip; displacing does not.
    spread = (per_day["max"] - per_day["min"] / round_trip).clip(lower=0.0)

    eta = float(np.sqrt(round_trip))
    usable_kwh = capacity_kwh * usable_fraction
    # A cycle is also capped by what the inverter can move in a day.
    movable_kwh = min(usable_kwh, power_kw * 24.0 * eta)

    total = float((spread * movable_kwh).sum())
    years = (idx.max() - idx.min()).total_seconds() / (365.25 * 24 * 3600)
    return {
        "ceiling_eur_yr": total / years if years > 0 else float("nan"),
        "best_spread_eur_kwh": float(spread.max()) if len(spread) else 0.0,
        "usable_days": int((spread > 0).sum()),
        "years": float(years),
    }


# Real quotes, not assumptions. Gathered 2026-09-05 from the retailers named,
# ex-VAT, and kept here as one dated table so the report derives every euro
# figure from them rather than repeating literals. Replace the rows when
# quotes change; nothing downstream needs editing.
QUOTES = pd.DataFrame([
    {"product": "BSL B-LFP48-100E 3U", "kwh": 5.12, "eur": 759.95,
     "source": "nkon.nl"},
    {"product": "BSL B-LFP48-200E", "kwh": 10.24, "eur": 1249.95,
     "source": "nkon.nl"},
    {"product": "Dyness PowerBrick Plus (low)", "kwh": 16.07, "eur": 2100.00,
     "source": "ess parts list"},
    {"product": "Dyness PowerBrick Plus (high)", "kwh": 16.07, "eur": 2650.00,
     "source": "ess parts list"},
]).assign(eur_per_kwh=lambda d: d["eur"] / d["kwh"])

# The hardware that does not scale with capacity, from the same parts list:
# MultiPlus-II 48/4k5 769.00, Class-T holder 42.50, Class-T fuse 45.00,
# three 35 mm2 lugs 5.94, two metres of 35 mm2 cable 19.38, DC isolator 50.00,
# VE.Bus cable 10.00. The Cerbo GX and the P1 feed are already owned.
#
# It does not move the optimum -- it is identical at every capacity, so it
# cancels in the argmax -- but it decides whether to build anything at all.
FIXED_COST_EUR = 941.82

# Which terms of that sum are sourced and which are not. The parts list gives
# per-unit prices; only some carry counts, so this names what is a reading and
# what is a judgement rather than letting EUR 941.82 look uniformly solid.
FIXED_COST_TERMS = pd.DataFrame([
    {"item": "MultiPlus-II 48/4k5/55-32", "eur": 769.00, "basis": "quoted"},
    {"item": "Class-T fuse holder 110-200 A", "eur": 42.50, "basis": "quoted"},
    {"item": "Class-T fuse 125 A", "eur": 45.00, "basis": "quoted"},
    {"item": "3 x 35 mm2 lug", "eur": 5.94, "basis": "quoted (list states M6x2 + M8x1)"},
    {"item": "2 m of 35 mm2 cable", "eur": 19.38, "basis": "ASSUMED length"},
    {"item": "DC isolator", "eur": 50.00, "basis": "ASSUMED needed (list says optional)"},
    {"item": "VE.Bus cable", "eur": 10.00, "basis": "ASSUMED (may ship with the Cerbo)"},
])

# Retailers quote ex-VAT; a household cannot reclaim it, so what is actually
# paid is the inclusive figure. Named here so a rate change is one edit.
VAT_RATE = 0.21


def incl_vat(eur: float, rate: float = VAT_RATE) -> float:
    return float(eur * (1.0 + rate))


def quote_table(annual: pd.DataFrame, fixed_cost_eur: float = FIXED_COST_EUR,
                years: tuple[float, ...] = (10.0, 15.0)) -> pd.DataFrame:
    """Each real quote against the measured savings curve.

    One row per product: its price per kWh, the capacity that price implies
    is optimal, what the product itself would save, and its payback including
    the fixed hardware.
    """
    rows = []
    for q in QUOTES.itertuples():
        saving = float(np.interp(q.kwh, annual["capacity_kwh"],
                                 annual["saving_eur"]))
        total_ex = q.eur + fixed_cost_eur
        total_in = incl_vat(total_ex)
        row = {"product": q.product, "kwh": q.kwh, "eur": q.eur,
               "eur_per_kwh": q.eur_per_kwh, "saving_eur_yr": saving,
               "total_ex_vat": total_ex, "total_incl_vat": total_in,
               "payback_yr": payback_years(saving, q.kwh, q.eur_per_kwh,
                                           fixed_cost_eur),
               "payback_yr_incl_vat": (total_in / saving if saving > 0
                                       else float("inf"))}
        for y in years:
            row[f"optimum_kwh_{y:.0f}yr"] = recommend_capacity_eur(
                annual, q.eur_per_kwh, fixed_cost_eur, y)
        rows.append(row)
    return pd.DataFrame(rows)
