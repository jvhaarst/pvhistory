"""Is the chosen battery worth buying? Run: uv run python analyze_investment.py

Phase 4 sized a battery from measurements. This prices a specific purchase
against the alternative of leaving the money invested, using the saving curve
phase 4 measured and three rates the owner supplied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pvnight import economics, finance, investment_report

# Both lifetimes come from the datasheet, and they disagree about the
# verdict. The page leads with the warranty because it is the contractual
# floor, and publishes the design life beside it. An earlier draft led with
# 15 as a bare constant with no source -- the single choice that turned a
# loss into a win.
HEADLINE_YEARS = finance.WARRANTY_YEARS
HORIZONS = range(6, 26)


def saving_curve(repo_root: Path) -> pd.DataFrame:
    """Mean euro saving per capacity over the complete calendar years.

    Read from phase 4's output rather than recomputed, so this page cannot
    drift from the sizing page it depends on. Partial years are excluded
    there and stay excluded here.
    """
    e = pd.read_csv(repo_root / "out" / "meter_economics.csv")
    full = e[e["is_full_year"]]
    if full.empty:
        raise ValueError(
            "meter_economics.csv has no complete years — run analyze_meter.py")
    return (full.groupby("capacity_kwh", as_index=False)["saving_eur"]
            .mean().sort_values("capacity_kwh").reset_index(drop=True))


def run(repo_root: Path, out_dir: Path) -> dict:
    repo_root, out_dir = Path(repo_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = saving_curve(repo_root)
    # The saving the sizing page publishes, at full capacity and today's
    # prices. The tile must show this, not year one's inflated-and-degraded
    # figure, or the two pages appear to disagree about the same house.
    measured_saving = float(np.interp(finance.CAPACITY_KWH,
                                      curve["capacity_kwh"],
                                      curve["saving_eur"]))
    cf = finance.cashflows(curve, finance.CAPACITY_KWH, finance.COST_EUR,
                           HEADLINE_YEARS)
    sweep = finance.horizon_sweep(curve, finance.CAPACITY_KWH,
                                  finance.COST_EUR, HORIZONS)
    sens = finance.sensitivity(curve, finance.CAPACITY_KWH, finance.COST_EUR,
                               HEADLINE_YEARS)
    # Which limit actually binds: calendar life or rated cycles?
    sweep_csv = pd.read_csv(repo_root / "out" / "meter_battery_sweep.csv")
    row = sweep_csv[(sweep_csv.bound == "charge_first")
                    & (sweep_csv.power_kw == 3.0)
                    & (sweep_csv.capacity_kwh == 10.0)]
    cycle_years = finance.cycle_limited_years(float(row.cycles_per_yr.iloc[0]))

    cf.to_csv(out_dir / "investment_cashflow.csv", index=False)
    sweep.to_csv(out_dir / "investment_horizons.csv", index=False)
    sens.to_csv(out_dir / "investment_sensitivity.csv", index=False)
    (out_dir / "battery_investment.html").write_text(
        investment_report.build_html(
            finance.ASSUMPTIONS, cf, sweep, sens, HEADLINE_YEARS,
            finance.DISCOUNT, finance.BATTERY, finance.CAPACITY_KWH,
            finance.COST_EUR, measured_saving,
            purchase=finance.purchase_table(),
            fixed_cost_terms=economics.FIXED_COST_TERMS,
            warranty_years=finance.WARRANTY_YEARS,
            design_life_years=finance.DESIGN_LIFE_YEARS,
            cycle_years=cycle_years, vat_rate=finance.VAT_RATE))

    row = sweep[sweep["years"] == HEADLINE_YEARS].iloc[0]
    return {
        "battery": finance.BATTERY,
        "cost_eur": finance.COST_EUR,
        "measured_saving_eur": measured_saving,
        "achieved_return": float(
            sweep[sweep["years"] == HEADLINE_YEARS]["achieved_return"].iloc[0]),
        "npv_eur": float(row["npv_eur"]),
        "implied_return": float(row["implied_return"]),
        "break_even_year": finance.break_even_year(sweep),
        "headline_years": HEADLINE_YEARS,
        "design_life_years": finance.DESIGN_LIFE_YEARS,
        "npv_at_design_life_eur": float(
            sweep[sweep["years"] == finance.DESIGN_LIFE_YEARS]["npv_eur"].iloc[0]),
        "cycle_limited_years": round(cycle_years),
    }


if __name__ == "__main__":
    here = Path(__file__).parent
    for k, v in run(here, here / "out").items():
        print(f"{k}: {v}")
