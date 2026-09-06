"""Is buying this battery better than leaving the money invested elsewhere?

Every earlier phase asked what size to buy. This asks whether to buy at all,
for hardware already chosen, and it is a different kind of question: the
arithmetic is trivial and the assumptions decide everything. So the rates live
in one declared table, the headline is the return the investment implies
rather than a net present value computed at someone's guess, and the report
publishes what happens when each assumption moves.

The answer for this installation is not robust to the assumed lifetime, and
saying so is the point rather than an apology: over ten years the battery
returns about what the alternative does, and over fifteen it clearly wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import economics

# Every rate the model uses, declared once with its provenance. Nothing may
# reach the report except through here -- these three numbers decide the
# answer, so a reader must be able to see and challenge all of them at once.
DATASHEET = ("BSL B-LFP48-200PW datasheet, bsl-battery.com/uploads/f91d46eb.pdf, "
             "read 2026-09-06")

ASSUMPTIONS = pd.DataFrame([
    {"name": "degradation", "value": 0.015,
     "label": "battery capacity lost per year",
     "source": "owner's assumption, 2026-09-06"},
    {"name": "inflation", "value": 0.03,
     "label": "energy price rise per year",
     "source": "owner's assumption, 2026-09-06"},
    {"name": "discount", "value": 0.10,
     "label": "return available on the money elsewhere",
     "source": "owner's assumption, 2026-09-06"},
    # The horizon decides the verdict, so it belongs here with a citation
    # rather than sitting in the entry point as a bare constant. An earlier
    # draft used 15 years with no source, which is precisely the choice that
    # turns a EUR 40 loss into a EUR 724 win.
    {"name": "warranty_years", "value": 10.0,
     "label": "manufacturer warranty",
     "source": DATASHEET},
    {"name": "design_life_years", "value": 15.0,
     "label": "manufacturer design life, at 25 C",
     "source": DATASHEET},
    {"name": "cycle_life", "value": 6000.0,
     "label": "rated cycles",
     "source": DATASHEET},
])


def _rate(name: str) -> float:
    return float(ASSUMPTIONS.set_index("name").loc[name, "value"])


DEGRADATION = _rate("degradation")
INFLATION = _rate("inflation")
DISCOUNT = _rate("discount")
WARRANTY_YEARS = int(_rate("warranty_years"))
DESIGN_LIFE_YEARS = int(_rate("design_life_years"))
CYCLE_LIFE = _rate("cycle_life")

# The measured saving is priced against the 2027 tariff and the battery is
# installed in 2027, so year one is already at the measured price level and
# must not be inflated again. Inflation therefore runs from n-1, not n. An
# earlier draft used n and overstated the case by EUR 76 at ten years.
INSTALL_YEAR_OFFSET = 1

# The purchase, derived from phase 4's quote table and parts list rather than
# retyped, so the two pages cannot drift apart on what the hardware costs.
# The wall-mounted model, chosen 2026-09-06. Its datasheet is the
# source for WARRANTY_YEARS, DESIGN_LIFE_YEARS and CYCLE_LIFE above,
# so the lifetime figures and the price now describe one product.
BATTERY = "BSL B-LFP48-200PW"
_QUOTE = economics.QUOTES.set_index("product").loc[BATTERY]
CAPACITY_KWH = float(_QUOTE["kwh"])
BATTERY_EUR = float(_QUOTE["eur"])
FIXED_EUR = float(economics.FIXED_COST_EUR)
VAT_RATE = float(economics.VAT_RATE)
COST_EX_VAT_EUR = BATTERY_EUR + FIXED_EUR
COST_EUR = economics.incl_vat(COST_EX_VAT_EUR)


def purchase_table() -> pd.DataFrame:
    """What is being bought, itemised, ex- and incl-VAT.

    A household cannot reclaim VAT, so the inclusive figure is what the
    decision is actually made against; the exclusive one is shown because
    that is how retailers quote.
    """
    rows = [
        {"item": f"{BATTERY} — {CAPACITY_KWH:.2f} kWh", "ex_vat": BATTERY_EUR},
        {"item": "Victron inverter and DC hardware", "ex_vat": FIXED_EUR},
    ]
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"item": "total", "ex_vat": out["ex_vat"].sum()}
    out["incl_vat"] = out["ex_vat"] * (1.0 + VAT_RATE)
    return out


def cashflows(saving_curve: pd.DataFrame, capacity_kwh: float,
              cost_eur: float, years: int,
              degradation: float = DEGRADATION,
              inflation: float = INFLATION,
              discount: float = DISCOUNT) -> pd.DataFrame:
    """One row per year: what the battery saves, and what that is worth today.

    **Degradation is applied to the capacity and then looked up on the saving
    curve — it is never used to scale the saving.** The curve is concave, and
    a capacity past its elbow loses very little to early degradation, so
    scaling the euros instead would understate the early years badly. That
    asymmetry is a real property of an oversized battery and it only appears
    if the lookup is done properly.

    Everything is nominal: a nominal alternative return discounts nominal
    cash flows that inflate. Deflating the savings as well would count
    inflation twice.

    Inflation runs from `n - INSTALL_YEAR_OFFSET`. The saving was measured
    against the 2027 tariff and the battery is installed in 2027, so year one
    already sits at the measured price level; inflating it again would price
    2027 at 2028 rates.
    """
    n = np.arange(1, int(years) + 1)
    caps = capacity_kwh * (1.0 - degradation) ** n
    real = np.interp(caps, saving_curve["capacity_kwh"],
                     saving_curve["saving_eur"])
    nominal = real * (1.0 + inflation) ** (n - INSTALL_YEAR_OFFSET)
    factor = 1.0 / (1.0 + discount) ** n
    return pd.DataFrame({
        "year": n,
        "capacity_kwh": caps,
        "saving_eur": real,
        "nominal_eur": nominal,
        "discount_factor": factor,
        "present_value_eur": nominal * factor,
        "cost_eur": np.where(n == 1, cost_eur, 0.0),
    })


def npv(cf: pd.DataFrame) -> float:
    """Present value of the savings, less what the hardware cost."""
    return float(cf["present_value_eur"].sum() - cf["cost_eur"].sum())


def implied_return(saving_curve: pd.DataFrame, capacity_kwh: float,
                   cost_eur: float, years: int,
                   degradation: float = DEGRADATION,
                   inflation: float = INFLATION) -> float:
    """The discount rate at which this purchase exactly breaks even.

    Reported instead of a net present value so the owner's alternative return
    stays visible rather than buried inside the result: compare this number
    with whatever the money would otherwise earn. `nan` when no rate repays
    the cost, which is a real answer and not an error.
    """
    def f(r: float) -> float:
        return npv(cashflows(saving_curve, capacity_kwh, cost_eur, years,
                             degradation, inflation, r))

    # Searched over non-negative rates only. At sufficiently negative rates
    # the discount factor explodes and a root always exists, so an unbounded
    # search happily returns things like -79.6% -- arithmetic, not meaning.
    # f(0) is the undiscounted total: if that does not cover the cost, the
    # purchase never repays and `nan` says so.
    lo, hi = 0.0, 10.0
    if f(lo) < 0:
        return float("nan")
    if f(hi) > 0:
        return float("nan")
    for _ in range(60):   # [0,10] is exhausted to double precision by ~50
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def horizon_sweep(saving_curve: pd.DataFrame, capacity_kwh: float,
                  cost_eur: float, years=range(8, 26),
                  degradation: float = DEGRADATION,
                  inflation: float = INFLATION,
                  discount: float = DISCOUNT) -> pd.DataFrame:
    """NPV and implied return at every plausible lifetime.

    This is the deliverable rather than a single figure, because the assumed
    life decides the answer: on this installation the purchase is within a
    few tens of euros of its alternative at ten years and clearly ahead at
    fifteen. One NPV would hide that entirely.
    """
    rows = []
    for y in years:
        cf = cashflows(saving_curve, capacity_kwh, cost_eur, y,
                       degradation, inflation, discount)
        rows.append({
            "years": int(y),
            "npv_eur": npv(cf),
            "implied_return": implied_return(saving_curve, capacity_kwh,
                                             cost_eur, y, degradation,
                                             inflation),
            "achieved_return": achieved_return(saving_curve, capacity_kwh,
                                               cost_eur, y, degradation,
                                               inflation, discount),
        })
    return pd.DataFrame(rows)


def break_even_year(sweep: pd.DataFrame) -> int | None:
    """First horizon at which the purchase beats the alternative, or None.

    Sorts before reading. Taking `.iloc[0]` of an unsorted frame gave 19 on a
    shuffled sweep and 25 on a descending one, and the page's most quotable
    sentence rests on this. Returns None when the first row is already ahead,
    because "first" is then unknowable from the range examined rather than
    equal to its lowest row.
    """
    s = sweep.sort_values("years")
    ahead = s[s["npv_eur"] >= 0]
    if not len(ahead) or int(ahead["years"].iloc[0]) == int(s["years"].iloc[0]):
        return None
    return int(ahead["years"].iloc[0])


def sensitivity(saving_curve: pd.DataFrame, capacity_kwh: float,
                cost_eur: float, years: int,
                spans: dict[str, tuple[float, ...]] | None = None
                ) -> pd.DataFrame:
    """Move one assumption at a time, holding the other two at their declared
    values.

    Three guessed rates decide this answer, so the honest thing is to show
    how far each can move it rather than to defend any of them.
    """
    spans = spans or {
        "degradation": (0.005, 0.010, 0.015, 0.020, 0.030),
        "inflation": (0.00, 0.02, 0.03, 0.05, 0.07),
        "discount": (0.03, 0.05, 0.07, 0.10, 0.15),
    }
    base = {"degradation": DEGRADATION, "inflation": INFLATION,
            "discount": DISCOUNT}
    rows = []
    for rate, values in spans.items():
        for v in values:
            kw = dict(base, **{rate: v})
            cf = cashflows(saving_curve, capacity_kwh, cost_eur, years, **kw)
            rows.append({"rate": rate, "value": v, "npv_eur": npv(cf),
                         "is_base": bool(np.isclose(v, base[rate]))})

    # The assumed life belongs here too. The page calls it the input that
    # decides the answer, and an earlier version of this card left it out --
    # varying the three rates while omitting the one that flips the sign.
    for y in (8, 10, 12, 15, 20, 25):
        cf = cashflows(saving_curve, capacity_kwh, cost_eur, y, **base)
        rows.append({"rate": "life_years", "value": float(y), "npv_eur": npv(cf),
                     "is_base": y == years})
    return pd.DataFrame(rows)


def cycle_limited_years(cycles_per_yr: float,
                        cycle_life: float = CYCLE_LIFE) -> float:
    """Years before the rated cycle count is used up, at a measured rate.

    Published so a reader can see which limit binds. On this installation it
    is not the cycles: about 155 a year against 6,000 rated is roughly 39
    years, far beyond the calendar warranty and design life.
    """
    return float("inf") if cycles_per_yr <= 0 else float(cycle_life / cycles_per_yr)


def achieved_return(saving_curve: pd.DataFrame, capacity_kwh: float,
                    cost_eur: float, years: int,
                    degradation: float = DEGRADATION,
                    inflation: float = INFLATION,
                    reinvest_at: float = DISCOUNT) -> float:
    """What the money actually compounds at, if the savings are reinvested.

    The implied return (IRR) assumes each year's saving is reinvested at the
    IRR itself. Nobody can do that -- if it were available, it would be the
    alternative. This reinvests at the stated alternative rate instead, which
    is the comparison the owner actually faces, and is the standard MIRR.

    It is always the more conservative of the two when the IRR beats the
    alternative, and the two agree exactly when they are equal.
    """
    cf = cashflows(saving_curve, capacity_kwh, cost_eur, years,
                   degradation, inflation, reinvest_at)
    n = cf["year"].to_numpy()
    future_value = float((cf["nominal_eur"].to_numpy()
                          * (1.0 + reinvest_at) ** (years - n)).sum())
    if cost_eur <= 0 or future_value <= 0:
        return float("nan")
    return float((future_value / cost_eur) ** (1.0 / years) - 1.0)
