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
])


def _rate(name: str) -> float:
    return float(ASSUMPTIONS.set_index("name").loc[name, "value"])


DEGRADATION = _rate("degradation")
INFLATION = _rate("inflation")
DISCOUNT = _rate("discount")

# The purchase, derived from phase 4's quote table and parts list rather than
# retyped, so the two pages cannot drift apart on what the hardware costs.
BATTERY = "BSL B-LFP48-200E"
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
    """
    n = np.arange(1, int(years) + 1)
    caps = capacity_kwh * (1.0 - degradation) ** n
    real = np.interp(caps, saving_curve["capacity_kwh"],
                     saving_curve["saving_eur"])
    nominal = real * (1.0 + inflation) ** n
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
    for _ in range(200):
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
        })
    return pd.DataFrame(rows)


def break_even_year(sweep: pd.DataFrame) -> int | None:
    """First horizon at which the purchase beats the alternative, or None."""
    ahead = sweep[sweep["npv_eur"] >= 0]
    return int(ahead["years"].iloc[0]) if len(ahead) else None


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
    return pd.DataFrame(rows)
