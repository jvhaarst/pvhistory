"""The purchase decision, as a page.

Separate from `compare_report` on purpose: that page sizes a battery from
measurements, this one asks whether to buy a chosen one, and the two rest on
different kinds of input. Everything here traces back to three assumed rates,
and the page is built so a reader can see all three and watch them move the
answer rather than having to take the headline on trust.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .finance import break_even_year  # noqa: E402
from .night_report import TABLE_CSS  # noqa: E402
from .report import FURNITURE, SERIES, STYLE, _svg  # noqa: E402

RATE_LABEL = {"degradation": "battery degradation",
              "inflation": "energy price rise",
              "discount": "alternative return"}


def chart_horizon(sweep: pd.DataFrame, discount: float) -> str:
    """Net present value against assumed battery life.

    The deliverable of the whole page: the answer is decided by where this
    line crosses zero, not by any single year's figure.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    s = sweep.sort_values("years")
    ax.plot(s["years"], s["npv_eur"], lw=2, color=SERIES[0])
    ax.axhline(0, color=FURNITURE, lw=1)
    be = break_even_year(s)
    if be is not None:
        ax.axvline(be, color=SERIES[1], ls="--", lw=1)
        ax.annotate(f"breaks even at {be} years", xy=(be, 0),
                    xytext=(6, 12), textcoords="offset points",
                    fontsize=8, color=FURNITURE)
    ax.set_xlabel("assumed battery life (years)")
    ax.set_ylabel(f"net present value at {discount:.0%} (€)")
    return _svg(fig)


def chart_cashflow(cf: pd.DataFrame) -> str:
    """What each year contributes once discounted."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(cf["year"], cf["nominal_eur"], 0.75, color=SERIES[2],
           label="nominal saving")
    ax.bar(cf["year"], cf["present_value_eur"], 0.45, color=SERIES[0],
           label="worth today")
    ax.set_xlabel("year")
    ax.set_ylabel("€")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_sensitivity(sens: pd.DataFrame) -> str:
    """Each assumption moved alone, so their weights can be compared."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (rate, g) in enumerate(sens.groupby("rate")):
        g = g.sort_values("value")
        ax.plot(100 * g["value"], g["npv_eur"], lw=2, marker="o", ms=4,
                color=SERIES[i % len(SERIES)], label=RATE_LABEL.get(rate, rate))
        base = g[g["is_base"]]
        ax.scatter(100 * base["value"], base["npv_eur"], s=70,
                   facecolors="none", edgecolors=FURNITURE, zorder=3)
    ax.axhline(0, color=FURNITURE, lw=1)
    ax.set_xlabel("rate (%/yr) — circled point is the assumed value")
    ax.set_ylabel("net present value (€)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def _assumptions_table(assumptions: pd.DataFrame) -> str:
    head = "".join(f"<th>{h}</th>" for h in ["assumption", "value", "source"])
    rows = "".join(
        f"<tr><td>{a.label}</td><td>{a.value:.1%} per year</td>"
        f"<td>{a.source}</td></tr>" for a in assumptions.itertuples())
    return (f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")



def _purchase_table(purchase: pd.DataFrame, vat_rate: float) -> str:
    head = "".join(f"<th>{h}</th>" for h in
                   ["what is bought", "ex-VAT", f"incl. {vat_rate:.0%} VAT"])
    rows = "".join(
        (f"<tr><td><strong>{r.item}</strong></td>"
         f"<td><strong>€{r.ex_vat:,.2f}</strong></td>"
         f"<td><strong>€{r.incl_vat:,.2f}</strong></td></tr>"
         if r.item == "total" else
         f"<tr><td>{r.item}</td><td>€{r.ex_vat:,.2f}</td>"
         f"<td>€{r.incl_vat:,.2f}</td></tr>")
        for r in purchase.itertuples())
    return (f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def _inputs_card(purchase: pd.DataFrame, assumptions: pd.DataFrame,
                 vat_rate: float, measured_saving: float,
                 capacity_kwh: float, headline_years: int) -> str:
    """Everything that goes in, before anything that comes out.

    The page previously opened with its conclusions and left the three rates
    that determine them buried in the third card. A reader cannot judge a
    verdict without first seeing what it was computed from.
    """
    return (
        '<p class="sub">Two kinds of input, and they are not equally solid. '
        "The prices are quotes and the saving is measured; the three rates "
        "below are assumptions, and they decide the answer.</p>"
        + _purchase_table(purchase, vat_rate)
        + '<p class="sub">The saving comes from the meter-based sizing '
        f"analysis: <strong>€{measured_saving:,.0f} per year</strong> at "
        f"{capacity_kwh:.2f} kWh, at today's prices, averaged over the "
        "complete calendar years in the record. It is read from that page's "
        "output rather than recomputed here, so the two cannot disagree.</p>"
        + _assumptions_table(assumptions)
        + '<p class="sub">Everything is nominal: a nominal alternative '
        "return discounts nominal savings that inflate, so deflating the "
        "savings as well would count inflation twice. The battery is assumed "
        f"worthless after {headline_years} years and no residual value is "
        "credited.</p>")


def _horizon_table(sweep: pd.DataFrame, discount: float) -> str:
    head = "".join(f"<th>{h}</th>" for h in
                   ["assumed life", f"NPV at {discount:.0%}", "implied return"])
    rows = "".join(
        f"<tr><td>{r.years} yr</td><td>€{r.npv_eur:,.0f}</td>"
        f"<td>{r.implied_return:.1%}</td></tr>"
        for r in sweep.itertuples() if r.years % 2 == 0)
    return (f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def _verdict(sweep: pd.DataFrame, headline_years: int, discount: float,
             battery: str, capacity_kwh: float, cost_eur: float) -> str:
    """The answer, with its own fragility stated in the same breath."""
    row = sweep[sweep["years"] == headline_years].iloc[0]
    be = break_even_year(sweep)
    verdict = "beats" if row.npv_eur >= 0 else "loses to"
    return (
        f"A <strong>{battery}</strong> at {capacity_kwh:.2f} kWh, with the "
        f"inverter and DC hardware, costs <strong>€{cost_eur:,.0f}</strong> "
        "including VAT. Over an assumed "
        f"<strong>{headline_years} year</strong> life it {verdict} leaving "
        f"the money invested at {discount:.0%}, by "
        f"<strong>€{abs(row.npv_eur):,.0f}</strong> in today's money, and "
        f"implies a return of <strong>{row.implied_return:.1%} per year</strong>."
        + (f" It first pulls ahead at <strong>{be} years</strong>."
           if be is not None else
           " It never pulls ahead within the range examined.")
        + " Compare that implied return against what the money would actually "
        "earn: it is stated this way so the "
        f"{discount:.0%} assumption stays visible rather than being buried "
        "inside a single number.")


def build_html(assumptions: pd.DataFrame, cf: pd.DataFrame,
               sweep: pd.DataFrame, sens: pd.DataFrame, headline_years: int,
               discount: float, battery: str, capacity_kwh: float,
               cost_eur: float, saving_year_one: float,
               purchase: pd.DataFrame, vat_rate: float = 0.21) -> str:
    """Assemble the page. No document wrapper — the host supplies it.

    `purchase` is required, not optional. An earlier draft let it default to
    None, which meant the page could render with its verdict intact and its
    assumptions missing entirely — the reader would see a confident 14.2%
    with nothing to judge it against. A page that can omit its own inputs is
    the wrong shape for this question.
    """
    be = break_even_year(sweep)
    stats = [
        (f"{sweep[sweep.years == headline_years].implied_return.iloc[0]:.1%}",
         f"implied return over {headline_years} years"),
        (f"€{cost_eur:,.0f}", "total cost including VAT"),
        (f"€{saving_year_one:,.0f}",
         f"saved per year at {capacity_kwh:.2f} kWh, today's prices"),
        (f"{be} yr" if be is not None else "never",
         f"before it beats {discount:.0%} elsewhere"),
    ]
    stat_html = "".join(
        f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)

    cards = [
        ("What goes in",
         _inputs_card(purchase, assumptions, vat_rate, saving_year_one,
                      capacity_kwh, headline_years),
         ""),
        ("The verdict",
         _verdict(sweep, headline_years, discount, battery, capacity_kwh,
                  cost_eur),
         chart_horizon(sweep, discount) + _horizon_table(sweep, discount)),
        ("Year by year",
         "The saving rises with energy prices and falls as the battery "
         "ages, and discounting pulls the later years down hardest. "
         "Degradation is applied to the <em>capacity</em> and then read off "
         "the measured saving curve, never used to scale the euros: the "
         "curve is concave and this battery sits past its elbow, so the "
         "first years lose very little to it.",
         chart_cashflow(cf)),
        ("How much the assumptions matter",
         "Each rate moved alone, the other two held at their assumed "
         "values. The circled point on each line is the assumption in the "
         "table above. A line that crosses zero inside its plausible range "
         "is an assumption the answer genuinely depends on.",
         chart_sensitivity(sens)),
        ("What this does not include",
         "No residual value: the battery is assumed worthless at the end of "
         "the horizon, which is deliberately conservative — at 1.5% a year "
         "it still holds about four fifths of its capacity after fifteen "
         "years. No maintenance, no inverter replacement, no insurance, and "
         "no subsidy. The saving itself comes from the meter-based analysis "
         "and inherits its assumptions, including that the tariff's shape "
         "holds: the entire case rests on exported energy earning about a "
         "cent while imported energy costs 17 to 30 times that.",
         ""),
    ]
    card_html = "".join(
        f'<section class="card"><h2>{t}</h2><p>{c}</p>'
        + (f'<div class="chart">{svg}</div>' if svg else "")
        + "</section>"
        for t, c, svg in cards)

    return (
        "<title>Is the Battery Worth Buying?</title>"
        + STYLE
        + TABLE_CSS
        + "<main><h1>Is the battery worth buying?</h1>"
        + '<p class="lede">Every earlier page asked what size to buy. This '
        "one asks whether to buy at all, for hardware already chosen, and "
        "compares it against leaving the money invested elsewhere. The "
        "arithmetic is simple; three assumed rates decide the answer, and "
        "the assumed battery life decides it most of all.</p>"
        + f'<div class="stats">{stat_html}</div>'
        + card_html + "</main>")
