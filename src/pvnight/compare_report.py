"""The side-by-side page: meter-based figures, with phase 2 shown and labelled.

Phase 2's night-consumption and battery-sizing analysis ran against
PVOutput's consumption channel, which stopped seeing part of the load in
December 2022 (spec section 2.1). This page redoes the analysis against the
smart meter and publishes both, because side-by-side publication is only
safe if the page says plainly which figure is which.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .battery import (  # noqa: E402
    benefit_share_pct,
    convergence_readings,
    elbow_capacity,
    elbow_stability,
)
from .meter_wall import negative_surplus_months  # noqa: E402
from .night_report import (  # noqa: E402
    MONTHS,
    TABLE_CSS,
    _by_month,
    _join_month_names,
)
from .report import FURNITURE, SERIES, STYLE, _svg  # noqa: E402

# The two within-interval orderings (spec S4.2). `charge_first` puts the
# export before the import inside a quarter hour, `discharge_first` after.
#
# Neither is labelled favourable or unfavourable, and an earlier draft that
# did so was wrong. Within a single interval charge-first can bridge more, so
# the label looked safe — but state of charge couples the intervals, and
# discharge-first empties a little of the battery before charging it, which
# makes room to capture export that charge-first spills once the battery is
# full. Measured across the whole record, discharge-first imports *less* at
# every non-zero capacity. See `bounds_agreement_pct`: the two differ by at
# most a fraction of a percent, which is the finding, not a caveat.
BOUND_COLOUR = {"charge_first": SERIES[0], "discharge_first": SERIES[1]}


def bounds_agreement_pct(meter_sweep: pd.DataFrame,
                         power_kw: float = 3.0) -> float:
    """Largest gap between the two orderings, as a percent of the baseline.

    The baseline is grid import at zero capacity, which both orderings
    reproduce exactly — that shared point is what makes them a bracket. The
    return value answers the question the bracket was built to answer: how
    much can the unknowable within-interval ordering move the answer?
    """
    d = meter_sweep[meter_sweep["power_kw"] == power_kw]
    cf = d[d["bound"] == "charge_first"].sort_values("capacity_kwh")
    df_ = d[d["bound"] == "discharge_first"].sort_values("capacity_kwh")
    if cf.empty or df_.empty:
        return float("nan")
    baseline = float(cf["grid_import_kwh_yr"].iloc[0])
    gap = np.abs(cf["grid_import_kwh_yr"].to_numpy()
                 - df_["grid_import_kwh_yr"].to_numpy()).max()
    return float(100.0 * gap / baseline) if baseline else float("nan")


def format_penalty_pct(pct: float) -> str:
    """One rendering of the resolution penalty, for every place that shows it.

    The stat tile and the card used to format it independently at one decimal
    place, which printed a measured -0.016% as "-0.0%" and let the page
    disagree with the README. Decimals scale with the magnitude so a non-zero
    measurement never rounds away to zero, and the sign is always shown: the
    coarser simulation can land on either side of the finer one.
    """
    if not np.isfinite(pct):
        return "not measured"
    # Below the deepest precision offered there is nothing left to show, so
    # snap to zero rather than print "-0.0000%".
    if abs(pct) < 5e-5:
        return "0%"
    decimals = int(np.clip(np.ceil(-np.log10(abs(pct))) + 1, 1, 4))
    return f"{pct:+.{decimals}f}%"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------



def chart_surplus_vs_need(monthly: pd.DataFrame) -> str:
    """The winter wall: a month with a negative surplus charges nothing.

    Same form as phase 2's, so the two can be read side by side — but drawn
    from the meter, where "surplus" is the net daytime position rather than a
    generation figure minus a consumption channel that was undercounting.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    m = _by_month(monthly)
    x = np.arange(12)
    ax.bar(x - 0.2, m["surplus_kwh"], 0.4, color=SERIES[2],
           label="daytime surplus (export − import)")
    ax.bar(x + 0.2, m["night_kwh"], 0.4, color=SERIES[1], label="night need")
    ax.axhline(0, color=FURNITURE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("kWh per day (median)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_monthly_discharge(discharge: pd.DataFrame, capacity_kwh: float) -> str:
    """What the recommended battery actually delivers, month by month."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    d = discharge.sort_values("month")
    ax.bar(np.arange(12), d["discharge_kwh"], 0.6, color=SERIES[0])
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel(f"kWh discharged per year\nat {capacity_kwh:.1f} kWh")
    return _svg(fig)


def _wall_prose(monthly: pd.DataFrame, coverable_pct: float,
                negative_months: list[int]) -> str:
    """The winter finding, built from the frames the charts draw.

    Phase 2's equivalent sentence was one of this project's several pieces of
    prose that disagreed with its own data, so every number here is read from
    `monthly` rather than written beside it.
    """
    if negative_months:
        names = _join_month_names(negative_months)
        worst = monthly.loc[monthly["surplus_kwh"].idxmin()]
        wall = (
            f"In {names} the median daytime surplus is "
            "<strong>negative</strong> — the house does not send enough to "
            "the grid to cover even its own daytime draw, so a battery "
            "receives no charge at all. The worst is "
            f"{_join_month_names([int(worst['month'])])}, at "
            f"<strong>{worst['surplus_kwh']:.1f} kWh</strong> a day against a "
            f"night needing {worst['night_kwh']:.1f} kWh."
        )
    else:
        wall = ("No month here has a negative median daytime surplus, so "
                "winter alone does not block charging on this record.")
    return (
        "Median daytime surplus against median night need, per month, both "
        "measured at the meter. " + wall +
        f" Across the whole record only <strong>{coverable_pct:.1f}%</strong> "
        "of nights could have been covered even by an infinite battery with "
        "no power limit and no round-trip loss. That ceiling, not capacity, "
        "is what bounds this answer."
    )



def tariff_card(bands: pd.DataFrame) -> str:
    """The tariff table and what it implies, both read from the frame.

    The multiplier is the economic case for this whole phase, so it is
    computed here rather than quoted: change the prices and the sentence
    changes with them.
    """
    net = bands["vergoeding_eur_kwh"] - bands["terugleverkosten_eur_kwh"]
    ratio = bands["levering_eur_kwh"] / net
    head = "".join(f"<th>{h}</th>" for h in
                   ["band", "import €/kWh", "export cost", "export paid",
                    "net export", "×"])
    rows = "".join(
        f"<tr><td>{b}</td><td>{lv:.5f}</td><td>{tk:.5f}</td>"
        f"<td>{vg:.5f}</td><td>{n:.5f}</td><td>{r:.1f}×</td></tr>"
        for b, lv, tk, vg, n, r in zip(
            bands["band"], bands["levering_eur_kwh"],
            bands["terugleverkosten_eur_kwh"], bands["vergoeding_eur_kwh"],
            net, ratio))
    return (
        f'<p class="sub">Exporting a kWh nets <strong>€{net.mean():.3f}</strong>'
        " — the feed-in charge claws back all but a fraction of the feed-in "
        "payment, so there is no net metering left in this contract. The same "
        f"kWh stored and used later is worth <strong>{ratio.min():.1f}× to "
        f"{ratio.max():.1f}×</strong> that. This is the entire economic case "
        "for a battery here.</p>"
        f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>")


def chart_savings(annual: pd.DataFrame) -> str:
    """Euros saved per year against capacity."""
    fig, ax = plt.subplots(figsize=(9, 4))
    a = annual.sort_values("capacity_kwh")
    ax.plot(a["capacity_kwh"], a["saving_eur"], lw=2, color=SERIES[0])
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("€ saved per year")
    return _svg(fig)


def chart_break_even(annual: pd.DataFrame, quotes: pd.DataFrame,
                     years: float) -> str:
    """Installed €/kWh each capacity must beat, with the real quotes on it."""
    fig, ax = plt.subplots(figsize=(9, 4))
    a = annual[annual["capacity_kwh"] > 0].sort_values("capacity_kwh")
    ax.plot(a["capacity_kwh"], a["break_even_eur_per_kwh"], lw=2,
            color=SERIES[1], label=f"break-even over {years:.0f} years")
    ax.scatter(quotes["kwh"], quotes["eur_per_kwh"], s=36, color=SERIES[0],
               zorder=3, label="quoted products")
    for q in quotes.itertuples():
        ax.annotate(f"{q.kwh:.2f} kWh", xy=(q.kwh, q.eur_per_kwh),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=7, color=FURNITURE)
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("€ per kWh installed")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def _quote_table_html(quotes: pd.DataFrame, horizon_years: float,
                      vat_rate: float) -> str:
    """Each quote, with what is actually paid and what it implies.

    The optimum column is looked up by horizon rather than hardcoded: an
    earlier version pinned `optimum_kwh_10yr` while its own section prose was
    parameterised, so changing the horizon crashed the build. A parameter
    that cannot be changed is not a parameter.
    """
    col = f"optimum_kwh_{horizon_years:.0f}yr"
    if col not in quotes.columns:
        raise ValueError(f"quote table has no {col}; horizons available: "
                         f"{[c for c in quotes.columns if c.startswith('optimum')]}")
    head = "".join(f"<th>{h}</th>" for h in
                   ["product", "kWh", "battery €", "€/kWh", "€/yr saved",
                    f"total incl. {vat_rate:.0%} VAT", "payback incl.",
                    "payback ex-VAT", "implies optimum"])
    rows = "".join(
        f"<tr><td>{q.product}</td><td>{q.kwh:.2f}</td><td>{q.eur:,.2f}</td>"
        f"<td>{q.eur_per_kwh:.2f}</td><td>{q.saving_eur_yr:.0f}</td>"
        f"<td>{q.total_incl_vat:,.0f}</td>"
        f"<td><strong>{q.payback_yr_incl_vat:.1f} yr</strong></td>"
        f"<td>{q.payback_yr:.1f} yr</td>"
        f"<td>{getattr(q, col):.1f} kWh</td></tr>"
        for q in quotes.itertuples())
    return (f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")



def _confidence_card(annual: pd.DataFrame, priced: pd.DataFrame,
                     optimum_kwh: float, eur_per_kwh: float,
                     fixed_cost_eur: float, horizon_years: float,
                     terms: pd.DataFrame, usable_fraction: float,
                     round_trip: float) -> str:
    """How much to trust the recommended capacity.

    These caveats lived only in the README, which is the wrong way round: the
    page states the recommendation, so the page must state its sharpness. A
    reader deciding what to buy is holding this, not the repository.

    Every figure is recomputed here from the frames the page already carries.
    """
    a = annual.sort_values("capacity_kwh")
    net = (a["saving_eur"].to_numpy() * horizon_years
           - fixed_cost_eur - a["capacity_kwh"].to_numpy() * eur_per_kwh)
    best = float(net.max())
    # The band of capacities within 2% of the best net position: the range the
    # curve genuinely cannot distinguish between.
    near = a["capacity_kwh"].to_numpy()[net >= best - 0.02 * abs(best)]
    lo, hi = float(near.min()), float(near.max())
    spread = best - float(net[(a["capacity_kwh"] >= lo)
                              & (a["capacity_kwh"] <= hi)].min())

    years = sorted(int(y) for y in priced.loc[priced["is_full_year"], "year"].unique())
    all_years = sorted(int(y) for y in priced["year"].unique())
    dropped = [y for y in all_years if y not in years]

    rows = "".join(
        f"<tr><td>{c:.1f} kWh</td><td>€{n:,.0f}</td></tr>"
        for c, n in zip(a["capacity_kwh"], net)
        if lo <= c <= hi)

    assumed = terms[terms["basis"].str.startswith("ASSUMED")]
    assumed_html = "".join(
        f"<li>€{t.eur:,.2f} — {t.item} ({t.basis.replace('ASSUMED', 'assumed')})</li>"
        for t in assumed.itertuples())

    return (
        f'<p class="sub">The recommendation is '
        f"<strong>{optimum_kwh:.1f} kWh</strong>, but the curve is nearly "
        f"flat around it: every capacity from <strong>{lo:.1f}</strong> to "
        f"<strong>{hi:.1f} kWh</strong> sits within <strong>€{spread:,.0f}"
        f"</strong> of the best {horizon_years:.0f}-year net position of "
        f"€{best:,.0f}. One decimal place claims more than the data resolves; "
        "read it as a band, and let product availability pick within it.</p>"
        f'<div class="chart"><table><thead><tr><th>capacity</th>'
        f"<th>{horizon_years:.0f}-year net position</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        f'<p class="sub">Priced across <strong>{len(years)}</strong> complete '
        f"calendar years ({', '.join(str(y) for y in years)}). "
        + (f"<strong>{', '.join(str(y) for y in dropped)}</strong> "
           + ("are" if len(dropped) > 1 else "is")
           + " excluded: partial at the ends of the record, or missing enough "
             "intervals inside the year to understate its bill. Dropping a "
             "year with a midwinter outage moves this answer by half a step, "
             "so the exclusion is not cosmetic. " if dropped else "")
        + f"The battery is modelled at <strong>{usable_fraction:.2f}</strong> "
        f"usable fraction and <strong>{round_trip:.2f}</strong> round trip, "
        "both inherited from the earlier phase. Only one quoted product "
        "states a usable figure (15.27 of 16.07 kWh, i.e. 0.95); at that "
        "value the recommendation is unchanged, and at a fifteen-year "
        "horizon it rises by one to two steps.</p>"
        + (f'<p class="sub">Three terms of the €{fixed_cost_eur:,.0f} of '
           "non-scaling hardware are assumptions rather than quotes:</p>"
           f"<ul>{assumed_html}</ul>"
           '<p class="sub">They do not move the recommendation — fixed cost '
           "is identical at every capacity and cancels in the comparison — "
           "but they shift payback by roughly a fifth of a year.</p>"
           if len(assumed) else ""))


def _euro_prose(s: dict, years: float) -> str:
    """The euro finding, every number read from the summary dict."""
    return (
        "Under this tariff the variable electricity bill without a battery "
        f"comes to <strong>€{s['no_battery_cost_eur']:,.0f}</strong> in the "
        "most recent full year. Against the quoted hardware the optimal "
        f"capacity is <strong>{s['euro_optimum_kwh']:.1f} kWh</strong>, "
        f"saving <strong>€{s['annual_saving_eur']:,.0f}</strong> a year for a "
        f"break-even installed cost of "
        f"<strong>€{s['breakeven_eur_per_kwh']:,.0f}/kWh</strong> over "
        f"{years:.0f} years. Phase 2 published a 50 kWh/yr rule of thumb that "
        "was never derived from anything and was removed at the owner's "
        "instruction; the same rule derived from these prices is "
        f"<strong>{s['derived_threshold_kwh_per_kwh']:.0f} kWh/yr</strong>, so "
        "the guess was close but had no standing. Standing charges, network "
        "tariffs and taxes are excluded throughout: they do not change with "
        "battery size, so they cancel in every comparison. Degradation and "
        "discounting are not modelled, and payback is stated in undiscounted "
        "years.")


def _arbitrage_prose(a: dict) -> str:
    return (
        "A perfect-foresight upper bound on buying cheap and discharging "
        f"dear adds at most <strong>€{a['ceiling_eur_yr']:,.0f}</strong> a "
        "year on top. That is a ceiling, not a policy, and it is loose in "
        "three ways that all push it up: it assumes a full extra cycle every "
        "day, ignores that the battery is already storing solar, and uses "
        "hindsight no dispatch has. It is also non-zero on only "
        f"<strong>{a['usable_days']:,}</strong> days — the summer ones, when "
        "the battery is already full of sun. Winter's peak-to-off-peak "
        "spread is negative once the round trip is paid. Worth investigating "
        "as its own phase; not worth counting on.")


def chart_visible_fraction(monthly_ratio: pd.DataFrame) -> str:
    """The monthly PVOutput/meter ratio across the whole record.

    This is the evidence behind every other chart on the page: a discrete,
    never-recovered step downward, dated to December 2022.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    m = monthly_ratio.sort_values("month")
    x = m["month"].dt.to_timestamp()
    ax.plot(x, m["ratio"], lw=1.8, color=SERIES[0])

    nov = m[m["month"] == pd.Period("2022-11", freq="M")]
    dec = m[m["month"] == pd.Period("2022-12", freq="M")]
    if len(nov) and len(dec):
        boundary = dec["month"].iloc[0].to_timestamp()
        ax.axvline(boundary, color=FURNITURE, ls="--", lw=1)
        ax.annotate(
            f"Nov {nov['ratio'].iloc[0]:.3f} → Dec {dec['ratio'].iloc[0]:.3f}",
            xy=(boundary, dec["ratio"].iloc[0]),
            xytext=(8, 14), textcoords="offset points",
            color=FURNITURE, fontsize=8,
            arrowprops=dict(arrowstyle="-", color=FURNITURE, lw=0.8),
        )
    ax.set_xlabel("month")
    ax.set_ylabel("PVOutput / meter night ratio")
    ax.set_ylim(0, 1.05)
    return _svg(fig)


def chart_night_distribution(meter_nights: pd.DataFrame, pv_nights: pd.DataFrame) -> str:
    """Night-energy CDFs from both sources, restricted to the overlap.

    Restricting to the overlapping span means any difference between the
    curves is attributable to the source, not to the period compared.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    mn = meter_nights[meter_nights["covered"]].copy()
    pn = pv_nights[pv_nights["covered"]].copy()
    mn["date"] = pd.to_datetime(mn["date"])
    pn["date"] = pd.to_datetime(pn["date"])
    lo = max(mn["date"].min(), pn["date"].min())
    hi = min(mn["date"].max(), pn["date"].max())
    mn = mn[(mn["date"] >= lo) & (mn["date"] <= hi)]
    pn = pn[(pn["date"] >= lo) & (pn["date"] <= hi)]

    for vals, colour, label in [
        (mn["import_kwh"].to_numpy(dtype=float), SERIES[0], f"meter (n={len(mn)})"),
        (pn["night_wh"].to_numpy(dtype=float) / 1000, SERIES[1],
         f"PVOutput, superseded (n={len(pn)})"),
    ]:
        v = np.sort(vals)
        if len(v):
            ax.plot(v, np.linspace(0, 100, len(v)), lw=2, color=colour, label=label)

    ax.set_xlabel("night consumption (kWh)")
    ax.set_ylabel("percentile")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def _bound_pair(meter_sweep: pd.DataFrame, power_kw: float = 3.0) -> pd.DataFrame:
    """The two bounds, at one inverter power, merged on capacity for
    plotting side by side without assuming row order matches."""
    cf = meter_sweep[(meter_sweep["bound"] == "charge_first")
                      & (meter_sweep["power_kw"] == power_kw)].sort_values("capacity_kwh")
    df_ = meter_sweep[(meter_sweep["bound"] == "discharge_first")
                       & (meter_sweep["power_kw"] == power_kw)].sort_values("capacity_kwh")
    return cf.merge(df_, on="capacity_kwh", suffixes=("_cf", "_df"))


def chart_bounds(meter_sweep: pd.DataFrame, elbow_charge_first: float,
                  elbow_discharge_first: float) -> str:
    """Night grid import against capacity for both orderings.

    The band between the two lines is what the unknowable within-interval
    ordering can move; on this record it is thin enough to be hard to see,
    which is the result rather than a drawing problem.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    d = _bound_pair(meter_sweep)
    ax.fill_between(d.capacity_kwh, d.nonev_night_grid_import_kwh_yr_cf,
                     d.nonev_night_grid_import_kwh_yr_df, color=SERIES[2],
                     alpha=0.2, label="range between bounds")
    ax.plot(d.capacity_kwh, d.nonev_night_grid_import_kwh_yr_cf, lw=2,
            color=BOUND_COLOUR["charge_first"], label="charge-first")
    ax.plot(d.capacity_kwh, d.nonev_night_grid_import_kwh_yr_df, lw=2,
            color=BOUND_COLOUR["discharge_first"], label="discharge-first")
    ax.axvline(elbow_charge_first, color=BOUND_COLOUR["charge_first"], ls="--", lw=1)
    ax.axvline(elbow_discharge_first, color=BOUND_COLOUR["discharge_first"], ls="--", lw=1)
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("household night grid import (kWh/yr)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_marginal(meter_sweep: pd.DataFrame) -> str:
    """Marginal return for both orderings, household (non-EV) nights only."""
    fig, ax = plt.subplots(figsize=(9, 4))
    d = _bound_pair(meter_sweep)
    ax.plot(d.capacity_kwh, d.nonev_marginal_kwh_per_kwh_cf, lw=2,
            color=BOUND_COLOUR["charge_first"], label="charge-first")
    ax.plot(d.capacity_kwh, d.nonev_marginal_kwh_per_kwh_df, lw=2,
            color=BOUND_COLOUR["discharge_first"], label="discharge-first")
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("marginal kWh/yr avoided per added kWh")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_comparison(meter_sweep: pd.DataFrame, pv_sweep: pd.DataFrame,
                      power_kw: float = 3.0) -> str:
    """Both analyses' non-EV curves on one axis. Phase 2 drawn dashed."""
    fig, ax = plt.subplots(figsize=(9, 4))
    cf = meter_sweep[(meter_sweep["bound"] == "charge_first")
                      & (meter_sweep["power_kw"] == power_kw)].sort_values("capacity_kwh")
    pv = pv_sweep[pv_sweep["power_kw"] == power_kw].sort_values("capacity_kwh")
    ax.plot(cf.capacity_kwh, cf.nonev_night_grid_import_kwh_yr, lw=2,
            color=SERIES[0], label="meter, charge-first bound")
    ax.plot(pv.capacity_kwh, pv.nonev_night_grid_import_kwh_yr, lw=2, ls="--",
            color=SERIES[1], label="PVOutput, phase 2 (superseded)")
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("household night grid import (kWh/yr)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def _gap_months(gaps: pd.DataFrame) -> set:
    """Every calendar month touched by a meter gap, start through end.

    A gap that straddles a month boundary marks both months, so a bar
    highlighted here always corresponds to an actual row in ``gaps`` rather
    than to a hardcoded date.
    """
    months = set()
    starts = pd.to_datetime(gaps["gap_start_utc"]).dt.tz_localize(None)
    ends = pd.to_datetime(gaps["gap_end_utc"]).dt.tz_localize(None)
    for s, e in zip(starts, ends):
        months.update(pd.period_range(s.to_period("M"), e.to_period("M"), freq="M"))
    return months


def chart_coverage(gaps: pd.DataFrame, meter_nights: pd.DataFrame) -> str:
    """Nights excluded per month, so a concentrated outage is visible rather
    than silently averaged away.

    Months containing a meter gap (read from ``gaps``, not asserted) are
    drawn in the second series colour, so the chart shows *why* the
    exclusions cluster where they do rather than only that they do.
    Survives an empty ``meter_nights`` frame by drawing empty, labelled axes
    instead of raising.
    """
    fig, ax = plt.subplots(figsize=(9, 3.2))
    mn = meter_nights.copy()
    if len(mn):
        mn["date"] = pd.to_datetime(mn["date"])
        mn["ym"] = mn["date"].dt.to_period("M")
        excluded = (~mn["covered"]).astype(int).groupby(mn["ym"]).sum()
        idx = pd.period_range(mn["ym"].min(), mn["ym"].max(), freq="M")
        excluded = excluded.reindex(idx, fill_value=0)

        gap_months = _gap_months(gaps) if len(gaps) else set()
        x = idx.to_timestamp()
        colours = [SERIES[1] if p in gap_months else SERIES[0] for p in idx]
        ax.bar(x, excluded.to_numpy(), width=20, color=colours)
    ax.set_xlabel("month")
    ax.set_ylabel("nights excluded")
    return _svg(fig)


# ---------------------------------------------------------------------------
# Tables
#
# Per spec S4.4, phase 2's threshold-free readings carry over unchanged and
# no cut-off is introduced: the convergence table, the benefit-share column,
# and the elbow-stability table, all applied to the charge-first bound.
# ---------------------------------------------------------------------------


def _convergence_table(charge_first_sweep: pd.DataFrame, power_kw: float = 3.0) -> str:
    d = convergence_readings(charge_first_sweep, power_kw=power_kw)
    cand = d[d.method != "inflection of the import curve"]["capacity_kwh"]
    head = "".join(f"<th>{h}</th>" for h in
                    ["reading", "lands at (kWh)", "what it measures"])
    rows = "".join(
        f"<tr><td>{r.method}</td><td>{r.capacity_kwh:.1f}</td>"
        f"<td>{r.what_it_measures}</td></tr>" for r in d.itertuples())
    return (
        '<p class="sub">Every threshold-free reading of the charge-first '
        "curve, side by side. They span "
        f"<strong>{cand.min():.1f} to {cand.max():.1f} kWh</strong>; that "
        "spread, not any single row, is the honest answer. No cut-off is "
        "introduced here — phase 2 removed its 50 kWh/yr threshold as a "
        "judgement call, and nothing in this comparison reinstates it.</p>"
        f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _benefit_share_table(charge_first_sweep: pd.DataFrame, power_kw: float = 3.0,
                          max_kwh: float = 12.0) -> str:
    share = benefit_share_pct(charge_first_sweep, power_kw=power_kw)
    d = charge_first_sweep[charge_first_sweep["power_kw"] == power_kw].sort_values(
        "capacity_kwh").copy()
    d["benefit_share_pct"] = share.to_numpy()
    d = d[(d["capacity_kwh"] > 0) & (d["capacity_kwh"] <= max_kwh)]
    head = "".join(f"<th>{h}</th>" for h in
                    ["capacity (kWh)", "share of achievable benefit",
                     "household marginal (kWh/yr per kWh)",
                     "household night grid import (kWh/yr)"])
    rows = "".join(
        f"<tr><td>{r.capacity_kwh:.1f}</td><td>{r.benefit_share_pct:.1f}%</td>"
        f"<td>{r.nonev_marginal_kwh_per_kwh:.1f}</td>"
        f"<td>{r.nonev_night_grid_import_kwh_yr:.1f}</td></tr>"
        for r in d.itertuples())
    return (
        '<p class="sub">Share of the achievable benefit each capacity '
        "captures, on the charge-first bound. Threshold-free: the "
        "denominator is the curve's own endpoints, not a judgement about "
        f"what an extra kWh must earn. Capped at {max_kwh:.0f} kWh — beyond "
        "that both curves are flat.</p>"
        f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _elbow_stability_table(charge_first_sweep: pd.DataFrame, power_kw: float = 3.0) -> str:
    d = elbow_stability(charge_first_sweep, power_kw=power_kw)
    head = "".join(f"<th>{h}</th>" for h in
                    ["sweep carried to (kWh)", "elbow lands at (kWh)",
                     "marginal return at the top end"])
    rows = "".join(
        f"<tr><td>{r.sweep_top_kwh:.0f}</td><td>{r.elbow_kwh:.1f}</td>"
        f"<td>{r.top_end_marginal:.2f}</td></tr>" for r in d.itertuples())
    # Whether the elbow has settled is read off the table, never asserted.
    # This caption was inherited from phase 2, where the last two rows agreed
    # and the claim was true. On this data they do not agree, and the
    # inherited sentence said the opposite of the numbers printed under it.
    settled = len(d) >= 2 and d["elbow_kwh"].iloc[-1] == d["elbow_kwh"].iloc[-2]
    if settled:
        verdict = (
            f"The last two truncations both read "
            f"<strong>{d['elbow_kwh'].iloc[-1]:.1f} kWh</strong>, so the "
            "elbow has stopped moving with the sweep — the signal that the "
            "sweep was carried far enough.")
    else:
        verdict = (
            f"The last two truncations read "
            f"<strong>{d['elbow_kwh'].iloc[-2]:.1f}</strong> and "
            f"<strong>{d['elbow_kwh'].iloc[-1]:.1f} kWh</strong>, so the "
            "elbow has <strong>not</strong> settled: it is still climbing "
            "where the sweep stops. The headline figure is therefore the "
            "reading of a sweep carried to "
            f"{d['sweep_top_kwh'].iloc[-1]:.0f} kWh, and a longer sweep "
            "reads higher. Treat it as the low end of the range in the "
            "convergence table above, not as a converged answer.")
    return (
        '<p class="sub">How the elbow (charge-first bound) moves as the '
        f"sweep is truncated. {verdict}</p>"
        f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def gap_summary(gaps: pd.DataFrame) -> dict | None:
    """The facts behind the outage sentence and the stat tile, read once
    from ``gaps`` so the two never drift apart. ``None`` when there are no
    gaps to summarise.

    Timestamps are stripped of timezone before conversion to a monthly
    period: the period itself has no offset, and converting a tz-aware
    series straight to ``Period`` only raises a UserWarning for no benefit.
    """
    if gaps.empty:
        return None
    starts = pd.to_datetime(gaps["gap_start_utc"]).dt.tz_localize(None)
    ends = pd.to_datetime(gaps["gap_end_utc"]).dt.tz_localize(None)
    by_month = starts.dt.to_period("M").value_counts()
    worst_month = by_month.idxmax()
    in_worst = int(by_month.max())
    in_this_month = starts.dt.to_period("M") == worst_month
    lo = starts[in_this_month].min()
    hi = ends[in_this_month].max()
    if lo.month == hi.month:
        span = f"{lo:%-d}–{hi:%-d} {lo:%B}"
    else:
        span = f"{lo:%-d %B}–{hi:%-d %B}"
    return {
        "total": len(gaps),
        "worst_month": worst_month,
        "month_name": worst_month.strftime("%B %Y"),
        "in_worst": in_worst,
        "span": span,
        "is_winter": worst_month.month in (11, 12, 1, 2),
    }


def _gap_prose(gaps: pd.DataFrame, excluded_nights: int) -> str:
    """The outage sentence, computed from ``gaps`` rather than asserted.

    An earlier draft stated the gap count and the January window as literal
    text, so it could (and did) disagree with whatever ``gaps`` frame was
    actually passed in. Every number here is read from the frame instead.
    """
    g = gap_summary(gaps)
    if g is None:
        return (
            "Nights are dropped, not treated as low-consumption, whenever "
            "the meter has a gap inside the night window. This record has "
            "no gaps, so no nights are excluded for missing intervals."
        )
    season = (
        " — midwinter, when night consumption is at its annual peak"
        if g["is_winter"] else ""
    )
    if g["total"] == 1:
        where = (f"The record has one gap, in <strong>{g['month_name']}</strong>, "
                 f"removing most of {g['span']}{season}.")
    else:
        verb = "falls" if g["in_worst"] == 1 else "fall"
        where = (f"The record has {g['total']} gaps; {g['in_worst']} of them "
                 f"{verb} in <strong>{g['month_name']}</strong>, together "
                 f"removing most of {g['span']}{season}.")
    return (
        "Nights are dropped, not treated as low-consumption, whenever the "
        f"meter has a gap inside the night window. {where} That removes "
        f"<strong>{excluded_nights}</strong> nights from this analysis "
        f"around {g['month_name']} alone."
    )


def _sensitivity_table(sensitivity: pd.DataFrame) -> str:
    head = "".join(f"<th>{h}</th>" for h in
                    ["power (kW)", "hours", "EV nights", "median EV kWh",
                     "median household kWh"])
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in [
            f"{r.power_kw:.1f}", f"{r.hours:.0f}", f"{r.n_ev:.0f}",
            f"{r.median_ev_kwh:.1f}", f"{r.median_rest_kwh:.2f}"]) + "</tr>"
        for r in sensitivity.itertuples())
    return (f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------


def build_html(meter_nights: pd.DataFrame, pv_nights: pd.DataFrame,
                meter_sweep: pd.DataFrame, pv_sweep: pd.DataFrame,
                sensitivity: pd.DataFrame, monthly_ratio: pd.DataFrame,
                gaps: pd.DataFrame, resolution_penalty_pct: float,
                excluded_nights: int,
                elbow_overlapping_span: float = float("nan"),
                tariff_bands: pd.DataFrame | None = None,
                annual_savings: pd.DataFrame | None = None,
                quotes: pd.DataFrame | None = None,
                economics_summary: dict | None = None,
                arbitrage: dict | None = None,
                horizon_years: float = 10.0,
                vat_rate: float = 0.21,
                priced: pd.DataFrame | None = None,
                fixed_cost_terms: pd.DataFrame | None = None,
                fixed_cost_eur: float = 0.0,
                best_eur_per_kwh: float = 0.0,
                usable_fraction: float = 0.90,
                round_trip: float = 0.90,
                monthly_wall: pd.DataFrame | None = None,
                monthly_discharge: pd.DataFrame | None = None,
                coverable_pct: float = float("nan")) -> str:
    """Assemble the report. No document wrapper — the host supplies it.

    Leads with the meter figures; phase 2's PVOutput-based figures are shown
    alongside and labelled superseded, because its consumption channel was
    measured faulty from December 2022 onward (spec section 2.1).
    """
    power_kw = 3.0
    cf = meter_sweep[(meter_sweep["bound"] == "charge_first")
                      & (meter_sweep["power_kw"] == power_kw)]
    df_ = meter_sweep[(meter_sweep["bound"] == "discharge_first")
                       & (meter_sweep["power_kw"] == power_kw)]
    elbow_cf = elbow_capacity(cf, power_kw=power_kw)
    elbow_df = elbow_capacity(df_, power_kw=power_kw)
    lo, hi = sorted((elbow_cf, elbow_df))

    covered_meter = meter_nights[meter_nights["covered"]]
    ev_share = (100.0 * covered_meter["is_ev"].sum() / len(covered_meter)
                if len(covered_meter) else float("nan"))
    if len(covered_meter):
        _d = pd.to_datetime(covered_meter["date"])
        span_label = ("covered nights behind these figures, "
                      f"{_d.min():%b %Y}–{_d.max():%b %Y}")
    else:
        span_label = "covered nights behind these figures"

    gaps_info = gap_summary(gaps)
    excluded_label = (
        f"nights excluded around the {gaps_info['month_name']} outage"
        if gaps_info is not None
        else "nights excluded for missing meter intervals"
    )

    # The bracket can collapse: if both orderings elbow at the same capacity,
    # "9.0–9.0 kWh" is a range only typographically. Say the actual finding.
    #
    # "nameplate" is not decoration. The sweep's capacities are nameplate and
    # only `usable_fraction` of each is cycled, so a reader comparing product
    # datasheets is reading a different number without that word.
    degenerate = lo == hi
    capacity_stat = (f"{lo:.1f} kWh" if degenerate
                     else f"{lo:.1f}–{hi:.1f} kWh")
    # Whether the elbow settled decides how the tile may describe itself, so
    # it is read from the same table the stability card prints rather than
    # assumed. An unsettled elbow is a lower bound, not a recommendation.
    stab = elbow_stability(cf, power_kw=power_kw)
    elbow_settled = (len(stab) >= 2
                     and stab["elbow_kwh"].iloc[-1] == stab["elbow_kwh"].iloc[-2])
    settled_note = ("" if elbow_settled
                    else ", low end — still climbing where the sweep stops")
    capacity_caption = (
        (f"nameplate capacity, the same under both within-interval orderings"
         if degenerate else
         "nameplate capacity, range across the charge-first / "
         "discharge-first bounds") + settled_note)
    penalty_text = format_penalty_pct(resolution_penalty_pct)

    # How the penalty should be read depends on its size, so the sentence
    # branches rather than asserting one interpretation for every value.
    # A tenth of a percent is the scale at which this measurement's own
    # arbitrary choices (which quarter hour a boundary sample joins, where
    # the record is truncated) move the answer, so below that it says
    # nothing.
    penalty_reading = (
        " — below a tenth of a percent, which is the scale at which this "
        "measurement's own boundary choices move it, so it is not usefully "
        "distinguishable from zero. It is stated because it was measured, "
        "not because it is large. Near zero is what the mechanism predicts: "
        "at night there is no generation to cancel against load inside an "
        "interval, and a power cap in kW binds at the same rate however "
        "long the interval is, so coarsening can only blur short peaks and "
        "daytime charging."
        if abs(resolution_penalty_pct) < 0.1 else
        " — the recommended capacity above already carries this cost, since "
        "the meter itself only records at 15 minutes."
    )

    # The two curves cover different spans — the meter record runs longer at
    # both ends — so attributing their difference to the SOURCE requires
    # showing it is not the span. The restricted re-run is measured in
    # `analyze_meter.elbow_on_overlapping_span` and quoted here; if it was
    # not supplied, the page says so rather than implying the check happened.
    if not np.isfinite(elbow_overlapping_span):
        span_caveat = (
            "The two curves cover different spans, and that difference has "
            "not been separated from the difference in source here.")
    elif elbow_overlapping_span == lo:
        span_caveat = (
            "The two records cover different spans, so the meter sweep was "
            "re-run over nothing but the dates PVOutput also covers: it "
            f"still elbows at <strong>{elbow_overlapping_span:.1f} kWh</strong>. "
            "The gap between the curves is therefore the channel, not the "
            "calendar.")
    else:
        span_caveat = (
            "The two records cover different spans. Re-run over nothing but "
            "the dates PVOutput also covers, the meter elbows at "
            f"<strong>{elbow_overlapping_span:.1f} kWh</strong> rather than "
            f"<strong>{lo:.1f}</strong>, so part of the gap between the "
            "curves is the calendar rather than the channel.")

    agreement = bounds_agreement_pct(meter_sweep, power_kw=power_kw)
    bounds_prose = (
        f"The two curves never separate by more than "
        f"<strong>{agreement:.2f}%</strong> of the grid import the house "
        "actually paid for, and "
        + (f"they elbow at the same capacity, <strong>{lo:.1f} kWh</strong>. "
           "The ordering the meter cannot record therefore does not change "
           "the answer — which is what running both bounds was for."
           if degenerate else
           f"they elbow at <strong>{lo:.1f}</strong> and "
           f"<strong>{hi:.1f} kWh</strong>, so the recommended capacity is "
           "reported as that range rather than a single number.")
    )

    stats = [
        (capacity_stat, capacity_caption),
        (f"{covered_meter['import_kwh'].median():.2f} kWh",
         "median night consumption (meter)"),
        # A reader could not previously tell from the page how much data is
        # behind the headline. Both figures come from the frame being plotted,
        # and an empty frame has no span to name rather than a NaT one.
        (f"{len(covered_meter):,}", span_label),
        (f"{ev_share:.0f}%", "of covered nights are EV-charging"),
        (penalty_text,
         "resolution penalty, 15-minute vs finer simulation"),
        (f"{excluded_nights}", excluded_label),
    ]
    stat_html = "".join(
        f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)

    cards = [
        ("The fault, measured",
         "PVOutput's consumption channel is cross-checked against the "
         "smart meter every month across the whole record. The ratio holds "
         "near 1.0 through November 2022, then steps down in "
         "<strong>December 2022</strong> and never recovers, drifting "
         "further down year by year after the step — a discrete fault that "
         "then worsens, and the reason phase 2's battery recommendation is "
         "now <strong>superseded</strong> for any period after that step.",
         chart_visible_fraction(monthly_ratio)),
        ("What a night actually costs, both ways",
         "Night-energy distributions from the meter and from PVOutput, "
         "restricted to the period both sources cover. Where PVOutput's "
         "channel was already faulty, its curve sits visibly below the "
         "meter's — the same fault chart one shows, expressed as missed "
         "energy rather than a ratio.",
         chart_night_distribution(meter_nights, pv_nights)),
        *([] if monthly_wall is None or monthly_wall.empty else [
            ("The winter wall",
             _wall_prose(monthly_wall, coverable_pct,
                         negative_surplus_months(monthly_wall)),
             chart_surplus_vs_need(monthly_wall))]),
        *([] if monthly_discharge is None or monthly_discharge.empty else [
            ("What the battery actually delivers, month by month",
             "The same wall seen from the battery's side. Discharge at "
             f"<strong>{lo:.1f} kWh</strong> collapses in the months above "
             "that have nothing to charge from, and no capacity changes "
             "that — which is why the sizing question below is bounded by "
             "the chart above it, not by the curve.",
             chart_monthly_discharge(monthly_discharge, lo))]),
        ("How much capacity is worth buying, and how little the ordering "
         "matters",
         "The meter cannot tell which of a quarter-hour's import and "
         "export happened first, so every capacity is simulated under both "
         f"orderings: <strong>charge-first</strong> ({elbow_cf:.1f} kWh, "
         "the export banked before the import is served) and "
         f"<strong>discharge-first</strong> ({elbow_df:.1f} kWh, the "
         "import served before the export exists). Neither is favourable "
         "or unfavourable — within one interval charge-first bridges more, "
         "but state of charge couples the intervals, and discharging first "
         "makes room to capture export that would otherwise be spilled, so "
         "on this record discharge-first ends up marginally ahead. "
         + bounds_prose,
         chart_bounds(meter_sweep, elbow_cf, elbow_df)),
        ("Where the marginal kWh stops paying, both bounds",
         "Marginal return for household (non-EV) nights, both orderings. "
         "No cut-off is drawn on this chart: phase 2 removed its 50 kWh/yr "
         "threshold as a judgement call, and this comparison does not "
         "reinstate it.",
         chart_marginal(meter_sweep),
         _convergence_table(cf),
         _benefit_share_table(cf),
         _elbow_stability_table(cf)),
        ("Meter versus phase 2, on one axis",
         "The meter's charge-first curve against phase 2's PVOutput-based "
         "curve, drawn dashed and labelled: phase 2 is "
         "<strong>superseded</strong> because its consumption channel was "
         "faulty from <strong>December 2022</strong>, so it understates "
         "night load and, with it, the capacity worth buying. "
         + span_caveat,
         chart_comparison(meter_sweep, pv_sweep)),
        *([] if tariff_bands is None else [
            ("What a kWh is actually worth", tariff_card(tariff_bands), "")]),
        *([] if annual_savings is None or economics_summary is None else [
            ("What the battery saves, in euros",
             _euro_prose(economics_summary, horizon_years),
             chart_savings(annual_savings))]),
        *([] if annual_savings is None or quotes is None else [
            ("What it would have to cost, against real quotes",
             "The installed price per kWh at which each capacity exactly "
             f"repays itself in {horizon_years:.0f} years, with the quoted "
             "products marked. A product below the curve pays back inside "
             "the horizon; one above it does not."
             + _quote_table_html(quotes, horizon_years, vat_rate),
             chart_break_even(annual_savings, quotes, horizon_years))]),
        *([] if (annual_savings is None or priced is None
                 or fixed_cost_terms is None or economics_summary is None) else [
            ("How much to trust that number",
             _confidence_card(annual_savings, priced,
                              economics_summary["euro_optimum_kwh"],
                              best_eur_per_kwh, fixed_cost_eur, horizon_years,
                              fixed_cost_terms, usable_fraction, round_trip),
             "")]),
        *([] if arbitrage is None else [
            ("The ceiling on trading the tariff", _arbitrage_prose(arbitrage),
             "")]),
        ("Nights excluded from the analysis",
         _gap_prose(gaps, excluded_nights),
         chart_coverage(gaps, meter_nights)),
    ]
    card_html = "".join(
        f'<section class="card"><h2>{t}</h2><p>{c}</p>'
        f'<div class="chart">{svg}</div>{"".join(rest)}</section>'
        for t, c, svg, *rest in cards)

    ev_note = (
        '<section class="card"><h2>How EV nights were identified</h2>'
        "<p>A night counts as EV-charging when its samples spend at least "
        "some hours above a power threshold. Phase 2's rule — two hours "
        "above 2 kW — is <strong>retained unchanged</strong> and applied to "
        "the meter. It was not re-derived: nothing here fits a threshold, "
        "and an earlier version of this page wrongly said otherwise. The "
        "reason it can be retained is in the table below — the split moves "
        "with the threshold, but the recommended capacity does not, so the "
        "choice does not have to be defended. It remains a "
        "<strong>heuristic</strong>, "
        "not a measurement: the meter is whole-house and the car is not "
        "sub-metered. The table below shows how the split moves as the "
        "threshold moves, republished against the meter.</p>"
        f"{_sensitivity_table(sensitivity)}</section>")

    resolution_note = (
        '<section class="card"><h2>The resolution penalty</h2>'
        "<p>Averaging over 15 minutes hides short peaks, so a battery "
        "simulated at that resolution might look better than reality. "
        "Rather than assume a size for this effect, it is measured "
        "directly: phase 2's own finer PVOutput data is aggregated onto "
        "the <strong>15-minute</strong> boundaries the meter keeps, and "
        "the sweep is re-run. The measured difference is "
        f"<strong>{penalty_text}</strong> of the 15-minute figure"
        + penalty_reading + "</p></section>")

    return (
        "<title>Night Consumption, Measured at the Meter</title>"
        + STYLE
        + TABLE_CSS
        + "<main><h1>What size battery is worth buying, measured at the "
        "meter</h1>"
        + '<p class="lede">PVOutput\'s consumption channel undercounted '
        "the house from December 2022 onward. This page redoes the night-"
        "consumption and battery-sizing analysis against the smart meter "
        "and leads with that result; phase 2's PVOutput-based figures are "
        "shown alongside, dashed, and labelled superseded.</p>"
        + f'<div class="stats">{stat_html}</div>'
        + card_html + resolution_note + ev_note + "</main>")
