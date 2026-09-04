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
from .night_report import TABLE_CSS  # noqa: E402
from .report import FURNITURE, SERIES, STYLE, _svg  # noqa: E402

# The two within-interval orderings (spec S4.2). "charge_first" is the
# favourable bound (the battery stores the export and spends it on the
# import moments later); "discharge_first" is unfavourable (the import
# arrives first, so only previously-stored energy can serve it). Neither is
# the answer; the pair is the range.
BOUND_LABEL = {"charge_first": "charge-first", "discharge_first": "discharge-first"}
BOUND_COLOUR = {"charge_first": SERIES[0], "discharge_first": SERIES[1]}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


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

    The shaded band between the two lines is the answer; the report treats
    neither edge as a point estimate.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    d = _bound_pair(meter_sweep)
    ax.fill_between(d.capacity_kwh, d.nonev_night_grid_import_kwh_yr_cf,
                     d.nonev_night_grid_import_kwh_yr_df, color=SERIES[2],
                     alpha=0.2, label="range between bounds")
    ax.plot(d.capacity_kwh, d.nonev_night_grid_import_kwh_yr_cf, lw=2,
            color=BOUND_COLOUR["charge_first"], label="charge-first (favourable)")
    ax.plot(d.capacity_kwh, d.nonev_night_grid_import_kwh_yr_df, lw=2,
            color=BOUND_COLOUR["discharge_first"], label="discharge-first (unfavourable)")
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
    return (
        '<p class="sub">How the elbow (charge-first bound) moves as the '
        "sweep is truncated. Once the top-end marginal return has gone "
        "flat, the elbow stops moving with it — the signal the sweep went "
        "far enough.</p>"
        f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _gap_summary(gaps: pd.DataFrame) -> dict | None:
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
    g = _gap_summary(gaps)
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
    gap_word = "gap" if g["total"] == 1 else "gaps"
    return (
        "Nights are dropped, not treated as low-consumption, whenever the "
        f"meter has a gap inside the night window. The record has "
        f"{g['total']} {gap_word}; {g['in_worst']} of them fall in "
        f"<strong>{g['month_name']}</strong>, together removing most of "
        f"{g['span']}{season}. That removes "
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
                excluded_nights: int) -> str:
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
    gap_summary = _gap_summary(gaps)
    excluded_label = (
        f"nights excluded around the {gap_summary['month_name']} outage"
        if gap_summary is not None
        else "nights excluded for missing meter intervals"
    )

    stats = [
        (f"{lo:.1f}–{hi:.1f} kWh",
         "recommended capacity, range across the charge-first / "
         "discharge-first bounds"),
        (f"{covered_meter['import_kwh'].median():.2f} kWh",
         "median night consumption (meter)"),
        (f"{ev_share:.0f}%", "of covered nights are EV-charging"),
        (f"{resolution_penalty_pct:.1f}%",
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
         "<strong>December 2022</strong> and never recovers — evidence of "
         "a hardware fault, not a gradual drift, and the reason phase 2's "
         "battery recommendation is now <strong>superseded</strong> for "
         "any period after that step.",
         chart_visible_fraction(monthly_ratio)),
        ("What a night actually costs, both ways",
         "Night-energy distributions from the meter and from PVOutput, "
         "restricted to the period both sources cover. Where PVOutput's "
         "channel was already faulty, its curve sits visibly below the "
         "meter's — the same fault chart one shows, expressed as missed "
         "energy rather than a ratio.",
         chart_night_distribution(meter_nights, pv_nights)),
        ("How much capacity is worth buying, as a range",
         "The meter cannot tell which of a quarter-hour's import and "
         "export happened first, so every capacity is simulated under both "
         f"orderings: <strong>charge-first</strong> ({elbow_cf:.1f} kWh, "
         "favourable — the battery banks the export and spends it on the "
         "import moments later) and <strong>discharge-first</strong> "
         f"({elbow_df:.1f} kWh, unfavourable — the import arrives first, "
         "so only energy already stored can serve it). Neither bound is "
         "the answer; the shaded band between them is. The recommended "
         f"capacity is therefore reported as a range, "
         f"<strong>{lo:.1f}–{hi:.1f} kWh</strong>, never a single "
         "number.",
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
         "night load and, with it, the capacity worth buying.",
         chart_comparison(meter_sweep, pv_sweep)),
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
        "some hours above a power threshold. This is re-derived here "
        "because phase 2's rule — two hours above 2 kW — was fitted "
        "against a signal that was missing much of the car; the threshold "
        "that separated a partial signal is not necessarily the one that "
        "separates a complete one. It remains a <strong>heuristic</strong>, "
        "not a measurement: the meter is whole-house and the car is not "
        "sub-metered. The table below shows how the split moves as the "
        "threshold moves, republished against the meter.</p>"
        f"{_sensitivity_table(sensitivity)}</section>")

    resolution_note = (
        '<section class="card"><h2>The resolution penalty</h2>'
        "<p>Averaging over 15 minutes hides short peaks, so a battery "
        "simulated at that resolution looks better than reality. Rather "
        "than assume a size for this effect, it is measured directly: "
        "phase 2's own finer PVOutput data is downsampled to the "
        "<strong>15-minute</strong> interval the meter is stuck with, and "
        "the sweep is re-run. The measured penalty is "
        f"<strong>{resolution_penalty_pct:.1f}%</strong> against the "
        "15-minute interval — the recommended capacity above already "
        "carries this cost, since the meter itself only records at 15 "
        "minutes.</p></section>")

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
