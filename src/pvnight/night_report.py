"""Charts and the battery-sizing report page.

Renders eight inline SVG charts plus the sensitivity table behind the EV
heuristic as a single self-contained HTML fragment, for publishing as an
Artifact (no document wrapper — the host supplies that). Reuses phase 1's
validated palette and page chrome from ``pvnight.report`` rather than
re-picking colours.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .report import COVERAGE_CMAP, FURNITURE, SERIES, STYLE, _svg  # noqa: E402,F401

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November",
               "December"]


def _by_month(monthly: pd.DataFrame) -> pd.DataFrame:
    """Twelve rows, January first, whatever the caller passed.

    The charts label months by position, so trusting the caller's row order
    would silently mislabel a frame that happened to be sorted differently.
    A missing month becomes NaN and simply renders as a gap.
    """
    return monthly.set_index("month").reindex(range(1, 13)).reset_index()


def _join_month_names(months: list[int]) -> str:
    """"November, December and January" from [11, 12, 1] — order preserved,
    no Oxford comma, singular list reads as a bare name."""
    names = [MONTH_NAMES[m - 1] for m in months]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def chart_night_distribution(nights: pd.DataFrame) -> str:
    """Empirical CDF of night energy, EV against non-EV."""
    fig, ax = plt.subplots(figsize=(9, 4))
    c = nights[nights["covered"]]
    for sub, colour, label in [
        (c[~c["is_ev"]], SERIES[0], "household nights"),
        (c[c["is_ev"]], SERIES[1], "EV-charging nights"),
    ]:
        v = np.sort(sub["night_wh"].to_numpy() / 1000)
        if len(v):
            ax.plot(v, np.linspace(0, 100, len(v)), lw=2, color=colour,
                    label=f"{label} (n={len(v)})")
    ax.set_xlabel("night consumption (kWh)")
    ax.set_ylabel("percentile")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_night_by_month(nights: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    c = nights[nights["covered"]].copy()
    c["m"] = pd.to_datetime(c["date"]).dt.month
    med = c.groupby("m")["night_wh"].median() / 1000
    p90 = c.groupby("m")["night_wh"].quantile(0.9) / 1000
    ax.fill_between(med.index, med, p90, color=SERIES[0], alpha=0.2, label="median to p90")
    ax.plot(med.index, med, lw=2, color=SERIES[0], marker="o", ms=4, label="median")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("night consumption (kWh)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_surplus_vs_need(monthly: pd.DataFrame) -> str:
    """The winter wall: months where surplus is negative can charge nothing."""
    fig, ax = plt.subplots(figsize=(9, 4))
    m = _by_month(monthly)
    x = np.arange(12)
    ax.bar(x - 0.2, m["surplus_kwh"], 0.4, color=SERIES[2], label="daytime surplus")
    ax.bar(x + 0.2, m["night_kwh"], 0.4, color=SERIES[1], label="night need")
    ax.axhline(0, color=FURNITURE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("kWh per day (median)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_knee(sweep: pd.DataFrame, recommended_kwh: float) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    d = sweep[sweep["power_kw"] == 3.0].sort_values("capacity_kwh")
    ax.plot(d.capacity_kwh, d.night_grid_import_kwh_yr, lw=2, color=SERIES[1],
            label="all nights")
    ax.plot(d.capacity_kwh, d.nonev_night_grid_import_kwh_yr, lw=2, color=SERIES[0],
            label="household nights only")
    ax.axvline(recommended_kwh, color=FURNITURE, ls="--", lw=1)
    ax.annotate(f"{recommended_kwh:.1f} kWh", xy=(recommended_kwh, ax.get_ylim()[1] * 0.9),
                color=FURNITURE, fontsize=8)
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("night grid import (kWh/yr)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_marginal(sweep: pd.DataFrame, recommended_kwh: float,
                    threshold: float = 50.0) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    d = sweep[sweep["power_kw"] == 3.0].sort_values("capacity_kwh")
    ax.plot(d.capacity_kwh, d.marginal_kwh_per_kwh, lw=2, color=SERIES[0])
    ax.axhline(threshold, color=SERIES[1], ls="--", lw=1,
               label=f"{threshold:.0f} kWh/yr per kWh cut-off")
    ax.axvline(recommended_kwh, color=FURNITURE, ls="--", lw=1)
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("marginal kWh/yr avoided per added kWh")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_utilisation(monthly: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    m = _by_month(monthly)
    ax.bar(np.arange(12), m["discharge_kwh"], color=SERIES[0])
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("battery discharge (kWh)")
    return _svg(fig)


def chart_night_self_sufficiency(sweep: pd.DataFrame, recommended_kwh: float) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for p, colour in zip(sorted(sweep.power_kw.unique()), SERIES):
        d = sweep[sweep.power_kw == p].sort_values("capacity_kwh")
        ax.plot(d.capacity_kwh, d.nonev_night_self_sufficiency_pct, lw=2,
                color=colour, label=f"{p:.1f} kW inverter")
    ax.axvline(recommended_kwh, color=FURNITURE, ls="--", lw=1)
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("household night self-sufficiency (%)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_above_cap(monthly: pd.DataFrame) -> str:
    """Night energy a 3 kW inverter physically cannot supply."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    m = _by_month(monthly)
    x = np.arange(12)
    ax.bar(x - 0.2, m["above_cap_pct_ev"], 0.4, color=SERIES[1], label="EV nights")
    ax.bar(x + 0.2, m["above_cap_pct_nonev"], 0.4, color=SERIES[0], label="household nights")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("% of night energy drawn above 3 kW")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


# ---------------------------------------------------------------------------
# Page assembly
#
# STYLE (imported from report.py) carries no `table` rule, so TABLE_CSS adds
# one for the EV-sensitivity table using the same theme tokens (--line,
# --muted) STYLE already defines, rather than new literals.
# ---------------------------------------------------------------------------

TABLE_CSS = """
<style>
table { border-collapse: collapse; font-size: 13px; width: 100%; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 500; }
</style>
"""


def _sensitivity_table(sens: pd.DataFrame) -> str:
    head = "".join(f"<th>{h}</th>" for h in
                    ["power (W)", "hours", "EV nights", "median EV kWh", "median other kWh"])
    body = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in [
            f"{r.power_w:.0f}", f"{r.hours:.0f}", f"{r.n_ev:.0f}",
            f"{r.median_ev_kwh:.1f}", f"{r.median_rest_kwh:.2f}"]) + "</tr>"
        for r in sens.itertuples())
    return (f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def _winter_wall_prose(coverable_pct: float, negative_surplus_months: list[int]) -> str:
    """The winter finding, built from real figures rather than literals.

    This is the analysis's single most important caveat, so it must move
    with the data: a refreshed dataset changes ``coverable_pct`` and
    ``negative_surplus_months``, and this sentence changes with it.
    """
    if negative_surplus_months:
        names = _join_month_names(negative_surplus_months)
        wall = (
            f"In {names} the surplus is <strong>negative</strong> — the "
            "house does not generate enough to cover even its daytime "
            "load, so no battery of any size receives a charge."
        )
    else:
        wall = (
            "No month in this dataset has a negative median daytime "
            "surplus, so winter alone does not block charging here."
        )
    return (
        "Median daytime surplus against median night need, per month. "
        f"{wall} Across all six years only {coverable_pct:.1f}% of nights "
        "could be covered even with infinite storage. This, not capacity, "
        "is what bounds the answer."
    )


def build_html(nights_df: pd.DataFrame, sweep_df: pd.DataFrame,
                monthly_df: pd.DataFrame, sensitivity_df: pd.DataFrame,
                recommended_kwh: float, coverable_pct: float,
                negative_surplus_months: list[int]) -> str:
    """Assemble the report. No document wrapper — the host supplies it.

    ``coverable_pct`` and ``negative_surplus_months`` drive the winter-wall
    prose directly, rather than being restated as literals: the caller
    computes them from the same data the charts draw from, so a refreshed
    dataset moves the sentence along with the numbers.
    """
    covered = nights_df[nights_df["covered"]]
    at = sweep_df[(sweep_df.power_kw == 3.0)
                  & np.isclose(sweep_df.capacity_kwh, recommended_kwh)]
    ss = float(at["nonev_night_self_sufficiency_pct"].iloc[0]) if len(at) else float("nan")
    cyc = float(at["cycles_per_yr"].iloc[0]) if len(at) else float("nan")

    stats = [
        (f"{recommended_kwh:.1f} kWh", "recommended capacity, nameplate"),
        (f"{covered['night_wh'].median() / 1000:.2f} kWh", "median night consumption"),
        (f"{int(covered['is_ev'].sum())}", f"EV-charging nights of {len(covered)}"),
        (f"{ss:.0f}%", "household night self-sufficiency"),
        (f"{cyc:.0f}", "full cycles per year"),
    ]
    stat_html = "".join(
        f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)

    cards = [
        ("What a night actually costs",
         "Every covered night, as a cumulative distribution. The household "
         "curve is tight; the EV curve is a different animal entirely.",
         chart_night_distribution(nights_df)),
        ("Night consumption through the year",
         "Longer nights and higher load compound: midwinter nights use "
         "roughly three times a midsummer night.",
         chart_night_by_month(nights_df)),
        ("The winter wall",
         _winter_wall_prose(coverable_pct, negative_surplus_months),
         chart_surplus_vs_need(monthly_df)),
        ("How much capacity is worth buying",
         "Night grid import against capacity, with and without EV nights. "
         "The curve flattens because winter contributes nothing no matter "
         "how large the battery is.",
         chart_knee(sweep_df, recommended_kwh)),
        ("Where the marginal kWh stops paying",
         "This curve rises before it falls: a battery this small is "
         "exhausted within minutes of sunset, so its first half-kWh barely "
         "helps, and marginal return climbs before it ever starts "
         "declining. That is why the rule can't just take the earliest "
         "point under the line — that would land on this leading dip, not "
         "the real knee. "
         f"So the recommendation of {recommended_kwh:.1f} kWh is the "
         "smallest capacity beyond which the marginal return never rises "
         "above 50 kWh/yr per added kWh again — an extra kWh cycling less "
         "than once a week from there on. <strong>That 50 is a stated "
         "judgement, not a derived constant</strong>: read your own "
         "cut-off straight off this curve if you prefer a different one.",
         chart_marginal(sweep_df, recommended_kwh)),
        ("When the battery actually works",
         "Discharge by month at the recommended capacity. The winter months "
         "are near-idle, which is the same finding as the third chart seen "
         "from the battery's side.",
         chart_utilisation(monthly_df)),
        ("What a bigger inverter would buy",
         "Household night self-sufficiency against capacity, for three "
         "single-phase inverter sizes. If these curves sit on top of each "
         "other, inverter power is not the binding constraint and the "
         "cheapest of the three will do.",
         chart_night_self_sufficiency(sweep_df, recommended_kwh)),
        ("What a 3 kW inverter cannot supply",
         "Share of night energy drawn above 3 kW, which no single-phase "
         "battery can deliver at any capacity. On household nights this is "
         "negligible; on EV nights it is about a third, which is the "
         "clearest argument for keeping the car off the battery.",
         chart_above_cap(monthly_df)),
    ]
    card_html = "".join(
        f'<section class="card"><h2>{t}</h2><p>{c}</p>'
        f'<div class="chart">{svg}</div></section>' for t, c, svg in cards)

    ev_note = (
        '<section class="card"><h2>How EV nights were identified</h2>'
        "<p>A night counts as EV-charging when at least two hours of its "
        "samples exceed 2 kW. This is a <strong>heuristic</strong>, not a "
        "measurement: the meter is whole-house and the car is not "
        "sub-metered. The obvious test — a peak above 5 kW — misses the "
        "slow charging mode entirely, which runs about 3.5 kW for twelve "
        "hours. The classifier has a known false-positive rate: it flags "
        "six pre-2022 winter evenings that cannot be a car, since the EV "
        "did not arrive until 2022. The table below shows how the split "
        "moves as the threshold moves, so you can judge whether it is "
        "stable or an artefact of where the line was drawn.</p>"
        f"{_sensitivity_table(sensitivity_df)}</section>")

    return (
        "<title>Night Consumption and Battery Sizing</title>"
        + STYLE
        + TABLE_CSS
        + "<main><h1>What size battery is worth buying</h1>"
        + '<p class="lede">Six years of measured five-minute data, simulated '
        + "against a sweep of capacities behind a single-phase 3 kW "
        + "inverter.</p>"
        + f'<div class="stats">{stat_html}</div>'
        + card_html + ev_note + "</main>")
