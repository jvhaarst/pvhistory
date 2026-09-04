"""Charts and the HTML report page.

Renders the fitted seasonal envelope, the observed events behind it, and the
resulting solar window as a single self-contained HTML fragment, for
publishing as an Artifact (no document wrapper — the host supplies that).
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import solar
from .config import LATITUDE, LONGITUDE, SITE_TZ, SUNRISE_ELEVATION_DEG

# Neutral grey for axis furniture. This is the dataviz skill's "muted"
# chart-chrome token (#898781) — it is specified identically for the light
# (#fcfcfb) and dark (#1a1a19) chart surfaces, so one static SVG reads on
# both grounds without needing to retheme.
FURNITURE = "#898781"

# Categorical series, dataviz skill slots 1/2/3 (blue, orange, aqua), dark
# step. Charts are static SVG embedded in a page whose theme can flip after
# render, so a single set of hexes has to work on both grounds at once —
# validated with the skill's script against both surfaces and the
# all-pairs pairlist (this module's scatter charts carry 2-3 series each):
#   node scripts/validate_palette.js "#3987e5,#d95926,#199e70" \
#       --mode dark  --surface "#1a1a19" --pairs all   -> ALL CHECKS PASS
#   node scripts/validate_palette.js "#3987e5,#d95926,#199e70" \
#       --mode light --surface "#fcfcfb" --pairs all   -> ALL CHECKS PASS
SERIES = ["#3987e5", "#d95926", "#199e70"]

# Sequential single-hue ramp (blue, light -> dark), dataviz skill's
# palette.md "Sequential hue" table, steps 100 through 700 in order. Used
# for the coverage heatmap instead of a stock multi-hue colormap: the skill
# lists rainbow/multi-hue sequential ramps as an anti-pattern, and this is
# the documented ramp for exactly this case (magnitude on a heatmap).
_SEQUENTIAL_BLUE_STOPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b",
]
COVERAGE_CMAP = LinearSegmentedColormap.from_list(
    "coverage", _SEQUENTIAL_BLUE_STOPS
)


def _svg(fig) -> str:
    """Render a figure to an inline SVG string with a transparent ground."""
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.tick_params(colors=FURNITURE)
        for spine in ax.spines.values():
            spine.set_color(FURNITURE)
        ax.xaxis.label.set_color(FURNITURE)
        ax.yaxis.label.set_color(FURNITURE)
        ax.title.set_color(FURNITURE)
    fig.patch.set_alpha(0.0)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg") :]


def _reference_year(windows: pd.DataFrame) -> int:
    """The most completely represented year in the window table."""
    if windows.empty:
        raise ValueError("_reference_year: windows frame is empty")
    return int(pd.to_datetime(windows["date"]).dt.year.value_counts().idxmax())


def _local_hours(ts) -> np.ndarray:
    """Local hour of day as a float, for plotting against day of year."""
    local = pd.DatetimeIndex(ts).tz_convert(SITE_TZ)
    return local.hour + local.minute / 60 + local.second / 3600


def _doy(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(list(dates))).dt.dayofyear.to_numpy()


def chart_light_times(events: pd.DataFrame, windows: pd.DataFrame) -> str:
    """Observed first and last light against the fitted window and sunrise."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    doy = _doy(events["solar_date"])
    ax.scatter(doy, _local_hours(events["first_light_utc"]), s=3, alpha=0.25,
               color=SERIES[0], linewidths=0, label="observed first light")
    ax.scatter(doy, _local_hours(events["last_light_utc"]), s=3, alpha=0.25,
               color=SERIES[1], linewidths=0, label="observed last light")

    year = _reference_year(windows)
    w = windows[pd.to_datetime(windows["date"]).dt.year == year]
    wd = _doy(w["date"])
    ax.plot(wd, _local_hours(w["solar_start_utc"]), color=SERIES[0], lw=2,
            label="fitted window")
    ax.plot(wd, _local_hours(w["solar_end_utc"]), color=SERIES[1], lw=2)
    ax.plot(wd, _local_hours(w["sunrise_utc"]), color=FURNITURE, lw=1, ls="--",
            label="sunrise / sunset")
    ax.plot(wd, _local_hours(w["sunset_utc"]), color=FURNITURE, lw=1, ls="--")

    ax.set_xlabel("day of year")
    ax.set_ylabel("local time (hours)")
    ax.set_xlim(1, 366)
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_elevation_envelope(model, events: pd.DataFrame) -> str:
    """The model itself: elevation at first/last light, and the fitted curves."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    doy = _doy(events["solar_date"])
    ax.scatter(doy, events["el_first"], s=3, alpha=0.18, color=SERIES[0],
               linewidths=0, label="first light")
    ax.scatter(doy, events["el_last"], s=3, alpha=0.18, color=SERIES[1],
               linewidths=0, label="last light")

    grid_doy = np.arange(1, 367)
    ax.plot(grid_doy, model.raw_start, color=SERIES[0], lw=0.8, ls=":")
    ax.plot(grid_doy, model.raw_end, color=SERIES[1], lw=0.8, ls=":")
    ax.plot(grid_doy, model.theta_start(model.grid_phi), color=SERIES[0], lw=2.2,
            label="fitted theta_start")
    ax.plot(grid_doy, model.theta_end(model.grid_phi), color=SERIES[1], lw=2.2,
            label="fitted theta_end")
    ax.axhline(SUNRISE_ELEVATION_DEG, color=FURNITURE, lw=1, ls="--")
    ax.annotate("sunrise elevation", xy=(300, SUNRISE_ELEVATION_DEG),
                xytext=(300, SUNRISE_ELEVATION_DEG + 1.2),
                color=FURNITURE, fontsize=8)

    ax.set_xlabel("day of year")
    ax.set_ylabel("solar elevation at edge (degrees)")
    ax.set_xlim(1, 366)
    ax.set_ylim(-6, 6)
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_azimuth_screen(events: pd.DataFrame) -> str:
    """Shading screen. NOT a horizon profile — the bins mix seasons."""
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(30, 341, 10)

    for col, el_col, colour, label in [
        ("first_light_utc", "el_first", SERIES[0], "morning"),
        ("last_light_utc", "el_last", SERIES[1], "evening"),
    ]:
        az = solar.azimuth(pd.DatetimeIndex(events[col])).to_numpy()
        el = events[el_col].to_numpy()
        placed = np.digitize(az, bins)
        centres, values = [], []
        for b in range(1, len(bins)):
            selected = placed == b
            if selected.sum() >= 5:
                centres.append((bins[b - 1] + bins[b]) / 2)
                values.append(np.percentile(el[selected], 5))
        ax.plot(centres, values, marker="o", ms=3, lw=1.5, color=colour,
                label=label)

    ax.axhline(SUNRISE_ELEVATION_DEG, color=FURNITURE, lw=1, ls="--")
    ax.set_xlabel("solar azimuth at edge (degrees from north)")
    ax.set_ylabel("5th-percentile elevation (degrees)")
    ax.set_title("Shading screen, not a horizon profile — bins mix seasons",
                 fontsize=10)
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_night_length(windows: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    year = _reference_year(windows)
    w = windows[pd.to_datetime(windows["date"]).dt.year == year].dropna(
        subset=["night_duration_h"]
    )
    ax.plot(_doy(w["date"]), w["night_duration_h"], color=SERIES[2], lw=1.6)
    ax.set_xlabel("day of year")
    ax.set_ylabel("hours of night")
    ax.set_xlim(1, 366)
    return _svg(fig)


def chart_coverage(loaded: pd.DataFrame) -> str:
    """Generating samples per day, by year — shows 2020's partial start."""
    if loaded.empty:
        raise ValueError("chart_coverage: loaded frame is empty")
    daily = loaded.groupby("solar_date")["generating"].sum()
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(daily.index))))
    frame = pd.DataFrame(
        {"year": idx.year, "doy": idx.dayofyear, "n": daily.to_numpy()}
    )
    grid = frame.pivot_table(
        index="year", columns="doy", values="n", aggfunc="max"
    ).reindex(columns=np.arange(1, 367))

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.imshow(
        grid.to_numpy(), aspect="auto", origin="lower", cmap=COVERAGE_CMAP,
        interpolation="nearest",
        extent=[1, 366, grid.index.min() - 0.5, grid.index.max() + 0.5],
    )
    ax.set_yticks(list(grid.index))
    ax.set_yticklabels([str(y) for y in grid.index])
    ax.set_xlabel("day of year")
    return _svg(fig)


# ---------------------------------------------------------------------------
# Page assembly
#
# Design: a technical field report, not a landing page — the deliverable is
# the five charts, so the chrome stays quiet. Palette and chart-chrome tokens
# come from the dataviz skill's validated instance (light chart surface
# #fcfcfb / dark #1a1a19, page plane #f9f9f7 / #0d0d0d); the accent reuses
# categorical slot 1 so the "fitted window" line in the charts and the accent
# in the prose read as the same idea. Headings set in a serif built for
# reading numbers in tables (Source Serif 4) nod to the printed ephemeris
# this page is a digital descendant of; body copy in Public Sans. The site
# coordinates line is set in IBM Plex Mono, since it is a run of distinct
# measurements rather than prose; the stat-tile values are large standalone
# numbers, so per the dataviz skill they stay in the page's primary sans
# with proportional (not tabular) figures — tabular-nums is reserved for
# numbers that actually stack in a column, which none here do.
# ---------------------------------------------------------------------------

STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --muted: #52514e;
  --axis: #898781;
  --line: #e1e0d9;
  --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0d0d0d; --surface: #1a1a19; --ink: #ffffff;
    --muted: #c3c2b7; --axis: #898781; --line: #2c2c2a; --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #ffffff;
  --muted: #c3c2b7; --axis: #898781; --line: #2c2c2a; --accent: #3987e5;
}

* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--ink);
  font: 400 16px/1.6 "Public Sans", ui-sans-serif, system-ui, sans-serif;
}
main { max-width: 900px; margin: 0 auto; padding: 40px 20px 72px; }

.eyebrow {
  font: 500 11px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent);
  margin: 0 0 10px;
}
h1 {
  font: 600 32px/1.15 "Source Serif 4", Georgia, serif;
  text-wrap: balance; margin: 0 0 10px; letter-spacing: -0.01em;
}
.lede {
  color: var(--muted); margin: 0 0 8px; max-width: 62ch; font-size: 15px;
}
.coords {
  font: 400 12px/1.5 "IBM Plex Mono", ui-monospace, monospace;
  color: var(--muted); margin: 0 0 30px;
}
.coords b { color: var(--ink); font-weight: 500; }

.stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1px; background: var(--line);
  border: 1px solid var(--line); border-radius: 10px;
  overflow: hidden; margin: 0 0 32px;
}
.stat { background: var(--surface); padding: 16px 18px; }
.stat b {
  display: block;
  font: 600 22px/1.2 "Public Sans", ui-sans-serif, system-ui, sans-serif;
  margin: 0 0 4px;
}
.stat span { color: var(--muted); font-size: 12px; }

.card {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 20px 22px; margin: 0 0 20px;
}
.card h2 {
  font: 600 16px/1.3 "Source Serif 4", Georgia, serif;
  margin: 0 0 6px; color: var(--ink);
}
.card p { color: var(--muted); font-size: 13.5px; margin: 0 0 14px; max-width: 68ch; }
.card p em { color: var(--ink); font-style: normal; font-weight: 500; }
.chart { overflow-x: auto; }
.chart svg { display: block; min-width: 640px; }

footer {
  color: var(--muted); font-size: 12px; margin-top: 8px;
  border-top: 1px solid var(--line); padding-top: 16px;
}
</style>
"""


def build_html(model, events: pd.DataFrame, windows: pd.DataFrame,
                loaded: pd.DataFrame) -> str:
    """Assemble the report. No document wrapper — the host supplies it."""
    grid_phi = model.grid_phi
    theta_start = model.theta_start(grid_phi)
    nights = windows["night_duration_h"].dropna()

    stats = [
        (f"{theta_start.min():.2f} to {theta_start.max():.2f}°",
         "fitted start threshold"),
        (f"{windows['start_offset_min'].median():.1f} min",
         "median start offset from sunrise"),
        (f"{nights.min():.1f}–{nights.max():.1f} h", "night length range"),
        (f"{len(events):,}", "days with observed generation"),
    ]
    stat_html = "".join(
        f'<div class="stat"><b>{value}</b><span>{label}</span></div>'
        for value, label in stats
    )

    cards = [
        ("When the panels actually wake",
         "Every observed first and last light across six years, against the "
         "fitted window and astronomical sunrise. Weather pushes points "
         "inward; the envelope tracks the outer edge.",
         chart_light_times(events, windows)),
        ("The model",
         "Solar elevation at each day edge. Dotted lines are the pooled "
         "5th percentiles, solid lines the 2-harmonic fit. The threshold "
         "rises by about a degree in winter, when the inverter needs more "
         "irradiance to start.",
         chart_elevation_envelope(model, events)),
        ("Shading screen",
         "A flat line means no obstruction. <em>Read this only as a "
         "screen, not a horizon profile</em> — the azimuth bins mix "
         "seasons, since morning azimuth 130° is reached both at "
         "midwinter sunrise and in summer mid-morning.",
         chart_azimuth_screen(events)),
        ("Night length through the year",
         "The complement of the solar window, in hours.",
         chart_night_length(windows)),
        ("Data coverage",
         "Generating samples per day. 2020 begins on 20 May, so early "
         "day-of-year thresholds draw on five years rather than six.",
         chart_coverage(loaded)),
    ]
    card_html = "".join(
        f'<section class="card"><h2>{title}</h2><p>{caption}</p>'
        f'<div class="chart">{svg}</div></section>'
        for title, caption, svg in cards
    )

    return (
        "<title>The Solar Window</title>"
        + STYLE
        + '<main>'
        + '<p class="eyebrow">Solar window model &middot; PVOutput history</p>'
        + '<h1>When the sun can reach the panels</h1>'
        + '<p class="lede">Six years of PVOutput history, reduced to a daily '
        + "window in which generation is physically possible. Everything "
        + "outside it is night.</p>"
        + f'<p class="coords">site <b>{LATITUDE:.4f}°N, '
        + f'{LONGITUDE:.4f}°E</b> &middot; tz <b>{SITE_TZ}</b> '
        + f'&middot; sunrise elevation <b>{SUNRISE_ELEVATION_DEG:.4f}°'
        + '</b></p>'
        + f'<div class="stats">{stat_html}</div>'
        + card_html
        + '<footer>Geometric solar elevation via pvlib, fitted as a '
        + '2-harmonic seasonal threshold on 5th-percentile pooled '
        + 'observations (±10 days).</footer>'
        + "</main>"
    )
