# pvnight

This repository determines, for every calendar date from 2020 through 2026, the
window in which solar generation is physically possible at a PV installation in
the Netherlands, fitted from six years of PVOutput history. "Night" is defined
as the complement of that window: the period during which any measured
consumption is unambiguously household draw, not solar shortfall.

## Running it

```
uv run python analyze.py
```

This loads the PVOutput history parquet files in the repository root, fits the
seasonal elevation-threshold envelope, and writes three files to `out/`:
`solar_thresholds.csv`, `solar_windows.csv`, and `report.html`. Run
`uv run pytest` to run the test suite.

## Outputs

**`out/solar_thresholds.csv`** — the fitted model itself, one row per day of a
366-day reference year (`doy` 1–366). Gives the fitted solar-elevation
threshold at which generation starts and stops that day (`theta_start_deg`,
`theta_end_deg`), the corresponding raw (unsmoothed) percentile estimates, and
the sample/year counts behind each fit.

**`out/solar_windows.csv`** — the model applied to every calendar date from
2020-01-01 to 2026-12-31 (2,557 rows). Each row gives the solar-possible
window in UTC and site-local time (`solar_start_utc`/`local`,
`solar_end_utc`/`local`), the corresponding astronomical sunrise/sunset for
comparison, and the complementary night window: **`night_start_utc`** and
**`night_end_utc`** mark the start and end of that date's night, with
`night_duration_h` its length. Rows for dates without observed generation
(before 2026, none; some future dates in this range) are flagged
`extrapolated`.

**`out/report.html`** — a static HTML page with five charts summarising the
fit and its outputs (light times, elevation envelope, azimuth-at-crossing,
night length across the year, and data coverage).

## Reasoning

See `docs/superpowers/specs/2026-09-03-solar-window-design.md` for the full
design: why an elevation-threshold model, how the seasonal fit works, and the
established facts about the source data that drove those choices.

## Not yet built

Night consumption statistics are **not** implemented here. This repository
only produces the window/night boundaries. A future piece of work will join
household consumption data against `night_start_utc`/`night_end_utc` in
`solar_windows.csv` to compute night-only consumption statistics.
