# Determining the daily solar-possible window from PVOutput history

**Date:** 2026-09-03
**Status:** Approved design, ready for implementation planning

## 1. Purpose

Establish, for any calendar date, the earliest moment the PV installation could
plausibly begin generating and the latest moment it could still be generating,
under clear conditions. Everything outside that window is *night*: a period with
no possible solar input, during which measured consumption is unambiguously
household draw.

Days lost to cloud, snow or outages must not shorten the window. The model
therefore describes the *envelope* of what is possible, not the average of what
happened.

This document covers the window only. Night consumption statistics are a
separate, later piece of work (§11).

## 2. Site and data

**Site:** 51.98° N, 5.80° E, timezone `Europe/Amsterdam`.

**Input:** six PVOutput history exports in the repository root,
`pvoutput_gethistory.YYYY.parquet.xz` for 2020–2025. Despite the `.xz`
suffix these are plain Apache Parquet files with internal compression; they
are read directly by `pandas.read_parquet` and must *not* be passed through
`lzma`.

5-minute interval data, 2020-05-20 → 2025-12-31, ~590,000 rows.

| Column | Unit | Notes |
|---|---|---|
| `Date` | `int64`, `YYYYMMDD` | local calendar date |
| `Time` | `str`, `HH:MM` | local wall-clock |
| `Energy Generation` | Wh | cumulative within the day |
| `Energy Efficiency` | kWh/kW | |
| `Instantaneous Power` | W | instantaneous or 5-min mean, inverter-dependent |
| `Average Power` | W | |
| `Normalised Output` | kW/kW | |
| `Energy Consumption` | Wh | cumulative within the day |
| `Power Consumption` | W | |
| `Temperature`, `Voltage` | — | **entirely empty in all six years; ignore** |

### 2.1 Established facts

These were measured during design, not assumed. They drive several decisions
below.

1. **Timestamps are local wall-clock with DST.** First light on 2023-10-25 is
   08:50 and on 2023-10-30 is 07:40, straddling the 2023-10-29 changeover.
2. **The NaN-versus-zero convention changes across years.** Non-null
   `Instantaneous Power` covers ~50% of rows in 2020–2021 but ~92% in
   2024–2025: later years write explicit zeros at night, earlier years write
   nulls. Null and zero must therefore be treated identically.
3. **There is no spurious night generation.** Across all six years, zero
   samples have positive `Instantaneous Power` before 03:00 local. The raw
   signal needs no outlier rejection.
4. **2,047 of 2,052 days recorded some generation.** Only five days are
   generation-free.
5. **Spring-forward days are correct; fall-back days lose an hour.**
   Spring-forward days hold 276 rows — a genuine 23-hour local day, with
   02:00–02:59 correctly absent. Fall-back days hold 288 rows with *no
   duplicate timestamps*, where 300 would represent the true 25 hours. One
   hour of consumption per year is unrecoverable.
6. **The lost fall-back hour leaves no detectable seam.** On all six
   fall-back days every 5-minute increment of cumulative `Energy Consumption`
   between 01:30 and 04:00 is normal-sized (2023-10-29: 9–12 Wh against a
   typical 11 Wh). Consequently **it is impossible to determine from the data
   which copy of the repeated 02:00–02:59 hour survived.** §5 handles this by
   making the question cosmetic rather than load-bearing.
7. **The horizon is effectively unobstructed.** Sun apparent elevation at
   first generation, 5th percentile by azimuth bin, is −1.2° to −1.5° across
   morning azimuths 50°–120° and about −0.2° to +0.2° across evening azimuths
   240°–310°. Geometric sunrise is −0.833°, so the panels wake on diffuse
   skylight alone. No tree or roof shading is detectable.
8. **The start threshold rises in winter.** In the deep-winter sunrise and
   sunset azimuth bins (130°–140° morning, 220°–230° evening) the same
   percentile is about +0.8° to +0.9°. This is the inverter's start-up
   irradiance threshold biting when the sun climbs at a shallow angle — not
   shading. **A fixed clock offset from sunrise would therefore be wrong in
   winter**, which is the central reason for the model in §4.

## 3. Definitions

- **Generating sample** — a 5-minute sample where *any* of the following
  hold: `Instantaneous Power > 0`, `Average Power > 0`, or `Energy
  Generation` increased relative to the previous sample of the same local
  date. This tri-condition makes 2020 and 2025 comparable despite fact 2.
  Null is treated as zero throughout. The first sample of a solar day has no
  predecessor and so satisfies the third condition only if the first two
  fail to apply — that is, it counts as generating on power alone.
- **Solar day** — a local calendar date. Grouping is local because a solar
  day *is* a local day.
- **First light / last light** — the instants of the first and last
  generating sample of a solar day.
- **Solar window** — `[solar_start, solar_end]` for a date, as produced by
  the model in §4.
- **Night** — the interval from `solar_end` on date *D* to `solar_start` on
  date *D+1*.

## 4. Model: seasonal elevation-threshold envelope

### 4.1 Rationale

Fact 8 rules out a fixed time offset from sunrise. Fact 7 rules out an
azimuth-resolved horizon profile as the primary model — there is no horizon
to resolve, and such a model would conflate obstruction with the inverter
threshold. What varies meaningfully and smoothly through the year is the
**sun elevation at which the installation starts and stops producing**. That
is the quantity modelled.

### 4.2 Fitting

1. For every solar day with generation, take the first-light and last-light
   instants and compute the sun's **apparent elevation** at each
   (`pvlib.solarposition.get_solarposition`, which accounts for refraction).
2. Assign each event a **year angle** φ = 2π · (t − start of its year) /
   (duration of its year). Using a fractional angle rather than an integer
   day-of-year handles leap years exactly and needs no special-casing of
   29 February.
3. For each target day, pool all events across all six years whose φ lies
   within a **circular ±10-day window**. This wraps the year end, so 1 January
   pools with 22 December. Yield is up to ~120 samples per target day.
4. Take the **5th percentile** of pooled elevations. Cloud-delayed mornings
   sit high in the distribution and are discarded; a single anomalous reading
   cannot drag the result, unlike a minimum.

   The **5th percentile is correct for both ends of the day**, not just the
   morning. A late last light means the sun was still *low* when generation
   stopped, so the latest possible sunset edge corresponds to the *lowest*
   elevation at last light. Morning and evening events are pooled and fitted
   separately, but with the same low percentile.
5. Smooth the resulting 366 values with a **2-harmonic Fourier fit**,

   θ(φ) = a₀ + Σ<sub>k=1,2</sub> [ a<sub>k</sub> cos kφ + b<sub>k</sub> sin kφ ]

   solved by weighted least squares (`numpy.linalg.lstsq`, weights = pooled
   sample count). Periodic by construction, so there is no discontinuity at
   the year boundary, and gap-free even where coverage is thin.

This yields two smooth curves, θ_start(φ) and θ_end(φ).

### 4.3 Applying

For a given date and threshold θ, evaluate apparent elevation on a
**one-minute grid built in UTC** spanning that local day's true extent (1380,
1440 or 1500 minutes as the case may be). `solar_start` is the first upward
crossing of θ; `solar_end` is the last downward crossing. One-minute
resolution gives ±30 s, well inside the 5-minute sampling interval.

If the sun never reaches θ on a date, both values are null and the row is
flagged. At 52° N with θ ≲ +2° and a winter-solstice maximum elevation of
about 14.5° this should never occur, but it is handled rather than assumed.

No hand-rolled astronomy: pvlib performs all solar-position calculation.

### 4.4 Rejected alternatives

- **Azimuth-resolved horizon profile.** Physically the correct model for a
  shaded site. Rejected: fact 7 shows no shading here, and it conflates
  horizon with inverter threshold, producing the misleading winter bins of
  fact 8. Retained only as a one-off diagnostic chart (§7).
- **Clock-offset percentile from sunrise/sunset.** Simplest to explain and
  rejected by fact 8: equal minute-offsets correspond to wildly unequal
  elevations in June and December.
- **Purely astronomical sunrise/sunset.** Ignores the inverter start-up
  threshold, so it would classify genuinely dark minutes as day.
- **Per-date union of all observed windows.** One anomalous sample would
  widen a window permanently, and 2020 contributes nothing before 20 May.

## 5. Time handling

DST is the largest correctness hazard in this project. The rule is:
**localize once, compute in UTC, present in local.**

- `loader` converts naive local timestamps with
  `tz_localize("Europe/Amsterdam", nonexistent="shift_forward",
  ambiguous=False)` and then **`tz_convert("UTC")`**. UTC is the internal
  representation for every subsequent computation.
- A separate `solar_date` column carries the **local** calendar date. This is
  the only place local time is authoritative, and it is correct there (§3).
- The elevation-crossing grid of §4.3 is built in UTC. Built in local time it
  would hit a one-hour hole on spring-forward days and duplicate timestamps
  on fall-back days, where `pandas.date_range` either raises or returns a
  malformed index.
- Every emitted timestamp appears **twice**: once as UTC ISO-8601 and once as
  local ISO-8601 **with explicit offset** (`2023-10-29T02:00:00+01:00`), so
  no downstream reader can misinterpret it.
- All durations are UTC subtractions. A night spanning a transition is
  genuinely 23 or 25 hours; subtracting naive local times would report 24 and
  corrupt any mean-power figure derived from it.

### 5.1 The ambiguous hour

Per fact 6 the surviving copy of the repeated fall-back hour cannot be
identified. `ambiguous=False` (standard time) is chosen **arbitrarily** and
documented as such; either choice yields a strictly monotonic, duplicate-free
UTC index with a one-hour gap in a different place.

To keep that arbitrary choice from mattering:

- Rows on affected nights carry a **`dst_hour_missing`** flag.
- Any later mean-power statistic must be computed as
  `Wh / (summed duration of samples actually present)`, never as
  `Wh / (night_end − night_start)`.

## 6. Architecture

A uv-managed project: `pyproject.toml`, `src/pvnight/`, `tests/`, and a
single entry point `analyze.py` that regenerates both data outputs and the
report. Five modules, each testable in isolation.

| Module | Responsibility | Depends on |
|---|---|---|
| `loader` | Parquet → tidy UTC-indexed frame with `generating` flag and `solar_date` | pandas, pyarrow |
| `solar` | pvlib wrapper: solar position, sunrise/sunset, elevation-crossing solver | pvlib |
| `events` | Per solar day: first/last light instants, daily yield, generating-sample count. Timestamps only — no astronomy. | loader |
| `envelope` | Attach solar elevation to events, fit and apply θ_start(φ), θ_end(φ) | events, solar, numpy |
| `report` | Charts and published HTML | envelope, matplotlib |

Dependencies are all established public packages: `pandas`, `pyarrow`,
`pvlib`, `numpy`, `matplotlib`. Nothing in the astronomy or statistics is
hand-written.

## 7. Outputs

### 7.1 `solar_thresholds.csv` — the model

366 rows, one per day of a leap reference year (2024, so all 366 slots
exist).

| Column | Description |
|---|---|
| `doy` | 1–366 |
| `theta_start_deg` | fitted θ_start, apparent elevation |
| `theta_end_deg` | fitted θ_end |
| `theta_start_raw_deg`, `theta_end_raw_deg` | pre-smoothing 5th percentiles |
| `n_samples` | pooled events in the ±10-day window |
| `n_years` | distinct years contributing |

### 7.2 `solar_windows.csv` — the usable artefact

One row per calendar date, 2020-01-01 → 2026-12-31.

| Column | Description |
|---|---|
| `date` | local calendar date |
| `solar_start_utc`, `solar_start_local` | window opens |
| `solar_end_utc`, `solar_end_local` | window closes |
| `sunrise_utc`, `sunset_utc` | pvlib astronomical reference |
| `start_offset_min`, `end_offset_min` | window edge minus sunrise/sunset |
| `night_start_utc`, `night_end_utc` | `solar_end` of *D* → `solar_start` of *D+1* |
| `night_duration_h` | UTC difference |
| `n_years` | years contributing to this date's threshold |
| `dst_hour_missing` | true on fall-back nights |
| `extrapolated` | true where no observed data backs this date (2026, and 2020 before 20 May) |

`night_start` / `night_end` are shaped so the later consumption work is a
single join.

### 7.3 HTML report

Published as a **private Artifact**, link handed to the user. Five charts:

1. Observed first and last light per date (scatter, all six years) against the
   fitted envelope and astronomical sunrise/sunset.
2. Apparent elevation at first and last light versus day of year, with the
   raw percentiles and the fitted curve — this is the chart that shows the
   winter threshold rise of fact 8.
3. Azimuth-binned horizon diagnostic, confirming fact 7.
4. Night length across the year.
5. Data-coverage heatmap by year, making 2020's partial coverage and the five
   generation-free days visible.

Charts are rendered by matplotlib to **inline SVG** with a transparent figure
background, axis furniture and text in a neutral mid-grey that reads on both
light and dark surfaces, and data colours from the `dataviz` skill palette.
The page follows the `artifact-design` and `dataviz` skills, including
theme-aware tokens.

## 8. Edge cases

| Case | Handling |
|---|---|
| DST transitions | §5 in full |
| NaN-vs-zero convention drift | tri-condition `generating` flag, §3 |
| 2020 starts 20 May | thin day-of-year coverage exposed via `n_years`; `extrapolated` flag |
| Five generation-free days | excluded from the event pool |
| 2026 dates with no data | computed from the fitted model, marked `extrapolated` |
| Sun never reaches θ | null window, flagged; not expected at this latitude |
| 5-minute sample granularity | first generating sample is treated as the instant of onset; true onset may be up to 5 minutes earlier. This biases toward a *shorter* night, which is the safe direction. Documented, not corrected. |

## 9. Testing

Test-driven throughout. The load-bearing tests:

1. **Cross-validated astronomy.** The elevation-crossing solver at
   θ = −0.833° must agree with pvlib's independent
   `sun_rise_set_transit_spa` to within one minute, across a full year. Two
   different code paths, one answer.
2. **UTC index invariant.** After loading all six years the UTC index is
   strictly monotonic and duplicate-free. One cheap assertion that catches the
   entire DST bug class.
3. **DST day shapes.** Spring-forward days load with 276 rows and no
   exception; fall-back days load with 288 rows and produce no duplicate UTC
   instants.
4. **Circular window wrap.** The ±10-day pool for 1 January must include
   events from late December.
5. **Percentile recovery.** Synthetic events with known injected elevations
   must return the injected 5th percentile.
6. **Containment regression.** The fitted window must contain first light on
   ≥95% of observed days and last light on ≥95% of observed days (≈90% of
   days at both ends, as a 5th-percentile threshold implies by construction);
   and for days that fall outside, the median violation must be under 10
   minutes. This is the guard that would catch a model that has quietly
   drifted.
7. **Sanity band.** Fitted `start_offset_min` must lie within −30 to +60
   minutes of sunrise for every day of the year.

## 10. Assumptions on record

- The site coordinates are as supplied by the user and are not fitted from
  the data.
- The choice of `ambiguous=False` is arbitrary (§5.1) and deliberately
  rendered inconsequential.
- The installation's orientation, tilt and capacity are unknown and not
  required by this model, which is empirical at the boundaries.
- Panel degradation and any hardware changes over 2020–2025 are assumed not
  to have materially moved the start-up threshold. Chart 2 will show if this
  is false, since events are colour-separable by year.

## 11. Out of scope

Per-night consumption statistics — total Wh, mean and median W, the baseload
floor, and trends by month, season and year. The `night_start` / `night_end`
columns of §7.2 exist to make that a single join against `Power Consumption`,
but it is not part of this spec.
