# Night consumption and battery sizing, measured at the meter

**Date:** 2026-09-04
**Status:** Approved design, ready for implementation planning
**Predecessors:**
`docs/superpowers/specs/2026-09-03-solar-window-design.md` (the solar window, unaffected)
`docs/superpowers/specs/2026-09-04-night-battery-design.md` (phase 2, whose consumption source is now known faulty)

## 1. Purpose

Redo the night-consumption and battery-sizing analysis against the smart
meter, because PVOutput's consumption channel stopped seeing part of the load
in December 2022. Publish the result **alongside** phase 2's, making clear
which is authoritative and how much the fault moved the answer.

## 2. Why this work exists

### 2.1 The fault, measured

Cross-checking PVOutput against the smart-meter exports in `data/meterdata/`
gives the **median per-night fraction** of true night consumption that
PVOutput's consumption channel reports. Note this is a different statistic
from the annual totals in §2.2: a median of nightly ratios, not a ratio of
annual sums. They answer different questions and do not have to match.

| period | visible fraction |
|---|---|
| 2020, 2021, 2022 to November | 0.97–0.99 |
| **November 2022** | **0.989** |
| **December 2022** | **0.778** |
| 2023 / 2024 / 2025 | 0.779 / 0.759 / 0.733 |

The step between November and December 2022 is discrete and has never
recovered. The owner reports changing nothing, and reads it as a CT clamp
coming off one of the main incoming lines — consistent with a hardware event
rather than a configuration change.

It is **not** a cleanly lost phase. One of three phases would sit at a stable
fraction near 0.667 on balanced load; the observed value starts near 0.78 and
declines year on year, which fits a lost phase whose share of load is
*growing*. The EV arrived in 2022 and would plausibly charge on it.

### 2.2 What the fault does and does not invalidate

| | status |
|---|---|
| Solar window (phase 1) | **Sound.** Generation-only; generation is measured well throughout. |
| The night-extraction method | **Sound.** Agrees with an independent meter to within 2% for 2020–2021 — a genuine external validation. |
| Night consumption 2023+ | **Understated by 24.9% (2023), 26.7% (2024), 30.8% (2025)**, as a ratio of annual totals. |
| EV nights specifically | **Worst affected.** For 2023+, ordinary nights are short by a median 1.17 kWh, EV nights by 11.39 kWh against a 34.55 kWh true figure. In 2021–22 the same split shows 0.10 and 0.34 kWh. |
| Phase 2's EV classifier | **Read a partial signal.** Its 72 nights are the charging that leaked through; more was never visible. |
| Phase 2's battery recommendation (6.5–9 kWh) | **Probably too small**, sized against a night load materially too low. |
| Daytime surplus, the 48.2% ceiling | **Unknown.** The daily gap is ~4–5 kWh and has not been split day versus night. |

### 2.3 The meter data

`data/meterdata/data_YYYY_*.xlsx`, one file per year 2020–2026.

- 15-minute intervals, 2019-12-31 23:15 UTC → 2026-09-03 22:00 UTC.
- Columns: `datum_tijd` (timestamp with `+0100`/`+0200` offset),
  `levering_normaal` / `levering_laag` (import, two tariffs, kWh),
  `teruglevering_normaal` / `teruglevering_laag` (export), and
  `buitentemperatuur` (unused; carries `-` for missing).
- **Every column reads as a string**, including the timestamp and all four
  energy columns (`'0,04'`), so each needs explicit parsing. Decimal comma.
- The two tariff columns of each pair are **null when the other tariff is
  active** — in 2020, `levering_normaal` has 18,752 nulls and `levering_laag`
  16,384, summing to exactly the 35,136 rows. So null means "not this tariff",
  and filling with zero before summing the pair is correct, not a fudge.
  Export rows can have *both* null, meaning no export that interval.
- Timestamps label the interval **end**: the first row of 2020 is
  `01-01-2020 00:15:00 +0100`.
- 233,358 rows against 234,044 the span implies **686 missing (0.3%)**, in
  **seven** gaps — and they are not evenly spread:

  | gap start (UTC) | duration |
  |---|---|
  | 2022-02-15 12:15 | 2 h 30 m |
  | 2023-07-21 06:15 | 1 h 30 m |
  | **2024-01-08 23:00** | **5 d 9 h 45 m** |
  | 2024-01-14 23:00 | 10 h 15 m |
  | 2024-01-15 23:00 | 24 h 15 m |
  | 2024-01-18 23:00 | 2 h 15 m |
  | 2024-07-19 07:45 | 2 h 45 m |

  Four of the seven fall in **January 2024**, together removing most of
  8–19 January (the fifth 2024 gap is in July) — midwinter, when night consumption is at its annual peak. Any
  2024 figure must exclude affected nights rather than treat them as low
  ones, and the report must say how many nights that removes.
- **31,528 intervals (13.5%) carry both import and export.** §4.2 turns this
  into a bound rather than an assumption.
- At 15-minute resolution, **16% of night energy is drawn above 3 kW**,
  against the 5.8% PVOutput reported — because the EV is finally visible.
  This strengthens phase 2's conclusion that a single-phase 3 kW inverter
  cannot serve the car.

## 3. Scope decisions

Taken with the owner, recorded because each shaped the design:

1. **Publish alongside phase 2, not superseding it.** Both analyses appear;
   the page states which to trust. The risk this accepts is someone quoting
   the superseded figure, so §6 makes the labelling a hard requirement rather
   than a matter of tone.
2. **Full meter span, 2020-01-01 to 2026-09-03.** The simulation needs only
   import and export, so PVOutput's end date does not bind. Gains eight
   months including a 2026 summer. Night windows for 2026 already exist from
   phase 1.
3. **Simulate at 15 minutes and measure the penalty**, rather than
   reconstructing a 5-minute shape from PVOutput — that shape is the broken
   signal for 2023 onward and would import the fault this work exists to
   correct.
4. **Run both bounds** of the both-flows question (§4.2).

## 4. Model

### 4.1 The signal

A battery interacts only with the grid connection: it absorbs what would
otherwise be exported and supplies what would otherwise be imported. The
meter measures exactly those two channels, so **no reconstructed consumption
is required** and the analysis is not limited to PVOutput's span.

Reconstructing `consumption = generation + import − export` and simulating on
`generation − consumption` yields an algebraically identical signal, so that
choice affects reporting only. It is rejected as the primary path because it
requires generation, which does not exist for 2026.

Night consumption is **meter import summed over the night window**. At night
generation is zero, so import *is* household consumption — the two
definitions coincide exactly where this analysis needs them to.

Daytime surplus is **meter export**: energy that demonstrably had nowhere to
go, which is precisely what a battery could have captured.

### 4.2 The both-flows bound

13.5% of intervals record both import and export, because a quarter hour is
long enough to do both.

- Simulating on `net = export − import` assumes the battery could not have
  caught both within the interval. This **understates** what it could do.
- Simulating on the two flows separately assumes it caught both. This
  **overstates**.

Both are run and the pair is reported as a range. Neither is presented as the
answer, and no midpoint is invented.

### 4.3 Parameters

Unchanged from phase 2 so the comparison is like for like: round trip 90%
split symmetrically as `sqrt(0.90)` each way, usable fraction 90%, charge and
discharge capped at 3.0 kW single-phase, capacity swept 0–30 kWh in 0.5 kWh
steps, power sensitivity 2.5 / 3.0 / 3.7 kW.

`battery.simulate` gains a `dt_hours` keyword defaulting to the existing
`DT_HOURS`, so phase 2's behaviour is untouched and phase 3 passes 0.25.

### 4.4 Reading the result

Phase 2's threshold-free readings carry over unchanged: the elbow, the
share-of-achievable-benefit table, the convergence table spanning every
method, and the elbow-stability table. **No cut-off is introduced.** Phase 2
removed its 50 kWh/yr threshold because it was a judgement call rather than a
derived figure, and nothing here changes that.

### 4.5 The resolution penalty, measured

Averaging over 15 minutes hides short peaks, so a simulated battery looks
better than reality. Rather than stating this, measure it: take phase 2's own
5-minute PVOutput data, downsample to 15 minutes, re-run the sweep, and report
the difference in recommended capacity and self-sufficiency. Both resolutions
genuinely exist for that dataset, so the penalty is measured on real data
rather than modelled.

## 5. Architecture

| file | responsibility |
|---|---|
| `src/pvnight/meter.py` | xlsx → 15-minute UTC frame. Decimal comma, both offsets, gap detection. |
| `src/pvnight/meter_nights.py` | Per-night import from the meter; EV classification on a signal where the car is visible. |
| `src/pvnight/battery.py` | **Modified:** `simulate(..., dt_hours=DT_HOURS)`. No other change. |
| `src/pvnight/compare_report.py` | The side-by-side page. |
| `analyze_meter.py` | Entry point. `analyze.py` and `analyze_night.py` are untouched. |

No phase-1 module is modified. Of phase 2, only `battery.py`'s signature
changes, additively.

### 5.1 The EV classifier, re-derived

Phase 2's rule — two hours above 2 kW — was fitted to a signal missing much
of the car. It is **re-derived against the meter**, and the sensitivity grid
is republished, because the threshold that separated a partial signal is not
necessarily the one that separates a complete one. The new classification is
compared against phase 2's 72 nights, and the difference reported.

## 6. Outputs

**`out/meter_night_summary.csv`** — one row per night: `date`,
`night_start_utc`, `night_end_utc`, `import_kwh`, `export_kwh`, `peak_kw`,
`hours_above_2kw`, `is_ev`, `missing_intervals`, `covered`. Note `covered` is a boolean, not a ratio as in phase 2: the meter either recorded an interval or it did not, so there is no partial coverage to express.

**`out/meter_battery_sweep.csv`** — one row per (capacity, power, bound):
the phase-2 metric set plus a `bound` column taking `net` or `gross`.

**HTML report**, published as a private Artifact. It must:

1. **Lead with the meter-based figures** and label phase 2's as measured
   through a faulty channel from December 2022. This is a requirement, not a
   stylistic preference: side-by-side publication is only safe if the page
   says which number is which.
2. Show the November→December 2022 step as the evidence, with the
   0.989 → 0.778 transition visible.
3. Report the both-flows range rather than a point estimate.
4. Report the measured resolution penalty from §4.5.
5. Restrict every phase-2 comparison to the overlapping period, so
   differences are attributable to the source and not the span.

Charts follow the `dataviz` and `artifact-design` skills. Matplotlib to
inline SVG, transparent background, **no embedded rasters** — an embedded PNG
trips the publish content scanner, established in phase 1. Use `pcolormesh`,
never `imshow`.

## 7. Testing

Test-driven. The load-bearing tests:

1. **The agreement, 2020–2021.** Meter and PVOutput night totals agree within
   3%. This is the evidence the night-extraction method is sound, and it must
   keep passing.
2. **The divergence, 2023–2025.** The same comparison shows an annual-total
   shortfall of 24.9%, 26.7% and 30.8% respectively. Assert each year is
   between **20% and 35%** — wide enough that the boundary value of 24.9%
   does not sit on a cliff edge, narrow enough that the fault must still be
   there. The fault is a fact about the data; a test that stops failing here
   means something upstream changed.
3. **The December 2022 step.** November 2022 ≥ 0.95 and December 2022 ≤ 0.85.
4. **Meter parsing.** Decimal comma yields floats, not strings; `+0100` and
   `+0200` rows both land at the right UTC instant; the seven gaps are
   detected and counted rather than silently interpolated; and a night
   overlapping the January 2024 outage is excluded rather than counted as a
   low-consumption night.
5. **Both bounds ordered.** For every capacity, the gross bound's grid import
   is less than or equal to the net bound's. A violation means the bounds are
   swapped or the dispatch is wrong.
6. **`dt_hours` actually binds.** The same synthetic series simulated at
   5 and 15 minutes gives a different power-cap outcome — otherwise the
   parameter is decorative.
7. **Phase 2 is unchanged.** Its full suite still passes, and
   `simulate` called without `dt_hours` reproduces its previous numbers.

## 8. Assumptions on record

- The meter is treated as ground truth for energy crossing the grid
  connection. Its own calibration is not independently verifiable here.
- The CT-clamp explanation is the owner's reading and is consistent with the
  evidence, but is **inferred, not confirmed**. The analysis does not depend
  on the cause — only on the measured undercount.
- Night consumption equals night import because generation is zero at night.
  On nights where the solar window is `extrapolated` (2026, and before
  2020-05-20) the window itself is modelled, though the import is measured.
- Temperature is present in the meter export and ignored.

## 9. Out of scope

- **Cost, payback and tariffs.** Still absent, so the answer remains in kWh.
  The meter's two tariff columns are summed, not used to price anything.
- **EV smart-charging.** Now more clearly valuable, since 16% of night energy
  exceeds the 3 kW cap at meter resolution, but it needs its own model.
- **Re-fitting the solar window.** Phase 1 is generation-based and unaffected.
- **Repairing the CT clamp**, which is physical work, and re-deriving phase 2
  once it is fixed.
