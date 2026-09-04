# Night consumption and battery sizing

**Date:** 2026-09-04
**Status:** Approved design, ready for implementation planning
**Predecessor:** `docs/superpowers/specs/2026-09-03-solar-window-design.md` (the solar
window this work consumes)

## 1. Purpose

Decide what size of home battery is worth installing to carry daily PV
production through the night, and give the reasoning behind that number rather
than the number alone.

The answer must account for EV charging, which dominates a small number of
nights, and for a **single-phase inverter capped at 3 kW**, which is the
hardware actually under consideration.

## 2. What phase 1 established, and what this work adds

Phase 1 produced `out/solar_windows.csv`: for every date 2020-01-01 to
2026-12-31, the window in which solar generation is physically possible, and
the complementary night interval as `night_start_utc` / `night_end_utc`. That
table is consumed unchanged. No phase-1 module is modified.

This work adds per-night consumption, EV classification, a battery simulation,
and a report.

### 2.1 Established facts

Measured during design against the real data, not assumed. Percentages are of
the 2,031 nights with better than 95% sample coverage, spanning 2020-05-20 to
2025-12-30.

1. **Night consumption is heavily right-skewed.** Median 4.85 kWh, p90 10.52,
   p99 29.66, max 44.53. The mean (6.04) exceeds the median by 25%, so every
   headline figure must be a median or a percentile, never a mean.

2. **Night consumption is strongly seasonal**, driven by both longer nights and
   higher load:

   | | Dec | Jan | Jul | Jun |
   |---|---|---|---|---|
   | median night | 9.31 | 9.09 | 2.75 | 2.80 kWh |

3. **Winter production cannot charge a battery at all.** Median daytime surplus
   (generation minus consumption *within* the solar window) is **negative** in
   November (−2.15), December (−3.50) and January (−3.44 kWh), and about zero
   in October (−0.01) and February (−0.31). In those months the house does not
   generate enough to cover even its daytime load, so no battery of any size
   receives a charge.

4. **Only 48.2% of nights could be fully covered by that day's surplus even
   with an infinite battery.** By month: June 90.6%, July 86.5%, September
   64.4%, March 52.9%, October 11.0%, and **0.0% in each of November, December
   and January**. The binding constraint on this installation is winter
   generation, not battery capacity — which is why §4 optimises the marginal
   value of capacity rather than a coverage target.

5. **The EV charges in two distinct modes**, and a single power threshold
   catches only one of them:

   | Mode | Power | Duration | Example |
   |---|---|---|---|
   | Fast | ~8 kW | 3–4 h | 2023-12-28, 44.5 kWh night, peak 9,684 W |
   | Slow | ~3.5 kW | 12+ h | 2022-11-13, 42.9 kWh night, peak 3,732 W |

   A "≥1 h above 5 kW" rule finds 53 nights and misses every slow-mode night.

6. **"≥2 h above 2 kW" separates EV nights cleanly:** 72 nights, median
   24.5 kWh, against 4.71 kWh for the remaining 1,959. This is the rule adopted
   in §3, with a published sensitivity table (§7.3) because it is a heuristic.

7. **EV charging began in 2022.** Nights with ≥1 h above 5 kW: zero in 2020 and
   2021, then 7 (2022), 20 (2023), 12 (2024), 14 (2025). Yearly maximum night
   consumption steps from ~16.5 kWh (2020–21) to ~43 kWh (2022 onward).

   **This was measured with the ≥1 h above 5 kW rule, not the rule §3 adopts.**
   The adopted "≥2 h above 2 kW" classifier additionally flags six pre-2022
   winter evenings — 2020-11-15, 2020-12-13, 2021-03-15, 2021-11-29,
   2021-12-19, 2021-12-25 — each 14.7 to 16.5 kWh with a 3.5–5.1 kW load for
   about two hours, and none showing the fast-charge signature. Plausibly a
   heat pump, oven or dryer; certainly not a car that did not exist yet.

   These six are a **known false-positive rate of 6 in 72**, and they are
   already included in fact 6's counts, so those figures stand. The rule is
   deliberately *not* tightened to remove them: six nights out of a 1,959-night
   household pool cannot move the sizing, and a heuristic whose limits are
   visible is worth more than one tuned until a claim comes true. The §3
   sensitivity table exists to expose exactly this.

8. **A 3 kW discharge cap is nearly free on ordinary nights and crippling on EV
   nights.** Share of night energy drawn above 3 kW, which no 3 kW battery can
   supply at any capacity:

   | | nights | total | above 3 kW |
   |---|---|---|---|
   | Non-EV | 1,959 | 10.47 MWh | **1.6%** |
   | EV | 72 | 1.79 MWh | **30.1%** |

   The median night draws 0% above the cap; the p90 night 1.7%. This is
   independent support for sizing on the non-EV curve (§4.3).

9. **The 3 kW charge cap is unlikely to bind.** It clips 15.6% of instantaneous
   surplus *flow* in June, but 21.2 kWh/day still passes the cap against a
   battery likely sized near 10 kWh, so it fills long before the clipping
   costs anything. Stated as an expectation the simulation must confirm, not
   as a finding.

10. **Consumption nulls are preserved, not zero-filled** (inherited from phase 1,
    spec §3 there): `power_cons_w` has 1,505 rows with no reading and
    `energy_cons_wh` has 1,315, concentrated in 2020 and 2022. A null means
    unrecorded, not zero. Nights below 95% sample coverage are therefore
    excluded from per-night statistics (§3).

## 3. Night aggregation and EV classification

**Night energy.** `energy_cons_wh` is cumulative within a local day and resets
at midnight, so per-sample increments are `groupby("solar_date").diff()`,
matching phase 1's treatment of generation. A night's energy is the sum of
increments whose timestamp falls in `[night_start_utc, night_end_utc)`. Verified:
zero negative increments across the dataset, so the counter never resets
mid-day.

**Coverage filter.** A night is included when it has more than 95% of the
samples its duration implies (duration ÷ 5 min). This drops 20 of 2,051 nights.
The seven nights flagged `dst_hour_missing` retain their energy but are excluded
from any *rate* statistic (mean W, or Wh per hour), because their true duration
and their sampled duration differ by an hour.

**EV classification.** A night is EV-charging when at least 2 hours of its
samples exceed 2,000 W. Chosen over the more obvious 5 kW test because fact 5
shows slow-mode charging never reaches 5 kW.

This is a heuristic and is presented as one. The report publishes a sensitivity
table across thresholds 1.5/2.0/2.5/3.0 kW crossed with durations 1/2/3 h,
showing night counts and median energies, so a reader can judge whether the
split is stable or an artefact of where the line was drawn. If the classified
count swings wildly across that grid, the heuristic is weak and the report says
so rather than burying it.

## 4. Battery model

### 4.1 Simulation

A chronological pass over all 5-minute samples. At each step
`net = power_gen_w − power_cons_w`; surplus charges the battery, deficit
discharges it, both subject to the power cap, the remaining headroom or charge,
and efficiency. Whatever the battery cannot supply is grid import; whatever it
cannot absorb is export.

Capacity is swept 0 to 30 kWh in 0.5 kWh steps. All capacities advance together
as a vector at each timestep, so the sweep is a single pass over the data rather
than 61 passes.

Rejected alternatives:

- **Daily energy balance.** Roughly 50× faster and far simpler, but structurally
  incapable of representing a power cap — which is precisely where the 8 kW EV
  draw bites (fact 8). Rejected on that ground alone.
- **Hybrid** (daily for the sweep, 5-minute for validation). Unnecessary; the
  vectorised 5-minute sweep is already fast enough.

### 4.2 Parameters

| Parameter | Default | Note |
|---|---|---|
| Round-trip efficiency | 90% | split symmetrically: √0.9 ≈ 0.9487 on charge, √0.9 on discharge |
| Usable depth of discharge | 90% | a 10 kWh nameplate holds at most 9 kWh at the terminal |
| Charge power cap | 3.0 kW | single-phase inverter |
| Discharge power cap | 3.0 kW | single-phase inverter |
| Capacity sweep | 0–30 kWh, 0.5 kWh steps | |
| Power sensitivity | 2.5 / 3.0 / 3.7 kW | 3.68 kW is 16 A at 230 V, the single-phase ceiling |

Capacities are reported as **nameplate**, the number a product is sold under,
not as usable energy.

Stored energy is tracked at the battery terminal. Filling a 10 kWh nameplate
battery to its 9 kWh usable ceiling therefore consumes 9 ÷ 0.9487 ≈ **9.49 kWh
of PV**, and emptying it delivers 9 × 0.9487 ≈ **8.54 kWh to the house** — an
8.54/9.49 = 90% round trip, as specified. Do not instead apply the full 90% to
one side only; that convention gives 8.1 kWh and is a different battery.

### 4.3 Dispatch and scenarios

Dispatch is realistic self-consumption: charge whenever production exceeds load,
discharge whenever load exceeds production, at any hour. This is how a real
home battery behaves, and a night-only policy would leave it idle through
daytime deficits and so understate its value. Night-specific results are
reported separately, so the original question is still answered directly.

Two scenarios are simulated and both are plotted:

- **All nights** — every night as it occurred.
- **Non-EV nights** — EV nights excluded from the reported night metrics.

**The simulation itself always runs the complete, unbroken timeline.** EV
nights are never removed from it: the battery really was drained by that
charging, and the following morning really did start from a lower state of
charge. The non-EV scenario changes only which nights are *counted* when
aggregating night metrics. Deleting EV nights from the timeline would leave the
battery unphysically full and overstate every result.

The recommendation is taken from the non-EV curve, because a 25 kWh EV charge is
a poor claim on scarce winter capacity and, per fact 8, a third of it cannot pass
a 3 kW inverter anyway. The all-nights curve is shown alongside so the cost of
that choice is visible rather than assumed.

### 4.4 Metrics per capacity

Grid import (kWh/yr), PV export (kWh/yr), night grid import (kWh/yr), night
self-sufficiency (%), and full-equivalent cycles per year.

The knee is located on **marginal grid import avoided per additional kWh of
capacity**. "Where the curve flattens" is not implementable as written, so the
rule is explicit: the recommended capacity is the **smallest capacity whose
marginal return has fallen below 50 kWh/yr per additional kWh**. An extra kWh
of battery earning less than 50 kWh/yr is cycling under about once a week, at
which point it is hard to justify buying.

The 50 kWh/yr figure is a stated judgement, not a derived constant. The report
therefore prints the full marginal-return table beside the curve and marks the
chosen knee on it, so a reader who prefers a different cut-off can read their
own answer straight off the same chart.

Cycles per year is reported because a large battery that rarely cycles is poor
value regardless of the energy it shifts.

## 5. Architecture

Four new files. No phase-1 module is modified.

| File | Responsibility | Depends on |
|---|---|---|
| `src/pvnight/nights.py` | Per-night energy from the window table; EV classification | loader, phase-1 window CSV |
| `src/pvnight/battery.py` | The simulation. Pure functions, no I/O | numpy |
| `src/pvnight/night_report.py` | Charts and page | nights, battery, matplotlib |
| `analyze_night.py` | Entry point | all of the above |

`analyze.py` is left untouched, so phase 1 remains independently runnable.

## 6. Outputs

**`out/night_summary.csv`** — one row per night: `date`, `night_start_utc`,
`night_end_utc`, `night_wh`, `peak_w`, `hours_above_2kw`, `is_ev`, `coverage`,
`dst_hour_missing`.

**`out/battery_sweep.csv`** — one row per (capacity, power cap, scenario):
`capacity_kwh`, `power_kw`, `scenario`, `grid_import_kwh_yr`,
`pv_export_kwh_yr`, `night_grid_import_kwh_yr`, `night_self_sufficiency_pct`,
`cycles_per_yr`, `marginal_kwh_per_kwh`.

**HTML report**, published as a private Artifact, with eight charts:

1. Night consumption distribution, EV against non-EV.
2. Night energy by month, median and spread.
3. Daytime surplus against night need by month — the chart that shows the
   winter wall of facts 3 and 4.
4. The knee curve: grid import avoided against capacity, both scenarios.
5. Marginal kWh avoided per additional kWh of capacity, with the chosen knee
   marked.
6. Battery utilisation by month, exposing winter idleness.
7. Night self-sufficiency by month at the recommended capacity.
8. Share of night energy above the 3 kW cap, by month and by EV/non-EV.

Charts follow the `dataviz` and `artifact-design` skills, as in phase 1:
matplotlib to inline SVG, transparent figure background, no embedded rasters
(phase 1 established that an embedded PNG trips the publish content scanner —
use `pcolormesh`, never `imshow`).

## 7. Testing

Test-driven. The load-bearing tests:

1. **Energy conservation.** Over a full simulation,
   `generation + grid_import = consumption + export + losses + Δstored`, to
   within floating-point tolerance. This single assertion catches most
   dispatch bugs.
2. **Monotonicity.** Increasing capacity never increases grid import, and never
   decreases self-consumption. A violation means the dispatch has a state bug.
3. **Power cap binds.** On a synthetic night drawing 8 kW against a 3 kW cap
   with a full battery, exactly 3 kW comes from the battery and 5 kW from the
   grid.
4. **Degenerate cases.** Zero capacity reproduces the no-battery grid import
   exactly; a battery that never sees surplus never discharges.
5. **Round-trip losses are actually applied.** Storing then withdrawing X kWh
   returns 0.9X, not X.
6. **Night aggregation.** Per-night sums reproduce the measured median of
   4.85 kWh and p90 of 10.52 kWh across the covered nights.
7. **EV classification.** Reproduces 72 nights, none before 2022, and the
   2023-12-28 and 2023-11-13 nights are both classified EV despite their very
   different power profiles.

## 8. Assumptions on record

- Battery efficiency and usable depth are generic values, not a specific
  product's datasheet. Both are parameters; the report states them.
- Consumption measured by PVOutput is whole-house draw. No sub-metering exists,
  so EV charging is inferred from the aggregate signal, never measured directly.
- The six-year record spans a hardware regime change in the PV reporting
  (predecessor spec §10) and the arrival of the EV in 2022. Night consumption
  before 2022 therefore describes a different household load than after it. The
  report shows per-year figures rather than pooling six years into one number.
- No degradation, calendar ageing, or temperature derating is modelled. A real
  battery loses capacity over its life, so the recommended size is a
  beginning-of-life figure.

## 9. Out of scope

Deliberately excluded, each a separate question:

- **Cost, payback and tariffs.** No prices are modelled, so the report says
  which capacity stops paying in *kWh*, never in euros.
- **Grid charging on a cheap night tariff**, which would change the winter
  conclusion entirely, since fact 3 shows solar cannot fill the battery then.
- **EV smart-scheduling** — moving charging into daylight. Fact 8 suggests this
  may be worth more than any battery, but it needs its own model.
- **Export limits and net-metering rules.**
