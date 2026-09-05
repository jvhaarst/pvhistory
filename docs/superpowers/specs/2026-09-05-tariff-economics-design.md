# Phase 4: pricing the battery against a real tariff

**Status:** design, approved in chat 2026-09-05.
**Supersedes:** nothing. Extends phase 3 (`2026-09-04-meter-based-night-design.md`).

## 1. Why this phase exists

Every number this project has published is energy. The battery recommendation
has therefore never had an exchange rate: sizing trades a **stock** (kWh
bought) against a **flow** (kWh/yr saved), and nothing in the data said what
one is worth in terms of the other.

Two attempts to bridge that gap failed, and both failures are on the record:

- A **50 kWh/yr per added kWh** threshold was invented, published, and removed
  at the user's instruction because it was not derived from anything.
- The **geometric elbow** that replaced it turned out not to be
  assumption-free: it drifts with where the capacity sweep is truncated, and
  on phase 3's data it never converges at all (8.5 kWh at a 25 kWh sweep,
  9.0 at 30, 11.0 at 60). Phase 3 publishes 9.0 kWh as a *lower bound* and
  leans on a 6.5–10.0 kWh spread of threshold-free readings.

`data/Greenchoice - Variabele kosten - 2027` supplies the missing rate. This
phase converts the analysis to euros, which is the only non-arbitrary way to
choose a capacity.

## 2. The tariff, as read

Verbatim from the file, which gives **three** prices per band:

| band | when (local) | season | levering €/kWh | terugleverkosten €/kWh | terugleververgoeding €/kWh |
|---|---|---|---|---|---|
| Normaal | 07:00–10:00, 17:00–22:00 | both | 0.30566 | 0.07049 | 0.08050 |
| Dal | 00:00–07:00, 22:00–24:00 | both | 0.27939 | 0.05964 | 0.06965 |
| Dal | 10:00–17:00 | winter | 0.27939 | 0.05964 | 0.06965 |
| SuperDal | 10:00–17:00 | summer | 0.18225 | 0.01951 | 0.02951 |

Summer is **1 April – 30 September**; winter is **1 October – 31 March**.
These are calendar dates, unrelated to DST. Each season's bands tile the full
24 hours with no gap and no overlap; the implementation must assert that.

Only three distinct price rows exist. The season matters solely because the
10:00–17:00 window is SuperDal in summer and Dal in winter.

## 3. What the tariff implies, and why it changes the answer

**Net feed-in is €0.01/kWh, flat.** In every band the *terugleverkosten*
claw back all but one cent of the *terugleververgoeding*:

```
0.08050 − 0.07049 = 0.01001    (Normaal)
0.06965 − 0.05964 = 0.01001    (Dal)
0.02951 − 0.01951 = 0.01000    (SuperDal)
```

So there is no net metering left in this contract. A kWh exported earns one
cent; the same kWh stored and later consumed avoids **17 to 30 cents**. That
is an 18–30× multiplier on self-consumption (30.5× in Normaal, 27.9× in Dal,
18.2× in SuperDal), and it is the whole reason a battery can be worth buying
here.

**The objective changes.** Phase 3 sized on non-EV *night* grid import.
Euros must be measured on **total** grid import: a battery also serves a dark
winter afternoon in the Normaal band, and that energy is expensive.
`battery.simulate` already discharges on any deficit rather than only at
night, so this requires no dispatch change — but it is a different objective
and may give a different capacity. That is expected, not a discrepancy.

**The winter wall still binds.** Phase 3 measured five months with a negative
median daytime surplus and a 42.9% coverable ceiling. Euros do not create
energy: a month with nothing to charge from earns nothing at any capacity.
The euro optimum is the meeting point of an expensive import price pushing
size up and the wall capping what can be stored.

## 4. Scope

**In scope.**

1. Parse the tariff file into a band table.
2. Assign every meter interval to a band.
3. Price the existing self-consumption dispatch at every swept capacity.
4. €/yr saved per capacity, broken down per calendar year.
5. Break-even installed cost per kWh, and a worked example at an assumed
   price band, labelled as an assumption rather than a quote.
6. A euro-derived capacity recommendation, plus the **derived** kWh/yr
   threshold that retires the invented 50.
7. A perfect-foresight arbitrage ceiling, reported as a bound only.

**Out of scope, and stated on the page rather than silently omitted.**

- **Standing charges, network tariffs and taxes.** They do not vary with
  battery capacity, so they cancel in every comparison. The page says so.
- **Battery degradation and cycle-life limits.** No warranty data. The cycle
  count per year is already computed and is published beside the euro figure
  so a reader can check it against a datasheet themselves.
- **Discounting.** Payback is stated in undiscounted years, and labelled.
- **Price-aware dispatch.** Only its ceiling is measured (section 9).
- **Any change to phase 3's energy figures.** They remain as published.

## 5. Modules

| file | responsibility |
|---|---|
| `src/pvnight/tariff.py` | parse the file; the band table; `band_index(ts_utc)` |
| `src/pvnight/economics.py` | price a sweep; €/yr; break-even; recommendation |
| `src/pvnight/battery.py` | **one additive keyword** on `simulate` |
| `src/pvnight/compare_report.py` | the euro cards |
| `analyze_meter.py` | wiring, new outputs, new summary keys |

### 5.1 The one change to `battery.simulate`

`simulate` already accumulates `monthly[month_idx[t]] += deliver`. Band
pricing needs the same pattern for both flows:

```python
def simulate(net_wh, night_mask, nonev_mask, month_idx, spec,
             dt_hours=DT_HOURS, band_idx=None):
```

When `band_idx` is not None, `SimResult` additionally carries
`band_grid_import_wh` and `band_export_wh`, each shaped `(n_bands, n_caps)`.
When it is None, behaviour and results are **bit-identical** to today.

All existing call sites pass five positional arguments plus an optional
`dt_hours`, so this is safe; it is the same additive pattern `dt_hours`
itself used.

**A second dispatch loop must not be written.** That loop is the most
correctness-critical code in the repository, it is what phases 2 and 3 both
rest on, and a copy would drift from it silently.

## 6. Band assignment

`tariff.band_index(ts_utc) -> np.ndarray` maps UTC timestamps to band rows.

**Local time is mandatory here.** The band boundaries are wall-clock
Amsterdam times, so `07:00–10:00` moves against UTC twice a year. This
project is UTC-internal by discipline, and this is one of the few places that
must convert: `ts_utc.tz_convert(SITE_TZ)`, then read `.hour` and `.month`/
`.day` from the local timestamps.

Getting this wrong is not cosmetic. An hour misassigned between Normaal and
SuperDal is a 12.3 cent/kWh error on that energy, and the error would land
systematically in summer afternoons — the single largest block of export in
the record.

**Rules.**

- Season by local calendar date: `4 <= month <= 9` is summer.
- Hour by local wall-clock hour, half-open intervals `[start, end)`.
- The two DST transition days are handled by `tz_convert` itself; no
  arithmetic on naive local times anywhere.
- The band table must be asserted to tile 24 hours in each season.

## 7. The pricing model

For capacity `c`, with band-resolved flows from the simulator:

```
cost(c) = Σ_b [ import_b(c) · levering_b
              + export_b(c) · terugleverkosten_b
              − export_b(c) · terugleververgoeding_b ]
```

`cost(0)` is the measured variable bill: what this household would actually
have paid under the 2027 tariff with no battery. It is a useful anchor and
is published.

```
saving(c) = cost(0) − cost(c)
```

Round-trip losses need no special term: energy lost is simply energy that
appears in neither `import_b` nor `export_b`.

### 7.1 Per-year figures

Costs are accumulated per calendar year as well as in total. **2020–2025 are
full years; 2026 ends on 3 September and must be labelled partial or
excluded from any €/yr figure** — never annualised silently. The headline
€/yr is the most recent *full* year; the spread across years is published
beside it, because a cloudy year and a sunny year may differ by more than
the capacity choice does.

The EV arrived in 2022. The per-year table will show that, and should: it is
information, not noise.

## 8. Break-even and recommendation

**Break-even installed cost**, the durable artefact — needs no quote and does
not go stale:

```
break_even_eur_per_kwh(c, years) = saving_per_yr(c) · years / c
```

Read as: at this installed price per kWh, capacity `c` exactly repays itself
in `years`. Undiscounted, degradation ignored, both stated.

**With a quote**, payback and the optimum:

```
payback_years(c) = (fixed_cost + c · eur_per_kwh) / saving_per_yr(c)
recommend(eur_per_kwh, fixed_cost, years) =
    argmax_c [ saving_per_yr(c) · years − fixed_cost − c · eur_per_kwh ]
```

**The worked example** uses an assumed installed price band, stated as an
assumption in the same sentence as the result, never as a quote.

### 8.1 Retiring the invented threshold

`battery.recommend_capacity` takes a kWh/yr-per-kWh threshold. It was kept
and tested precisely for this moment. The threshold is now derivable:

```
threshold_kwh_per_yr_per_kwh = eur_per_kwh / (years · blended_value_eur_per_kwh)
```

where `blended_value` is the measured euro value of a marginal avoided kWh,
computed from the band mix the battery actually displaces — not assumed.

The report must state what this derived threshold is and how it compares to
the removed 50. If they differ materially, say so plainly: it means the
invented number was biasing the recommendation, and in which direction.

## 9. The arbitrage ceiling

The tariff makes grid charging profitable on paper — buy SuperDal at
0.18225, displace Normaal at 0.30566, a 12.3 cent gross spread against a
~10% round trip. This phase does **not** build a dispatch policy for it.

It measures the ceiling: with perfect foresight over the whole record, how
much could price-aware charging add on top of self-consumption? Hindsight is
legitimate for an upper bound and illegitimate for a recommendation, and the
page must say which it is in the same breath as the number.

Its only purpose is to answer whether a phase 5 is worth building.

## 10. Outputs

- `out/meter_tariff_bands.csv` — the parsed band table, so the input to every
  euro figure is inspectable.
- `out/meter_economics.csv` — per capacity, per year: cost, saving,
  break-even €/kWh, cycles/yr.
- New cards on `out/meter_report.html`: the tariff and its 18–30× spread;
  €/yr saved against capacity; the break-even curve; the arbitrage ceiling.
- New `run()` summary keys: `annual_saving_eur`, `breakeven_eur_per_kwh`,
  `derived_threshold_kwh_per_kwh`, `arbitrage_ceiling_eur_yr`,
  `no_battery_cost_eur`.

## 11. Verification

Tests that must exist, because each covers a way this can be wrong without
looking wrong:

1. **DST transition days.** An interval at 02:30 local on both the March and
   October transition dates lands in the correct band, and no interval is
   unassigned.
2. **Season boundary.** 31 March 16:00 local is Dal; 1 April 16:00 local is
   SuperDal. Same clock hour, different band.
3. **Winter afternoon is not SuperDal.** 15 January 12:00 local is Dal.
4. **The band table tiles the day**, in both seasons, with no gap or overlap.
5. **`simulate` is unchanged when `band_idx` is None** — phase 2 and phase 3
   results reproduce exactly.
6. **Band flows reconcile**: `Σ_b import_b == grid_import_wh` and
   `Σ_b export_b == export_wh`, for every capacity.
7. **A hand-computable priced case**: a short synthetic series whose euro
   cost can be worked out on paper.
8. **`cost(0)` equals the measured bill** computed directly from the raw
   meter frame without the simulator.
9. **Break-even inverts payback**: at the break-even price, payback equals
   the horizon.
10. **The partial 2026 year is excluded or labelled**, never annualised.

## 12. Predictions on record

Stated before measuring, so that agreement is evidence and disagreement is a
signal to investigate rather than a result to accept:

1. **The euro-optimal capacity will be larger than phase 3's 9.0 kWh.** The
   import/export spread is 18–30×, so each stored kWh is worth far more than
   the energy curve implied. If the euro optimum lands *below* 9.0 kWh, treat
   it as a bug until proven otherwise.
2. **The non-convergent elbow should dissolve.** It was a geometric artifact
   of a curve with no exchange rate. A euro curve against a euro price has a
   real one, so a genuine optimum should exist without reference to where the
   sweep was truncated. If the euro optimum *also* drifts with the sweep top,
   that is a finding worth reporting, not hiding.
3. **The derived threshold will exceed the removed 50 kWh/yr.** A rough
   check — €500/kWh over 10 years at ~0.28 €/kWh blended — suggests ~180
   kWh/yr. If so, the invented threshold was roughly 3× too lenient and was
   recommending too large a battery.

## 13. Stated assumptions

Each of these is a modelling choice a reader could reasonably dispute, so
each is named on the page rather than buried:

- A **2027** tariff is applied to **2020–2026** behaviour. This answers "what
  would this household's pattern cost under the new contract", not "what did
  it cost".
- The household's pattern is assumed unchanged by owning a battery. No
  behavioural response, no load shifting.
- `usable_fraction` and round-trip efficiency carry over from phase 2
  (0.90 / 0.90). Phase 3's final review measured `usable_fraction` as the
  single largest lever on capacity — larger than the ordering bracket, the EV
  rule and the join convention combined — so the euro answer must be
  published with its assumed value visible, and re-run when a quote states a
  real one.
- The 3 kW single-phase inverter limit carries over.
