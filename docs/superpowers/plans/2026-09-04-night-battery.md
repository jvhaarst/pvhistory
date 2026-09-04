# Night Consumption and Battery Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine what size of single-phase home battery is worth installing, by simulating six years of measured 5-minute PV and consumption data against a sweep of capacities.

**Architecture:** Per-night consumption is aggregated against the phase-1 solar-window table and each night classified as EV-charging or not. A chronological 5-minute battery simulation then runs the complete timeline once per inverter power cap, advancing all candidate capacities together as a numpy vector, and accumulating grid import, export, throughput and night-specific metrics per capacity. The recommended capacity is read off the marginal-return curve.

**Tech Stack:** Python ≥3.11, uv, pandas, numpy, matplotlib, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-04-night-battery-design.md` — read it before starting. Its §2.1 "established facts" are measurements, not assumptions, and several tests assert them directly.

**Predecessor:** phase 1's spec `docs/superpowers/specs/2026-09-03-solar-window-design.md` and its modules, which this work consumes unchanged.

## Global Constraints

- **Do not modify any phase-1 module.** `src/pvnight/{config,loader,solar,events,envelope,report}.py` and `analyze.py` are read-only for this work. Import from them freely.
- **UTC internally**, local time only for grouping (`solar_date`) and display.
- **Never assert an exact datetime resolution.** This project resolves to pandas 3.x (microsecond default) where pandas 2.x gave nanoseconds. Assert tz-awareness and UTC, never a literal dtype string, and never add a cast to force a resolution.
- **Consumption nulls are preserved, never zero-filled.** A null `power_cons_w` means unrecorded, not zero. Zero-filling would understate night consumption, the quantity this project measures.
- **No embedded rasters in charts.** An embedded PNG trips the Artifact publish content scanner (established in phase 1). Use `pcolormesh`, never `imshow`.
- **Sample interval** is 5 minutes: `DT_HOURS = 5.0 / 60.0`.
- **Battery defaults:** round-trip 0.90 split symmetrically as `sqrt(0.90)` on each side; usable fraction 0.90; power cap 3.0 kW charge and discharge; capacity sweep 0 to 30 kWh in 0.5 kWh steps; power sensitivity 2.5 / 3.0 / 3.7 kW.
- **EV rule:** a night is EV-charging when ≥ 2.0 hours of its samples exceed 2000 W.
- **Knee rule:** the recommended capacity is the smallest whose marginal return has fallen below 50 kWh/yr per additional kWh.
- Run everything through `uv`. Tests: `uv run pytest`.

## File Structure

| File | Responsibility |
|---|---|
| `src/pvnight/nights.py` | Per-night energy from the window table; EV classification; threshold sensitivity grid |
| `src/pvnight/battery.py` | The simulation and the capacity sweep. Pure functions, no I/O |
| `src/pvnight/night_report.py` | Charts and page. Reuses `report.py`'s style helpers |
| `analyze_night.py` | Entry point; writes the two CSVs and the report |
| `tests/test_nights.py`, `tests/test_battery.py`, `tests/test_night_report.py`, `tests/test_analyze_night.py` | |

---

### Task 1: Per-night aggregation and EV classification

**Files:**
- Create: `src/pvnight/nights.py`
- Test: `tests/test_nights.py`

**Interfaces:**
- Consumes: `pvnight.loader.load(data_dir)` → frame with `ts_utc` (tz-aware UTC), `solar_date` (datetime.date), `power_gen_w`, `power_avg_w`, `energy_gen_wh`, `power_cons_w`, `energy_cons_wh`, `generating`. Note `power_cons_w` and `energy_cons_wh` deliberately contain NaN.
- Produces:
  - `nights.consumption_increments(samples) -> pd.Series` — per-sample Wh, from the within-day cumulative counter.
  - `nights.summarise_nights(samples, windows, ev_power_w=2000.0, ev_hours=2.0, min_coverage=0.95) -> pd.DataFrame` — one row per night with columns `date`, `night_start_utc`, `night_end_utc`, `night_wh`, `peak_w`, `hours_above_2kw`, `is_ev`, `coverage`, `dst_hour_missing`, `covered`.
  - `nights.ev_sensitivity(samples, windows, powers=(1500.0, 2000.0, 2500.0, 3000.0), hours=(1.0, 2.0, 3.0)) -> pd.DataFrame` — columns `power_w`, `hours`, `n_ev`, `median_ev_kwh`, `median_rest_kwh`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nights.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pvnight.nights import consumption_increments, ev_sensitivity, summarise_nights


def _samples(rows):
    """rows: (iso_utc, solar_date, power_cons_w, energy_cons_wh)."""
    return pd.DataFrame(
        [
            {
                "ts_utc": pd.Timestamp(ts, tz="UTC"),
                "solar_date": d,
                "power_cons_w": p,
                "energy_cons_wh": e,
            }
            for ts, d, p, e in rows
        ]
    )


def test_increments_come_from_the_within_day_counter():
    s = _samples([
        ("2023-06-01T00:00", dt.date(2023, 6, 1), 100.0, 0.0),
        ("2023-06-01T00:05", dt.date(2023, 6, 1), 100.0, 30.0),
        ("2023-06-01T00:10", dt.date(2023, 6, 1), 100.0, 55.0),
    ])
    assert consumption_increments(s).tolist() == [0.0, 30.0, 25.0]


def test_increments_do_not_leak_across_the_day_boundary():
    """The counter resets at midnight, so a global diff would go negative."""
    s = _samples([
        ("2023-06-01T23:55", dt.date(2023, 6, 1), 100.0, 5000.0),
        ("2023-06-02T00:00", dt.date(2023, 6, 2), 100.0, 0.0),
        ("2023-06-02T00:05", dt.date(2023, 6, 2), 100.0, 40.0),
    ])
    assert consumption_increments(s).tolist() == [0.0, 0.0, 40.0]


def _one_night_window(start, end):
    return pd.DataFrame([{
        "date": pd.Timestamp(start).date(),
        "night_start_utc": pd.Timestamp(start, tz="UTC"),
        "night_end_utc": pd.Timestamp(end, tz="UTC"),
        "dst_hour_missing": False,
    }])


def test_night_energy_sums_only_increments_inside_the_window():
    s = _samples([
        ("2023-06-01T19:55", dt.date(2023, 6, 1), 100.0, 900.0),   # before
        ("2023-06-01T20:00", dt.date(2023, 6, 1), 100.0, 1000.0),  # in: +100
        ("2023-06-01T20:05", dt.date(2023, 6, 1), 200.0, 1150.0),  # in: +150
        ("2023-06-01T20:10", dt.date(2023, 6, 1), 100.0, 1200.0),  # after end
    ])
    w = _one_night_window("2023-06-01T20:00", "2023-06-01T20:06")
    out = summarise_nights(s, w, min_coverage=0.0)
    assert out.loc[0, "night_wh"] == pytest.approx(250.0)
    assert out.loc[0, "peak_w"] == pytest.approx(200.0)


def test_a_night_is_ev_when_two_hours_exceed_two_kilowatts():
    """25 samples of 5 min = 2h05m above 2 kW, just over the 2h rule."""
    rows = [("2023-06-01T20:00", dt.date(2023, 6, 1), 100.0, 0.0)]
    for k in range(1, 26):
        ts = pd.Timestamp("2023-06-01T20:00", tz="UTC") + pd.Timedelta(minutes=5 * k)
        rows.append((ts.isoformat(), dt.date(2023, 6, 1), 3500.0, 291.7 * k))
    s = _samples(rows)
    w = _one_night_window("2023-06-01T20:00", "2023-06-01T22:10")
    out = summarise_nights(s, w, min_coverage=0.0)
    assert out.loc[0, "hours_above_2kw"] == pytest.approx(2.0833, abs=0.01)
    assert bool(out.loc[0, "is_ev"])


def test_a_short_high_power_burst_is_not_ev():
    """Six samples of 5 min = 30 min above 2 kW — a kettle, not a car."""
    rows = [("2023-06-01T20:00", dt.date(2023, 6, 1), 100.0, 0.0)]
    for k in range(1, 7):
        ts = pd.Timestamp("2023-06-01T20:00", tz="UTC") + pd.Timedelta(minutes=5 * k)
        rows.append((ts.isoformat(), dt.date(2023, 6, 1), 3500.0, 291.7 * k))
    s = _samples(rows)
    w = _one_night_window("2023-06-01T20:00", "2023-06-01T20:35")
    out = summarise_nights(s, w, min_coverage=0.0)
    assert not bool(out.loc[0, "is_ev"])


def test_coverage_flags_nights_with_missing_samples():
    """A 2-hour night should hold 24 samples; supply 3."""
    s = _samples([
        ("2023-06-01T20:00", dt.date(2023, 6, 1), 100.0, 0.0),
        ("2023-06-01T20:05", dt.date(2023, 6, 1), 100.0, 10.0),
        ("2023-06-01T20:10", dt.date(2023, 6, 1), 100.0, 20.0),
    ])
    w = _one_night_window("2023-06-01T20:00", "2023-06-01T22:00")
    out = summarise_nights(s, w, min_coverage=0.95)
    assert out.loc[0, "coverage"] < 0.2
    assert not bool(out.loc[0, "covered"])


@pytest.fixture(scope="session")
def real_nights(loaded):
    w = pd.read_csv(
        "out/solar_windows.csv",
        parse_dates=["date", "night_start_utc", "night_end_utc"],
    )
    return summarise_nights(loaded, w)


def test_real_data_reproduces_the_measured_distribution(real_nights):
    """Spec facts 1 and 10: 2,031 covered nights, median 4.85 kWh, p90 10.52."""
    c = real_nights[real_nights["covered"]]
    c = c[(c["date"] >= "2020-05-20") & (c["date"] <= "2025-12-30")]
    assert len(c) == 2031
    assert c["night_wh"].median() / 1000 == pytest.approx(4.85, abs=0.05)
    assert c["night_wh"].quantile(0.90) / 1000 == pytest.approx(10.52, abs=0.10)


def test_real_data_reproduces_the_measured_ev_split(real_nights):
    """Spec fact 6: 72 EV nights, median 24.5 kWh against 4.71 for the rest."""
    c = real_nights[real_nights["covered"]]
    c = c[(c["date"] >= "2020-05-20") & (c["date"] <= "2025-12-30")]
    ev, rest = c[c["is_ev"]], c[~c["is_ev"]]
    assert len(ev) == 72
    assert ev["night_wh"].median() / 1000 == pytest.approx(24.5, abs=0.3)
    assert rest["night_wh"].median() / 1000 == pytest.approx(4.71, abs=0.05)


def test_the_ev_heuristic_has_a_known_false_positive_rate(real_nights):
    """Spec fact 7 measured "no EV before 2022" with a stricter ">=1h above
    5 kW" rule. The rule is_ev implements also catches 6 pre-2022 winter
    evenings, materially smaller than real EV nights. Bounded and visible,
    not tuned away."""
    c = real_nights[real_nights["covered"]]
    c = c[(c["date"] >= "2020-05-20") & (c["date"] <= "2025-12-30")]
    ev = c[c["is_ev"]]
    pre = ev[pd.to_datetime(ev["date"]).dt.year < 2022]
    assert len(pre) == 6
    assert pre["night_wh"].median() / 1000 == pytest.approx(15.3, abs=0.3)
    post = ev[pd.to_datetime(ev["date"]).dt.year >= 2022]
    assert post["night_wh"].median() > 1.5 * pre["night_wh"].median()


def test_both_ev_charging_modes_are_caught(real_nights):
    """Spec fact 5: fast (~8 kW, 2023-12-28) and slow (~3.5 kW, 2022-11-13).
    A 5 kW threshold would miss the second entirely — its peak is 3732 W."""
    c = real_nights.set_index("date")
    assert bool(c.loc[pd.Timestamp("2023-12-28"), "is_ev"])
    assert bool(c.loc[pd.Timestamp("2022-11-13"), "is_ev"])
    assert c.loc[pd.Timestamp("2022-11-13"), "peak_w"] < 5000


def test_ev_sensitivity_grid_has_a_row_per_combination(loaded):
    w = pd.read_csv(
        "out/solar_windows.csv",
        parse_dates=["date", "night_start_utc", "night_end_utc"],
    )
    g = ev_sensitivity(loaded, w)
    assert len(g) == 12
    assert set(g.columns) == {"power_w", "hours", "n_ev", "median_ev_kwh", "median_rest_kwh"}
    chosen = g[(g.power_w == 2000.0) & (g.hours == 2.0)].iloc[0]
    assert chosen.n_ev == 72
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nights.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.nights'`

- [ ] **Step 3: Implement `nights.py`**

Create `src/pvnight/nights.py`:

```python
"""Per-night consumption, aggregated against the phase-1 solar window."""

from __future__ import annotations

import numpy as np
import pandas as pd

DT_HOURS = 5.0 / 60.0
EV_POWER_W = 2000.0
EV_HOURS = 2.0
MIN_COVERAGE = 0.95


def consumption_increments(samples: pd.DataFrame) -> pd.Series:
    """Wh consumed in each 5-minute sample.

    `energy_cons_wh` is cumulative within a local day and resets at midnight,
    so the diff is grouped by `solar_date` — a global diff would go sharply
    negative at every midnight. Mirrors phase 1's treatment of generation.
    """
    return (
        samples.groupby("solar_date")["energy_cons_wh"].diff().fillna(0.0)
    ).rename("cons_delta_wh")


def summarise_nights(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    ev_power_w: float = EV_POWER_W,
    ev_hours: float = EV_HOURS,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """One row per night, with energy, peak, EV flag and sample coverage."""
    s = samples.sort_values("ts_utc").reset_index(drop=True)
    cum = np.concatenate([[0.0], consumption_increments(s).to_numpy().cumsum()])
    ts = s["ts_utc"].to_numpy()
    power = s["power_cons_w"].to_numpy()

    w = windows.dropna(subset=["night_start_utc", "night_end_utc"]).reset_index(drop=True)
    lo = np.searchsorted(ts, w["night_start_utc"].to_numpy(), side="left")
    hi = np.searchsorted(ts, w["night_end_utc"].to_numpy(), side="right")

    duration_h = (
        w["night_end_utc"] - w["night_start_utc"]
    ).dt.total_seconds().to_numpy() / 3600.0
    expected = duration_h / DT_HOURS

    peak = np.zeros(len(w))
    hours_hi = np.zeros(len(w))
    present = np.zeros(len(w))
    for k, (a, b) in enumerate(zip(lo, hi)):
        seg = power[a:b]
        seg = seg[~np.isnan(seg)]
        present[k] = len(seg)
        if len(seg):
            peak[k] = seg.max()
            hours_hi[k] = (seg > ev_power_w).sum() * DT_HOURS

    out = pd.DataFrame({
        "date": w["date"],
        "night_start_utc": w["night_start_utc"],
        "night_end_utc": w["night_end_utc"],
        "night_wh": cum[hi] - cum[lo],
        "peak_w": peak,
        "hours_above_2kw": hours_hi,
        "coverage": np.where(expected > 0, present / expected, 0.0),
        "dst_hour_missing": w["dst_hour_missing"] if "dst_hour_missing" in w else False,
    })
    out["is_ev"] = out["hours_above_2kw"] >= ev_hours
    out["covered"] = out["coverage"] >= min_coverage
    return out


def ev_sensitivity(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    powers: tuple[float, ...] = (1500.0, 2000.0, 2500.0, 3000.0),
    hours: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """How the EV/non-EV split moves as the threshold moves.

    Published in the report because the classification is a heuristic: a split
    that swings wildly across this grid is an artefact of where the line was
    drawn, and the reader deserves to see that.
    """
    rows = []
    for p in powers:
        base = summarise_nights(samples, windows, ev_power_w=p, ev_hours=min(hours))
        base = base[base["covered"]]
        for h in hours:
            # NOTE: the column is named for the default 2 kW threshold, but it
            # holds hours above `p` on this pass. Do not read it as literally
            # 2 kW here.
            is_ev = base["hours_above_2kw"] >= h
            rows.append({
                "power_w": p,
                "hours": h,
                "n_ev": int(is_ev.sum()),
                "median_ev_kwh": base.loc[is_ev, "night_wh"].median() / 1000,
                "median_rest_kwh": base.loc[~is_ev, "night_wh"].median() / 1000,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nights.py -v`
Expected: PASS, 11 tests. The real-data tests load six years, so allow ~30 s.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/nights.py tests/test_nights.py
git commit -m "Add per-night consumption aggregation and EV classification

Classifies on hours above 2 kW rather than peak power, because the EV
charges in two modes and the slow one never exceeds 3.7 kW."
```

---

### Task 2: The battery simulator

**Files:**
- Create: `src/pvnight/battery.py`
- Test: `tests/test_battery.py`

**Interfaces:**
- Consumes: nothing from Task 1; this module is pure numpy.
- Produces:
  - `battery.BatterySpec` — frozen dataclass with `capacities_kwh: np.ndarray`, `power_kw: float`, `round_trip: float = 0.90`, `usable_fraction: float = 0.90`; properties `usable_wh -> np.ndarray` and `eta -> float`.
  - `battery.SimResult` — dataclass with `capacities_kwh`, `grid_import_wh`, `export_wh`, `charge_wh`, `discharge_wh`, `night_grid_import_wh`, `nonev_grid_import_wh`, `monthly_discharge_wh` (shape `(12, n_caps)`), `final_stored_wh`, all numpy arrays.
  - `battery.simulate(net_wh, night_mask, nonev_mask, month_idx, spec) -> SimResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_battery.py`:

```python
import numpy as np
import pytest

from pvnight.battery import BatterySpec, simulate


def _flat(n):
    """Helper masks for a run with no night/month structure."""
    return np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, int)


def test_zero_capacity_stores_nothing_and_imports_everything():
    net = np.array([500.0, -500.0, -200.0])
    nm, em, mi = _flat(3)
    r = simulate(net, nm, em, mi, BatterySpec(np.array([0.0]), power_kw=3.0))
    assert r.charge_wh[0] == pytest.approx(0.0)
    assert r.discharge_wh[0] == pytest.approx(0.0)
    assert r.grid_import_wh[0] == pytest.approx(700.0)
    assert r.export_wh[0] == pytest.approx(500.0)


def test_round_trip_losses_are_applied():
    """Store 1000 Wh at the AC side, then draw it all back: 90% returns."""
    net = np.array([1000.0] + [-100.0] * 20)
    nm, em, mi = _flat(21)
    spec = BatterySpec(np.array([10.0]), power_kw=100.0, round_trip=0.90)
    r = simulate(net, nm, em, mi, spec)
    assert r.charge_wh[0] == pytest.approx(1000.0)
    assert r.discharge_wh[0] == pytest.approx(900.0, abs=1e-6)


def test_usable_fraction_caps_what_the_battery_holds():
    """A 10 kWh nameplate at 90% usable holds 9 kWh at the terminal."""
    net = np.array([50_000.0])
    nm, em, mi = _flat(1)
    spec = BatterySpec(np.array([10.0]), power_kw=1e6, usable_fraction=0.90)
    r = simulate(net, nm, em, mi, spec)
    assert r.final_stored_wh[0] == pytest.approx(9000.0)


def test_power_cap_binds_against_an_ev_draw():
    """8 kW demand for one 5-minute step against a 3 kW inverter: the battery
    supplies 3 kW worth and the grid covers the other 5 kW."""
    dt = 5.0 / 60.0
    net = np.array([20_000.0, -8000.0 * dt])
    nm, em, mi = _flat(2)
    spec = BatterySpec(np.array([20.0]), power_kw=3.0, round_trip=1.0)
    r = simulate(net, nm, em, mi, spec)
    assert r.discharge_wh[0] == pytest.approx(3000.0 * dt)
    assert r.grid_import_wh[0] == pytest.approx(5000.0 * dt)


def test_energy_is_conserved():
    """G + I = L + X + A - D, the invariant that catches most dispatch bugs."""
    rng = np.random.default_rng(0)
    n = 5000
    gen = np.clip(rng.normal(300, 400, n), 0, None) * (5 / 60)
    load = np.clip(rng.normal(350, 250, n), 0, None) * (5 / 60)
    net = gen - load
    nm, em, mi = _flat(n)
    spec = BatterySpec(np.array([0.0, 5.0, 10.0, 20.0]), power_kw=3.0)
    r = simulate(net, nm, em, mi, spec)
    lhs = gen.sum() + r.grid_import_wh
    rhs = load.sum() + r.export_wh + r.charge_wh - r.discharge_wh
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_stored_energy_equals_charge_in_minus_discharge_out():
    rng = np.random.default_rng(1)
    net = rng.normal(0, 300, 4000)
    nm, em, mi = _flat(4000)
    spec = BatterySpec(np.array([5.0, 15.0]), power_kw=3.0)
    r = simulate(net, nm, em, mi, spec)
    eta = spec.eta
    assert np.allclose(r.final_stored_wh, r.charge_wh * eta - r.discharge_wh / eta, atol=1e-6)


def test_more_capacity_never_increases_grid_import():
    """Monotonicity. A violation means the dispatch has a state bug."""
    rng = np.random.default_rng(2)
    gen = np.clip(rng.normal(400, 500, 8000), 0, None) * (5 / 60)
    load = np.clip(rng.normal(300, 200, 8000), 0, None) * (5 / 60)
    nm, em, mi = _flat(8000)
    spec = BatterySpec(np.arange(0.0, 20.1, 1.0), power_kw=3.0)
    r = simulate(gen - load, nm, em, mi, spec)
    assert np.all(np.diff(r.grid_import_wh) <= 1e-9)
    assert np.all(np.diff(r.export_wh) <= 1e-9)


def test_night_and_nonev_import_are_accumulated_separately():
    net = np.array([-100.0, -100.0, -100.0])
    night = np.array([True, True, False])
    nonev = np.array([True, False, False])
    mi = np.zeros(3, int)
    r = simulate(net, night, nonev, mi, BatterySpec(np.array([0.0]), power_kw=3.0))
    assert r.grid_import_wh[0] == pytest.approx(300.0)
    assert r.night_grid_import_wh[0] == pytest.approx(200.0)
    assert r.nonev_grid_import_wh[0] == pytest.approx(100.0)


def test_monthly_discharge_lands_in_the_right_month():
    net = np.array([5000.0, -1000.0, -1000.0])
    nm, em = np.zeros(3, bool), np.zeros(3, bool)
    mi = np.array([0, 0, 6])
    spec = BatterySpec(np.array([10.0]), power_kw=1e6, round_trip=1.0)
    r = simulate(net, nm, em, mi, spec)
    assert r.monthly_discharge_wh[0, 0] == pytest.approx(1000.0)
    assert r.monthly_discharge_wh[6, 0] == pytest.approx(1000.0)
    assert r.monthly_discharge_wh[3, 0] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_battery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.battery'`

- [ ] **Step 3: Implement `battery.py`**

Create `src/pvnight/battery.py`:

```python
"""Chronological battery simulation, vectorised across candidate capacities.

The simulation is inherently sequential in time — each step's state depends on
the last — but every candidate capacity can advance together as a numpy vector,
so the whole sweep is one pass over the data rather than one pass per capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT_HOURS = 5.0 / 60.0


@dataclass(frozen=True)
class BatterySpec:
    """Candidate capacities and the inverter they sit behind.

    Capacities are nameplate — the number a product is sold under. Usable
    energy at the terminal is `capacity * usable_fraction`, and the round trip
    is split symmetrically so charging and discharging each cost sqrt(eta).
    """

    capacities_kwh: np.ndarray
    power_kw: float
    round_trip: float = 0.90
    usable_fraction: float = 0.90

    @property
    def usable_wh(self) -> np.ndarray:
        return np.asarray(self.capacities_kwh, dtype=float) * 1000.0 * self.usable_fraction

    @property
    def eta(self) -> float:
        """One-way efficiency: sqrt of the round trip, applied on each side."""
        return float(np.sqrt(self.round_trip))


@dataclass
class SimResult:
    capacities_kwh: np.ndarray
    grid_import_wh: np.ndarray
    export_wh: np.ndarray
    charge_wh: np.ndarray
    discharge_wh: np.ndarray
    night_grid_import_wh: np.ndarray
    nonev_grid_import_wh: np.ndarray
    monthly_discharge_wh: np.ndarray
    final_stored_wh: np.ndarray


def simulate(
    net_wh: np.ndarray,
    night_mask: np.ndarray,
    nonev_mask: np.ndarray,
    month_idx: np.ndarray,
    spec: BatterySpec,
) -> SimResult:
    """Run the whole timeline once for every capacity in `spec`.

    `net_wh` is generation minus load per interval. Positive charges the
    battery and exports the rest; negative discharges and imports the rest.
    `night_mask` and `nonev_mask` select which intervals contribute to the
    night-specific accumulators — the timeline itself is never broken up, so
    the battery's state always reflects every night that actually happened.
    """
    caps = spec.usable_wh
    n = len(caps)
    eta = spec.eta
    limit_wh = spec.power_kw * 1000.0 * DT_HOURS

    stored = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    grid = np.zeros(n)
    export = np.zeros(n)
    night_grid = np.zeros(n)
    nonev_grid = np.zeros(n)
    monthly = np.zeros((12, n))

    for t in range(len(net_wh)):
        e = float(net_wh[t])
        if e > 0.0:
            accept = np.minimum(np.minimum(e, limit_wh), (caps - stored) / eta)
            stored += accept * eta
            charge += accept
            export += e - accept
        elif e < 0.0:
            need = -e
            deliver = np.minimum(np.minimum(need, limit_wh), stored * eta)
            stored -= deliver / eta
            discharge += deliver
            shortfall = need - deliver
            grid += shortfall
            if night_mask[t]:
                night_grid += shortfall
            if nonev_mask[t]:
                nonev_grid += shortfall
            monthly[month_idx[t]] += deliver

    return SimResult(
        capacities_kwh=np.asarray(spec.capacities_kwh, dtype=float),
        grid_import_wh=grid,
        export_wh=export,
        charge_wh=charge,
        discharge_wh=discharge,
        night_grid_import_wh=night_grid,
        nonev_grid_import_wh=nonev_grid,
        monthly_discharge_wh=monthly,
        final_stored_wh=stored,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_battery.py -v`
Expected: PASS, 9 tests, fast — all synthetic.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/battery.py tests/test_battery.py
git commit -m "Add vectorised battery simulator

Advances every candidate capacity together through one pass of the
timeline. Guarded by an energy-conservation invariant and a monotonicity
check that a state bug would break."
```

---

### Task 3: The capacity sweep and the knee

**Files:**
- Modify: `src/pvnight/battery.py` (append)
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `battery.BatterySpec`, `battery.simulate` from Task 2; `nights.summarise_nights` from Task 1.
- Produces:
  - `battery.build_masks(samples, nights_df) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(night_mask, nonev_mask, month_idx)`, each aligned to `samples` sorted by `ts_utc`.
  - `battery.net_wh(samples) -> np.ndarray` — generation minus consumption per interval, in Wh, with NaN consumption treated as a gap (see below).
  - `battery.sweep(samples, nights_df, capacities_kwh, power_kws=(2.5, 3.0, 3.7), round_trip=0.90, usable_fraction=0.90) -> pd.DataFrame` — one row per (capacity, power), columns `capacity_kwh`, `power_kw`, `grid_import_kwh_yr`, `pv_export_kwh_yr`, `night_grid_import_kwh_yr`, `nonev_night_grid_import_kwh_yr`, `night_self_sufficiency_pct`, `nonev_night_self_sufficiency_pct`, `cycles_per_yr`, `marginal_kwh_per_kwh`.
  - `battery.recommend_capacity(sweep_df, power_kw=3.0, threshold_kwh_per_kwh=50.0, scenario="nonev") -> float`

**On NaN consumption:** 1,315 samples have no consumption reading. Treat those intervals as `net_wh = 0` — the battery neither charges nor discharges across an interval we cannot see — and record the count. Do NOT zero-fill the load, which would invent free energy and understate grid import.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sweep.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pvnight.battery import build_masks, net_wh, recommend_capacity, sweep


def test_net_treats_missing_consumption_as_a_gap():
    s = pd.DataFrame({
        "ts_utc": pd.to_datetime(
            ["2023-06-01T00:00", "2023-06-01T00:05"], utc=True
        ),
        "power_gen_w": [0.0, 1200.0],
        "power_cons_w": [np.nan, 200.0],
    })
    out = net_wh(s)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1000.0 * 5 / 60)


def test_masks_mark_the_right_intervals():
    s = pd.DataFrame({
        "ts_utc": pd.to_datetime(
            ["2023-06-01T21:00", "2023-06-01T22:00", "2023-07-02T21:00"], utc=True
        ),
        "power_gen_w": [0.0, 0.0, 0.0],
        "power_cons_w": [100.0, 100.0, 100.0],
    })
    n = pd.DataFrame([
        {"date": pd.Timestamp("2023-06-01"),
         "night_start_utc": pd.Timestamp("2023-06-01T20:00", tz="UTC"),
         "night_end_utc": pd.Timestamp("2023-06-01T23:00", tz="UTC"),
         "is_ev": True, "covered": True},
        {"date": pd.Timestamp("2023-07-02"),
         "night_start_utc": pd.Timestamp("2023-07-02T20:00", tz="UTC"),
         "night_end_utc": pd.Timestamp("2023-07-02T23:00", tz="UTC"),
         "is_ev": False, "covered": True},
    ])
    night, nonev, month = build_masks(s, n)
    assert night.tolist() == [True, True, True]
    assert nonev.tolist() == [False, False, True]   # first night is EV
    assert month.tolist() == [5, 5, 6]              # 0-based month index


def _toy_inputs():
    idx = pd.date_range("2023-06-01", periods=2000, freq="5min", tz="UTC")
    gen = np.where((idx.hour > 8) & (idx.hour < 17), 2500.0, 0.0)
    s = pd.DataFrame({"ts_utc": idx, "power_gen_w": gen, "power_cons_w": 400.0})
    n = pd.DataFrame([{
        "date": idx[0].date(),
        "night_start_utc": idx[0], "night_end_utc": idx[-1],
        "is_ev": False, "covered": True,
    }])
    return s, n


def test_sweep_returns_a_row_per_capacity_and_power():
    s, n = _toy_inputs()
    out = sweep(s, n, capacities_kwh=np.array([0.0, 5.0, 10.0]), power_kws=(3.0, 3.7))
    assert len(out) == 6
    assert set(out.power_kw.unique()) == {3.0, 3.7}


def test_sweep_grid_import_falls_with_capacity():
    s, n = _toy_inputs()
    out = sweep(s, n, capacities_kwh=np.arange(0.0, 11.0, 1.0), power_kws=(3.0,))
    g = out.sort_values("capacity_kwh").grid_import_kwh_yr.to_numpy()
    assert np.all(np.diff(g) <= 1e-9)
    assert g[0] > g[-1]


def test_marginal_return_is_the_gradient_of_avoided_import():
    s, n = _toy_inputs()
    out = sweep(s, n, capacities_kwh=np.array([0.0, 1.0, 2.0]), power_kws=(3.0,)).sort_values("capacity_kwh")
    g = out.grid_import_kwh_yr.to_numpy()
    m = out.marginal_kwh_per_kwh.to_numpy()
    assert m[1] == pytest.approx(g[0] - g[1], rel=1e-6)


def test_recommend_returns_the_first_capacity_below_the_threshold():
    df = pd.DataFrame({
        "capacity_kwh": [0.0, 1.0, 2.0, 3.0, 4.0],
        "power_kw": 3.0,
        "marginal_kwh_per_kwh": [np.nan, 900.0, 800.0, 700.0, 600.0],
        "nonev_marginal_kwh_per_kwh": [np.nan, 200.0, 120.0, 40.0, 10.0],
    })
    # default scenario is "nonev", so the nonev column decides
    assert recommend_capacity(df, power_kw=3.0, threshold_kwh_per_kwh=50.0) == 3.0
    # and "all" reads the other column, which never drops below the threshold
    assert recommend_capacity(df, power_kw=3.0, threshold_kwh_per_kwh=50.0,
                              scenario="all") == 4.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_masks'`

- [ ] **Step 3: Append the sweep to `battery.py`**

Add to `src/pvnight/battery.py`. Add `import pandas as pd` to the imports at the top of the file.

```python
def net_wh(samples: pd.DataFrame) -> np.ndarray:
    """Generation minus load per interval, in Wh.

    Intervals with no consumption reading become zero rather than
    generation-only: we cannot see that load, and pretending it was zero
    would invent free energy and understate grid import.
    """
    s = samples.sort_values("ts_utc")
    gen = s["power_gen_w"].to_numpy(dtype=float)
    load = s["power_cons_w"].to_numpy(dtype=float)
    net = (gen - load) * DT_HOURS
    return np.where(np.isnan(load), 0.0, net)


def build_masks(
    samples: pd.DataFrame, nights_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-interval flags: inside any night, inside a non-EV night, and month."""
    s = samples.sort_values("ts_utc")
    ts = s["ts_utc"].to_numpy()
    night = np.zeros(len(s), dtype=bool)
    nonev = np.zeros(len(s), dtype=bool)

    n = nights_df.dropna(subset=["night_start_utc", "night_end_utc"])
    lo = np.searchsorted(ts, n["night_start_utc"].to_numpy(), side="left")
    hi = np.searchsorted(ts, n["night_end_utc"].to_numpy(), side="right")
    for a, b, is_ev, covered in zip(lo, hi, n["is_ev"], n["covered"]):
        if not covered:
            continue
        night[a:b] = True
        if not is_ev:
            nonev[a:b] = True

    month_idx = pd.DatetimeIndex(s["ts_utc"]).month.to_numpy() - 1
    return night, nonev, month_idx


def sweep(
    samples: pd.DataFrame,
    nights_df: pd.DataFrame,
    capacities_kwh: np.ndarray,
    power_kws: tuple[float, ...] = (2.5, 3.0, 3.7),
    round_trip: float = 0.90,
    usable_fraction: float = 0.90,
) -> pd.DataFrame:
    """Simulate every capacity against every inverter power, once each."""
    s = samples.sort_values("ts_utc")
    net = net_wh(s)
    night, nonev, month = build_masks(s, nights_df)
    span = (s["ts_utc"].iloc[-1] - s["ts_utc"].iloc[0]).total_seconds()
    years = span / (365.25 * 24 * 3600)

    night_load = -net[night & (net < 0)].sum()
    nonev_load = -net[nonev & (net < 0)].sum()

    frames = []
    for p in power_kws:
        spec = BatterySpec(np.asarray(capacities_kwh, dtype=float), p, round_trip, usable_fraction)
        r = simulate(net, night, nonev, month, spec)
        usable = np.where(spec.usable_wh > 0, spec.usable_wh, np.nan)
        df = pd.DataFrame({
            "capacity_kwh": r.capacities_kwh,
            "power_kw": p,
            "grid_import_kwh_yr": r.grid_import_wh / 1000 / years,
            "pv_export_kwh_yr": r.export_wh / 1000 / years,
            "night_grid_import_kwh_yr": r.night_grid_import_wh / 1000 / years,
            "nonev_night_grid_import_kwh_yr": r.nonev_grid_import_wh / 1000 / years,
            "night_self_sufficiency_pct": 100 * (1 - r.night_grid_import_wh / night_load),
            "nonev_night_self_sufficiency_pct": 100 * (1 - r.nonev_grid_import_wh / nonev_load),
            "cycles_per_yr": r.discharge_wh / usable / years,
        }).sort_values("capacity_kwh").reset_index(drop=True)
        step = df["capacity_kwh"].diff()
        df["marginal_kwh_per_kwh"] = -df["grid_import_kwh_yr"].diff() / step
        df["nonev_marginal_kwh_per_kwh"] = -df["nonev_night_grid_import_kwh_yr"].diff() / step
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def recommend_capacity(
    sweep_df: pd.DataFrame,
    power_kw: float = 3.0,
    threshold_kwh_per_kwh: float = 50.0,
    scenario: str = "nonev",
) -> float:
    """Smallest capacity whose marginal return has fallen below the threshold.

    "Where the curve flattens" is not implementable, so the rule is explicit:
    an extra kWh of battery earning less than `threshold_kwh_per_kwh` per year
    is cycling under about once a week, which is hard to justify buying. The
    threshold is a stated judgement, not a derived constant — the report prints
    the whole curve so a reader can choose differently.
    """
    col = "marginal_kwh_per_kwh" if scenario == "all" else "nonev_marginal_kwh_per_kwh"
    d = sweep_df[sweep_df["power_kw"] == power_kw].sort_values("capacity_kwh")
    below = d[d[col].notna() & (d[col] < threshold_kwh_per_kwh)]
    if below.empty:
        return float(d["capacity_kwh"].max())
    return float(below["capacity_kwh"].iloc[0])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS. 62 from phase 1 plus 11 + 9 + 6 = 88.

- [ ] **Step 6: Commit**

```bash
git add src/pvnight/battery.py tests/test_sweep.py
git commit -m "Add capacity sweep, per-year metrics and the knee rule

The knee rule is explicit rather than eyeballed: the smallest capacity
whose marginal return falls below 50 kWh/yr per added kWh."
```

---

### Task 4: Charts and report page

**Files:**
- Create: `src/pvnight/night_report.py`
- Test: `tests/test_night_report.py`

**Interfaces:**
- Consumes: `nights.summarise_nights`, `nights.ev_sensitivity`, `battery.sweep`, `battery.recommend_capacity`. Reuses `pvnight.report`'s `_svg`, `SERIES`, `FURNITURE`, `COVERAGE_CMAP` and `STYLE` — import them, do not redefine.
- Produces: `night_report.build_html(nights_df, sweep_df, monthly_df, sensitivity_df, recommended_kwh) -> str` — a complete HTML fragment with a `<title>`, a `<style>` block and eight inline SVG charts.

**Before writing chart code, load the `dataviz` skill, then `artifact-design`.** Phase 1's palette in `report.py` was validated against the skill's `palette.md`; reuse it rather than picking new colours.

**`monthly_df`** is built by `analyze_night.py` in Task 5 with columns `month` (1-12), `gen_kwh`, `day_cons_kwh`, `surplus_kwh`, `night_kwh`, `above_cap_pct_ev`, `above_cap_pct_nonev`, `discharge_kwh` — the per-month medians the charts need.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_night_report.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pvnight.night_report import build_html


@pytest.fixture
def toy():
    nights = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=60, freq="D"),
        "night_wh": np.linspace(1000, 30000, 60),
        "peak_w": np.linspace(300, 9000, 60),
        "hours_above_2kw": np.linspace(0, 4, 60),
        "is_ev": [False] * 55 + [True] * 5,
        "covered": True,
        "coverage": 1.0,
    })
    caps = np.arange(0.0, 20.1, 1.0)
    sweep = pd.DataFrame({
        "capacity_kwh": np.tile(caps, 2),
        "power_kw": np.repeat([3.0, 3.7], len(caps)),
        "grid_import_kwh_yr": np.tile(3000 * np.exp(-caps / 6), 2),
        "pv_export_kwh_yr": np.tile(2000 * np.exp(-caps / 8), 2),
        "night_grid_import_kwh_yr": np.tile(1500 * np.exp(-caps / 5), 2),
        "nonev_night_grid_import_kwh_yr": np.tile(1200 * np.exp(-caps / 5), 2),
        "night_self_sufficiency_pct": np.tile(100 * (1 - np.exp(-caps / 5)), 2),
        "nonev_night_self_sufficiency_pct": np.tile(100 * (1 - np.exp(-caps / 5)), 2),
        "cycles_per_yr": np.tile(np.r_[0, 200 / caps[1:]], 2),
        "marginal_kwh_per_kwh": np.tile(np.r_[np.nan, np.diff(3000 * np.exp(-caps / 6)) * -1], 2),
        "nonev_marginal_kwh_per_kwh": np.tile(np.r_[np.nan, np.diff(1200 * np.exp(-caps / 5)) * -1], 2),
    })
    monthly = pd.DataFrame({
        "month": range(1, 13),
        "gen_kwh": [2.2, 4.6, 13.5, 25.2, 28.3, 32.7, 27.3, 24.9, 15.5, 5.7, 2.9, 1.7],
        "day_cons_kwh": [5.9, 5.8, 6.0, 7.5, 7.2, 8.0, 7.0, 6.6, 6.5, 5.9, 4.8, 5.4],
        "surplus_kwh": [-3.4, -0.3, 6.8, 15.9, 21.3, 23.9, 19.4, 17.4, 9.3, 0.0, -2.2, -3.5],
        "night_kwh": [9.1, 6.3, 5.5, 3.9, 2.8, 2.8, 2.8, 3.9, 5.0, 5.9, 8.3, 9.3],
        "above_cap_pct_ev": [30, 28, 25, 20, 18, 15, 16, 19, 22, 27, 31, 33],
        "above_cap_pct_nonev": [1.6, 1.5, 1.4, 1.2, 1.0, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.8],
        "discharge_kwh": [5, 20, 90, 160, 190, 200, 185, 170, 120, 30, 6, 3],
    })
    sens = pd.DataFrame({
        "power_w": np.repeat([1500.0, 2000.0, 2500.0, 3000.0], 3),
        "hours": np.tile([1.0, 2.0, 3.0], 4),
        "n_ev": [210, 80, 50, 200, 72, 42, 150, 65, 38, 120, 55, 30],
        "median_ev_kwh": np.linspace(12, 30, 12),
        "median_rest_kwh": np.linspace(4.4, 4.9, 12),
    })
    return nights, sweep, monthly, sens


def test_page_has_a_title_and_eight_charts(toy):
    html = build_html(*toy, recommended_kwh=7.0)
    assert "<title>" in html
    assert html.count("<svg") == 8


def test_page_omits_the_document_wrapper(toy):
    html = build_html(*toy, recommended_kwh=7.0)
    for tag in ["<!doctype", "<html", "<head>", "<body>"]:
        assert tag not in html.lower()


def test_page_embeds_no_raster_images(toy):
    """An embedded PNG trips the Artifact publish content scanner."""
    html = build_html(*toy, recommended_kwh=7.0)
    assert "<image" not in html
    assert "data:image" not in html


def test_page_defines_light_and_dark_palettes(toy):
    html = build_html(*toy, recommended_kwh=7.0)
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="dark"]' in html
    assert '[data-theme="light"]' in html


def test_page_states_the_winter_finding(toy):
    """The single most important caveat: no battery charges in midwinter."""
    html = build_html(*toy, recommended_kwh=7.0).lower()
    assert "winter" in html
    assert "negative" in html or "cannot" in html


def test_page_labels_the_ev_rule_as_a_heuristic(toy):
    html = build_html(*toy, recommended_kwh=7.0).lower()
    assert "heuristic" in html
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_night_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.night_report'`

- [ ] **Step 3: Implement `night_report.py`**

Create `src/pvnight/night_report.py`. Reuse phase 1's helpers:

```python
"""Charts and the battery-sizing report page."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .report import COVERAGE_CMAP, FURNITURE, SERIES, STYLE, _svg  # noqa: E402

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
```

Then the eight charts, each returning `_svg(fig)`:

```python
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
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("night consumption (kWh)")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)


def chart_surplus_vs_need(monthly: pd.DataFrame) -> str:
    """The winter wall: months where surplus is negative can charge nothing."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(12)
    ax.bar(x - 0.2, monthly["surplus_kwh"], 0.4, color=SERIES[2], label="daytime surplus")
    ax.bar(x + 0.2, monthly["night_kwh"], 0.4, color=SERIES[1], label="night need")
    ax.axhline(0, color=FURNITURE, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(MONTHS)
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
    ax.annotate(f"{recommended_kwh:.1f} kWh", xy=(recommended_kwh, ax.get_ylim()[1]*0.9),
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
    ax.bar(np.arange(12), monthly["discharge_kwh"], color=SERIES[0])
    ax.set_xticks(np.arange(12)); ax.set_xticklabels(MONTHS)
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
    x = np.arange(12)
    ax.bar(x - 0.2, monthly["above_cap_pct_ev"], 0.4, color=SERIES[1], label="EV nights")
    ax.bar(x + 0.2, monthly["above_cap_pct_nonev"], 0.4, color=SERIES[0], label="household nights")
    ax.set_xticks(x); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("% of night energy drawn above 3 kW")
    ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)
    return _svg(fig)
```

Then assemble the page:

```python
def _sensitivity_table(sens: pd.DataFrame) -> str:
    head = "".join(f"<th>{h}</th>" for h in
                   ["power (W)", "hours", "EV nights", "median EV kWh", "median other kWh"])
    body = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in [
            f"{r.power_w:.0f}", f"{r.hours:.0f}", f"{r.n_ev:.0f}",
            f"{r.median_ev_kwh:.1f}", f"{r.median_rest_kwh:.2f}"]) + "</tr>"
        for r in sens.itertuples())
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html(nights_df: pd.DataFrame, sweep_df: pd.DataFrame,
               monthly_df: pd.DataFrame, sensitivity_df: pd.DataFrame,
               recommended_kwh: float) -> str:
    """Assemble the report. No document wrapper — the host supplies it."""
    covered = nights_df[nights_df["covered"]]
    at = sweep_df[(sweep_df.power_kw == 3.0)
                  & np.isclose(sweep_df.capacity_kwh, recommended_kwh)]
    ss = float(at["nonev_night_self_sufficiency_pct"].iloc[0]) if len(at) else float("nan")
    cyc = float(at["cycles_per_yr"].iloc[0]) if len(at) else float("nan")

    stats = [
        (f"{recommended_kwh:.1f} kWh", "recommended capacity, nameplate"),
        (f"{covered['night_wh'].median()/1000:.2f} kWh", "median night consumption"),
        (f"{int(covered['is_ev'].sum())}", "EV-charging nights of "
                                           f"{len(covered)}"),
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
         "Median daytime surplus against median night need, per month. In "
         "November, December and January the surplus is <strong>negative</strong> "
         "— the house does not generate enough to cover even its daytime load, "
         "so no battery of any size receives a charge. Across all six years "
         "only 48.2% of nights could be covered even with infinite storage. "
         "This, not capacity, is what bounds the answer.",
         chart_surplus_vs_need(monthly_df)),
        ("How much capacity is worth buying",
         "Night grid import against capacity, with and without EV nights. "
         "The curve flattens because winter contributes nothing no matter "
         "how large the battery is.",
         chart_knee(sweep_df, recommended_kwh)),
        ("Where the marginal kWh stops paying",
         f"Each additional kWh of capacity avoids less than the last. The "
         f"recommendation of {recommended_kwh:.1f} kWh is the first capacity "
         "whose marginal return falls below 50 kWh/yr per added kWh — an "
         "extra kWh cycling less than once a week. <strong>That 50 is a "
         "stated judgement, not a derived constant</strong>: read your own "
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
        "hours. The table below shows how the split moves as the threshold "
        "moves, so you can judge whether it is stable or an artefact of "
        "where the line was drawn.</p>"
        f"{_sensitivity_table(sensitivity_df)}</section>")

    return (
        "<title>Night Consumption and Battery Sizing</title>"
        + STYLE
        + "<main><h1>What size battery is worth buying</h1>"
        + '<p class="lede">Six years of measured five-minute data, simulated '
        + "against a sweep of capacities behind a single-phase 3 kW "
        + "inverter.</p>"
        + f'<div class="stats">{stat_html}</div>'
        + card_html + ev_note + "</main>")
```

Add a `table` rule to the page CSS by appending a small `<style>` block after the
imported `STYLE`, so the sensitivity table is legible:

```python
TABLE_CSS = """
<style>
table { border-collapse: collapse; font-size: 13px; width: 100%; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 500; }
</style>
"""
```

Include `TABLE_CSS` immediately after `STYLE` in the returned string.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_night_report.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/night_report.py tests/test_night_report.py
git commit -m "Add battery-sizing charts and report page"
```

---

### Task 5: Entry point and outputs

**Files:**
- Create: `analyze_night.py`, modify `README.md`
- Test: `tests/test_analyze_night.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `analyze_night.run(data_dir, out_dir) -> dict` writing `out/night_summary.csv`, `out/battery_sweep.csv`, `out/night_report.html`, returning a summary dict with keys `n_nights`, `n_ev_nights`, `median_night_kwh`, `p90_night_kwh`, `recommended_kwh`, `night_self_sufficiency_pct`, `cycles_per_yr`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_night.py`:

```python
import pandas as pd
import pytest

from analyze_night import run


def test_run_writes_all_three_outputs(tmp_path, data_dir):
    s = run(data_dir, tmp_path)

    summary = pd.read_csv(tmp_path / "night_summary.csv")
    sweep = pd.read_csv(tmp_path / "battery_sweep.csv")
    assert len(summary) > 2000
    assert set(sweep.power_kw.unique()) == {2.5, 3.0, 3.7}
    assert (tmp_path / "night_report.html").read_text().count("<svg") == 8

    assert s["n_ev_nights"] == 72
    assert s["median_night_kwh"] == pytest.approx(4.85, abs=0.05)
    assert 0 < s["recommended_kwh"] <= 30


def test_power_cap_sensitivity_is_measured_not_assumed(tmp_path, data_dir):
    """Spec fact 9 says the 3 kW charge cap is *expected* not to bind, and
    that the simulation must confirm it rather than the spec asserting it.
    This pins that the figure is actually computed and is a sane percentage;
    the report states the value, whatever it turns out to be."""
    s = run(data_dir, tmp_path)
    gain = s["pct_gain_from_3p7kw_inverter"]
    assert gain == gain          # not NaN
    assert 0.0 <= gain < 100.0


def test_summary_csv_carries_the_ev_flag_and_coverage(tmp_path, data_dir):
    run(data_dir, tmp_path)
    d = pd.read_csv(tmp_path / "night_summary.csv")
    for col in ["date", "night_wh", "peak_w", "hours_above_2kw", "is_ev", "coverage"]:
        assert col in d.columns
    assert d["is_ev"].sum() == 72
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_analyze_night.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analyze_night'`

- [ ] **Step 3: Implement `analyze_night.py`**

Create `analyze_night.py` at the repository root:

```python
"""Night consumption and battery sizing. Run: uv run python analyze_night.py"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pvnight import battery, night_report, nights
from pvnight.loader import load

CAPACITIES = np.arange(0.0, 30.01, 0.5)
POWER_KWS = (2.5, 3.0, 3.7)
CAP_W = 3000.0


def _monthly(samples: pd.DataFrame, nights_df: pd.DataFrame,
             windows: pd.DataFrame, recommended_kwh: float) -> pd.DataFrame:
    """Per-month medians the charts need."""
    s = samples.sort_values("ts_utc").reset_index(drop=True)
    delta = nights.consumption_increments(s)
    cum = np.concatenate([[0.0], delta.to_numpy().cumsum()])
    ts = s["ts_utc"].to_numpy()

    gen = s.groupby("solar_date")["energy_gen_wh"].max().rename("gen_wh").reset_index()
    gen["date"] = pd.to_datetime(gen["solar_date"])
    ws = windows.dropna(subset=["solar_start_utc", "solar_end_utc"])
    i = np.searchsorted(ts, ws["solar_start_utc"].to_numpy(), "left")
    j = np.searchsorted(ts, ws["solar_end_utc"].to_numpy(), "right")
    day = pd.DataFrame({"date": ws["date"].values, "day_cons_wh": cum[j] - cum[i]})

    n = nights_df[nights_df["covered"]].copy()
    n["date"] = pd.to_datetime(n["date"])
    m = gen.merge(day, on="date").merge(n[["date", "night_wh", "is_ev"]], on="date")
    m["month"] = m["date"].dt.month
    m["gen_kwh"] = m["gen_wh"] / 1000
    m["day_cons_kwh"] = m["day_cons_wh"] / 1000
    m["surplus_kwh"] = m["gen_kwh"] - m["day_cons_kwh"]
    m["night_kwh"] = m["night_wh"] / 1000
    out = m.groupby("month")[["gen_kwh", "day_cons_kwh", "surplus_kwh", "night_kwh"]].median()

    # share of night energy above the inverter cap, by month and EV class
    lo = np.searchsorted(ts, n["night_start_utc"].to_numpy(), "left")
    hi = np.searchsorted(ts, n["night_end_utc"].to_numpy(), "right")
    p = s["power_cons_w"].to_numpy()
    tot = {True: np.zeros(13), False: np.zeros(13)}
    above = {True: np.zeros(13), False: np.zeros(13)}
    for a, b, mo, is_ev in zip(lo, hi, pd.to_datetime(n["date"]).dt.month, n["is_ev"]):
        seg = p[a:b]; seg = seg[~np.isnan(seg)]
        tot[bool(is_ev)][mo] += seg.sum()
        above[bool(is_ev)][mo] += np.clip(seg - CAP_W, 0, None).sum()
    out["above_cap_pct_ev"] = [
        100 * above[True][k] / tot[True][k] if tot[True][k] else 0.0 for k in out.index]
    out["above_cap_pct_nonev"] = [
        100 * above[False][k] / tot[False][k] if tot[False][k] else 0.0 for k in out.index]

    # battery discharge by month at the recommended capacity
    net = battery.net_wh(s)
    night, nonev, month = battery.build_masks(s, nights_df)
    spec = battery.BatterySpec(np.array([recommended_kwh]), 3.0)
    r = battery.simulate(net, night, nonev, month, spec)
    out["discharge_kwh"] = r.monthly_discharge_wh[:, 0] / 1000
    return out.reset_index()


def _coverable_fraction(samples: pd.DataFrame, nights_df: pd.DataFrame,
                        windows: pd.DataFrame) -> float:
    """Fraction of covered nights whose energy that day's daytime surplus
    could have supplied, with an infinite battery and no power limit.

    This is the ceiling the winter months impose: no capacity can beat it.
    Computed rather than quoted, so the report's headline caveat cannot drift
    away from the data it describes.
    """
    s = samples.sort_values("ts_utc").reset_index(drop=True)
    cum = np.concatenate([[0.0], nights.consumption_increments(s).to_numpy().cumsum()])
    ts = s["ts_utc"].to_numpy()
    gen = s.groupby("solar_date")["energy_gen_wh"].max().rename("gen_wh").reset_index()
    gen["date"] = pd.to_datetime(gen["solar_date"])
    ws = windows.dropna(subset=["solar_start_utc", "solar_end_utc"])
    i = np.searchsorted(ts, ws["solar_start_utc"].to_numpy(), "left")
    j = np.searchsorted(ts, ws["solar_end_utc"].to_numpy(), "right")
    day = pd.DataFrame({"date": ws["date"].values, "day_cons_wh": cum[j] - cum[i]})

    n = nights_df[nights_df["covered"]].copy()
    n["date"] = pd.to_datetime(n["date"])
    m = gen.merge(day, on="date").merge(n[["date", "night_wh"]], on="date")
    surplus = m["gen_wh"] - m["day_cons_wh"]
    return float((surplus >= m["night_wh"]).mean())


def run(data_dir: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load(Path(data_dir))
    windows = pd.read_csv(
        Path(data_dir) / "out" / "solar_windows.csv",
        parse_dates=["date", "solar_start_utc", "solar_end_utc",
                     "night_start_utc", "night_end_utc"],
    )
    nights_df = nights.summarise_nights(samples, windows)
    sweep_df = battery.sweep(samples, nights_df, CAPACITIES, POWER_KWS)
    recommended = battery.recommend_capacity(sweep_df, power_kw=3.0)
    sens = nights.ev_sensitivity(samples, windows)
    monthly = _monthly(samples, nights_df, windows, recommended)

    nights_df.to_csv(out_dir / "night_summary.csv", index=False)
    sweep_df.to_csv(out_dir / "battery_sweep.csv", index=False)
    # Derived rather than hardcoded: the winter prose must move with the data.
    covered_nights = nights_df[nights_df["covered"]]
    coverable_pct = 100.0 * _coverable_fraction(samples, nights_df, windows)
    negative_surplus_months = [
        int(m) for m in monthly.loc[monthly["surplus_kwh"] < 0, "month"]
    ]
    (out_dir / "night_report.html").write_text(
        night_report.build_html(
            nights_df, sweep_df, monthly, sens, recommended,
            coverable_pct=coverable_pct,
            negative_surplus_months=negative_surplus_months,
        )
    )

    covered = nights_df[nights_df["covered"]]
    at = sweep_df[(sweep_df.power_kw == 3.0) &
                  np.isclose(sweep_df.capacity_kwh, recommended)].iloc[0]

    # Spec fact 9 is stated as an expectation the simulation must CONFIRM, not
    # as a finding: the 3 kW charge cap clips 15.6% of instantaneous surplus
    # flow in June, but the battery should still fill long before that costs
    # anything. Measure it rather than assume it — how much does going to a
    # 3.7 kW inverter actually buy at the recommended capacity?
    at37 = sweep_df[(sweep_df.power_kw == 3.7) &
                    np.isclose(sweep_df.capacity_kwh, recommended)].iloc[0]
    base = float(at["nonev_night_grid_import_kwh_yr"])
    better = float(at37["nonev_night_grid_import_kwh_yr"])
    power_gain = 100.0 * (base - better) / base if base else 0.0

    return {
        "n_nights": int(len(covered)),
        "n_ev_nights": int(covered["is_ev"].sum()),
        "median_night_kwh": float(covered["night_wh"].median() / 1000),
        "p90_night_kwh": float(covered["night_wh"].quantile(0.9) / 1000),
        "recommended_kwh": float(recommended),
        "night_self_sufficiency_pct": float(at["nonev_night_self_sufficiency_pct"]),
        "cycles_per_yr": float(at["cycles_per_yr"]),
        "pct_gain_from_3p7kw_inverter": power_gain,
    }


if __name__ == "__main__":
    here = Path(__file__).parent
    for k, v in run(here, here / "out").items():
        print(f"{k}: {v}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_analyze_night.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Run the pipeline and the whole suite**

Run: `uv run python analyze_night.py`
Expected: the summary prints; `out/` gains `night_summary.csv`, `battery_sweep.csv`, `night_report.html`.

Run: `uv run pytest -q`
Expected: PASS, 101 tests (62 phase 1 + 13 + 9 + 8 + 8 + 3).

- [ ] **Step 6: Update the README**

Add a section covering: what `analyze_night.py` does, how to run it, what the two new CSVs contain, and the headline finding — that median daytime surplus is negative in November, December and January, so battery sizing is bounded by winter generation rather than capacity. State the recommended capacity and the rule behind it, and the measured `pct_gain_from_3p7kw_inverter` so a reader knows whether inverter power or capacity is the binding constraint. Keep the existing README's short, factual tone.

- [ ] **Step 7: Commit**

```bash
git add analyze_night.py README.md tests/test_analyze_night.py out/night_summary.csv out/battery_sweep.csv
git commit -m "Add night-consumption and battery-sizing entry point"
```

---

## Verification

Before reporting the work complete, invoke `superpowers:verification-before-completion`. Run `uv run pytest` and paste the real summary line; run `uv run python analyze_night.py` and quote the printed figures. Do not describe any number as confirmed unless it appears in output you have actually seen.

## Notes for the implementer

- **Runtime.** `sweep` runs 3 power caps × 589,429 timesteps, each step doing a handful of numpy operations on a 61-element vector. Expect roughly 30–60 seconds. Do not "optimise" by coarsening the timestep or subsampling the years without saying so — the power cap only bites at 5-minute resolution.
- **The simulation timeline is never broken up.** EV nights stay in it; only the *reported* night metrics exclude them. Removing them would leave the battery unphysically full.
- **If a test fails and you believe the assertion is wrong rather than the code**, do not adjust it. Report it with evidence — on this project implementers have correctly overturned the plan author seven times, so your judgement carries weight.
