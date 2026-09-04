# Meter-Based Night Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redo the night-consumption and battery-sizing analysis against the smart meter, which measures the grid connection directly, and publish it alongside phase 2's — whose consumption channel lost part of the load in December 2022.

**Architecture:** The meter's import and export columns are exactly the two channels a battery interacts with, so the simulation needs no reconstructed consumption and is not limited by PVOutput's end date. Phase 2's simulator, sweep, knee and threshold-free readings are reused unchanged apart from one additive parameter; only the input signal and the report are new.

**Tech Stack:** Python ≥3.11, uv, pandas, numpy, matplotlib, openpyxl (new — needed to read `.xlsx`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-meter-based-night-design.md` — read it before starting. Its §2 records the fault this work exists to correct, measured.

**Predecessors:** `docs/superpowers/specs/2026-09-03-solar-window-design.md` (the solar window, unaffected) and `docs/superpowers/specs/2026-09-04-night-battery-design.md` (phase 2).

## Global Constraints

- **No phase-1 module may be modified**: `src/pvnight/{config,loader,solar,events,envelope,report}.py`, `analyze.py`.
- **Of phase 2, only `battery.py` changes**, and only additively: `simulate` gains a `dt_hours` keyword defaulting to the existing `DT_HOURS`. `nights.py`, `night_report.py` and `analyze_night.py` are read-only. Phase 2's numbers must not move.
- **Meter facts, all measured** (spec §2.3): 15-minute intervals; every column reads as a **string**, including the timestamp; **decimal comma**; the paired tariff columns are null when the other tariff is active, so fill zero and sum the pair; timestamps label the interval **end**; 233,358 rows spanning 2019-12-31 23:15 UTC → 2026-09-03 22:00 UTC; **686 missing intervals in seven gaps**, four of them in January 2024.
- **Night consumption is meter import** over the night window. Generation is zero at night, so import *is* household consumption.
- **Daytime surplus is meter export.**
- **Both bounds are run**, never one. Each interval is split into two steps: `charge_first` is `+export` then `−import` (maximum bridging, favourable); `discharge_first` is `−import` then `+export` (minimum bridging, unfavourable). Both reproduce the measured grid import exactly at zero capacity. Report the range, invent no midpoint.
- **No cut-off is introduced.** Phase 2 removed its 50 kWh/yr threshold as a judgement call; the elbow, benefit-share, convergence and elbow-stability readings carry over.
- Battery parameters unchanged from phase 2: round trip 0.90 split as `sqrt(0.90)` each way, usable fraction 0.90, power caps 2.5 / 3.0 / 3.7 kW, capacities 0–30 kWh in 0.5 kWh steps.
- Never assert an exact datetime resolution (pandas 3.x gives microseconds).
- **No embedded rasters in charts** — an embedded PNG trips the Artifact publish scanner. `pcolormesh`, never `imshow`.
- Run everything through `uv`. Tests: `uv run pytest`.

## File Structure

| File | Responsibility |
|---|---|
| `src/pvnight/meter.py` | xlsx → 15-minute UTC frame; gap detection |
| `src/pvnight/meter_nights.py` | Per-night import/export from the meter; EV classification re-derived |
| `src/pvnight/battery.py` | **Modify:** `simulate(..., dt_hours=DT_HOURS)` only |
| `src/pvnight/compare_report.py` | The side-by-side page |
| `analyze_meter.py` | Entry point |
| `tests/test_meter.py`, `test_meter_nights.py`, `test_meter_battery.py`, `test_compare_report.py`, `test_analyze_meter.py` | |

---

### Task 1: Meter loader

**Files:**
- Create: `src/pvnight/meter.py`
- Modify: `pyproject.toml` (add `openpyxl`)
- Test: `tests/test_meter.py`

**Interfaces:**
- Consumes: `pvnight.config.DATA_SUBDIR` exists but is the *PVOutput* folder; this task adds its own.
- Produces:
  - `meter.METER_SUBDIR = "data/meterdata"`, `meter.METER_GLOB = "data_*.xlsx"`, `meter.DT_HOURS = 0.25`
  - `meter.load_meter(meter_dir) -> pd.DataFrame` — columns `ts_utc` (tz-aware UTC, interval END), `import_kwh`, `export_kwh` (float64), sorted by `ts_utc`, index reset.
  - `meter.find_gaps(frame) -> pd.DataFrame` — columns `gap_start_utc`, `gap_end_utc`, `missing_intervals`, one row per gap longer than one interval.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"openpyxl>=3.1"` to the `dependencies` list.

Run: `uv sync`
Expected: openpyxl resolves and installs.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_meter.py`:

```python
import pandas as pd
import pytest

from pvnight.meter import METER_SUBDIR, find_gaps, load_meter


@pytest.fixture(scope="session")
def meter(repo_root):
    return load_meter(repo_root / METER_SUBDIR)


def test_every_column_is_parsed_from_strings(meter):
    """The workbook stores the timestamp AND every energy value as text, with
    a decimal comma. Nothing may survive as an object dtype."""
    assert isinstance(meter["ts_utc"].dtype, pd.DatetimeTZDtype)
    assert str(meter["ts_utc"].dtype.tz) == "UTC"
    assert meter["import_kwh"].dtype == "float64"
    assert meter["export_kwh"].dtype == "float64"


def test_decimal_comma_becomes_a_real_number(meter):
    """'0,04' must be 0.04, not 4.0 and not a string."""
    assert meter["import_kwh"].max() < 20.0     # a 15-min interval, in kWh
    assert meter["export_kwh"].max() < 20.0
    assert (meter["import_kwh"] >= 0).all()
    assert (meter["export_kwh"] >= 0).all()


def test_span_and_row_count_match_the_measured_file_set(meter):
    """Spec §2.3: 233,358 rows, 2019-12-31 23:15 UTC to 2026-09-03 22:00."""
    assert len(meter) == 233358
    assert meter["ts_utc"].iloc[0] == pd.Timestamp("2019-12-31T23:15:00Z")
    assert meter["ts_utc"].iloc[-1] == pd.Timestamp("2026-09-03T22:00:00Z")


def test_timestamps_are_unique_and_ordered(meter):
    assert meter["ts_utc"].is_monotonic_increasing
    assert not meter["ts_utc"].duplicated().any()


def test_both_dst_offsets_land_at_the_right_utc_instant(meter):
    """The workbook writes +0100 in winter and +0200 in summer. A naive parse
    would put the summer rows an hour late."""
    winter = meter[meter["ts_utc"] == pd.Timestamp("2023-01-15T12:00:00Z")]
    summer = meter[meter["ts_utc"] == pd.Timestamp("2023-07-15T12:00:00Z")]
    assert len(winter) == 1
    assert len(summer) == 1


def test_tariff_pairs_are_summed_not_dropped(meter):
    """Each pair is null when the other tariff is active. Summing after a
    zero-fill is correct; dropping nulls would halve the totals."""
    assert meter["import_kwh"].sum() > 25_000     # ~4.5 MWh/yr over 6.7 years
    assert meter["export_kwh"].sum() > 20_000
    assert meter["import_kwh"].isna().sum() == 0
    assert meter["export_kwh"].isna().sum() == 0


def test_find_gaps_reports_all_seven(meter):
    """Spec §2.3 lists seven, four of them in January 2024. They must be
    reported, never silently interpolated."""
    g = find_gaps(meter)
    assert len(g) == 7
    assert int(g["missing_intervals"].sum()) == 686
    jan24 = g[g["gap_start_utc"].dt.strftime("%Y-%m") == "2024-01"]
    assert len(jan24) == 4     # the fifth 2024 gap is in July
    biggest = g.loc[g["missing_intervals"].idxmax()]
    assert biggest["gap_start_utc"] == pd.Timestamp("2024-01-08T23:00:00Z")


def test_find_gaps_on_a_clean_frame_returns_nothing():
    clean = pd.DataFrame({
        "ts_utc": pd.date_range("2023-01-01", periods=10, freq="15min", tz="UTC"),
        "import_kwh": 0.1, "export_kwh": 0.0,
    })
    assert len(find_gaps(clean)) == 0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_meter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.meter'`

- [ ] **Step 4: Implement the loader**

Create `src/pvnight/meter.py`:

```python
"""Read the smart-meter exports.

The meter measures what crosses the grid connection, which is exactly what a
battery interacts with. Unlike PVOutput's consumption channel, it sees the
whole house — see the spec's §2 for the December 2022 fault that makes this
module necessary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METER_SUBDIR = "data/meterdata"
METER_GLOB = "data_*.xlsx"
DT_HOURS = 0.25

_TS_FORMAT = "%d-%m-%Y %H:%M:%S %z"
_IMPORT_COLS = ("levering_normaal", "levering_laag")
_EXPORT_COLS = ("teruglevering_normaal", "teruglevering_laag")


def _decimal_comma(series: pd.Series) -> pd.Series:
    """'0,04' -> 0.04. Everything arrives as text, including the numbers."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )


def load_meter(meter_dir: Path) -> pd.DataFrame:
    """Every yearly workbook, as one UTC-indexed frame.

    Each tariff column is null when the *other* tariff is active, so the pair
    is zero-filled and summed. Dropping nulls instead would halve the totals.
    Timestamps label the interval END, and carry +0100 or +0200 explicitly,
    so they localise without any DST guesswork.
    """
    paths = sorted(Path(meter_dir).glob(METER_GLOB))
    if not paths:
        raise FileNotFoundError(f"no files matching {METER_GLOB!r} in {meter_dir}")

    frames = []
    for p in paths:
        raw = pd.read_excel(p)
        imp = sum(_decimal_comma(raw[c]).fillna(0.0) for c in _IMPORT_COLS)
        exp = sum(_decimal_comma(raw[c]).fillna(0.0) for c in _EXPORT_COLS)
        frames.append(pd.DataFrame({
            "ts_utc": pd.to_datetime(raw["datum_tijd"], format=_TS_FORMAT, utc=True),
            "import_kwh": imp.astype("float64"),
            "export_kwh": exp.astype("float64"),
        }))

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("ts_utc", kind="stable")
        .reset_index(drop=True)
    )


def find_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Every run of missing intervals, reported rather than interpolated.

    Four of the seven gaps fall in January 2024 and together remove most of
    8-19 January — midwinter, when night consumption peaks. A night touching
    one of these must be excluded, not counted as a quiet night.
    """
    ts = frame["ts_utc"]
    step = pd.Timedelta(hours=DT_HOURS)
    delta = ts.diff()
    idx = np.where(delta > step)[0]
    return pd.DataFrame({
        "gap_start_utc": ts.iloc[idx - 1].to_numpy(),
        "gap_end_utc": ts.iloc[idx].to_numpy(),
        "missing_intervals": (delta.iloc[idx] / step - 1).astype(int).to_numpy(),
    })
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_meter.py -v`
Expected: PASS, 8 tests. Reading seven workbooks takes a few seconds.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/pvnight/meter.py tests/test_meter.py
git commit -m "Add smart-meter loader

Every column arrives as text with a decimal comma, and the paired tariff
columns are null when the other tariff is active. Gaps are reported, not
interpolated: four of the seven fall in January 2024."
```

---

### Task 2: Per-night meter figures and the re-derived EV classifier

**Files:**
- Create: `src/pvnight/meter_nights.py`
- Test: `tests/test_meter_nights.py`

**Interfaces:**
- Consumes: `meter.load_meter`, `meter.DT_HOURS`, `meter.find_gaps`; `out/solar_windows.csv` from phase 1.
- Produces:
  - `meter_nights.summarise(meter_df, windows, gaps, ev_power_kw=2.0, ev_hours=2.0) -> pd.DataFrame` — one row per night: `date`, `night_start_utc`, `night_end_utc`, `import_kwh`, `export_kwh`, `peak_kw`, `hours_above_2kw`, `is_ev`, `missing_intervals`, `covered`.
  - `meter_nights.ev_sensitivity(meter_df, windows, gaps, powers=(1.5, 2.0, 2.5, 3.0), hours=(1.0, 2.0, 3.0)) -> pd.DataFrame` — columns `power_kw`, `hours`, `n_ev`, `median_ev_kwh`, `median_rest_kwh`.

**`covered` means:** the night lies inside the meter's span AND contains zero missing intervals. Unlike phase 2 there is no partial-coverage ratio — the meter either recorded an interval or it did not.

**Power from energy:** a 15-minute interval of `x` kWh is a mean of `x / DT_HOURS` kW, i.e. `x * 4`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_meter_nights.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pvnight.meter import METER_SUBDIR, find_gaps, load_meter
from pvnight.meter_nights import ev_sensitivity, summarise


@pytest.fixture(scope="session")
def parts(repo_root):
    m = load_meter(repo_root / METER_SUBDIR)
    w = pd.read_csv(
        repo_root / "out" / "solar_windows.csv",
        parse_dates=["date", "night_start_utc", "night_end_utc"],
    )
    return m, w, find_gaps(m)


@pytest.fixture(scope="session")
def nights(parts):
    return summarise(*parts)


def _toy():
    ts = pd.date_range("2023-06-01T20:00", periods=12, freq="15min", tz="UTC")
    m = pd.DataFrame({"ts_utc": ts, "import_kwh": 0.25, "export_kwh": 0.0})
    w = pd.DataFrame([{
        "date": pd.Timestamp("2023-06-01"),
        "night_start_utc": ts[0], "night_end_utc": ts[-1],
    }])
    return m, w, pd.DataFrame(columns=["gap_start_utc", "gap_end_utc", "missing_intervals"])


def test_night_import_is_the_sum_over_the_window():
    m, w, g = _toy()
    out = summarise(m, w, g)
    assert out.loc[0, "import_kwh"] == pytest.approx(0.25 * 12)


def test_peak_kw_converts_from_interval_energy():
    """0.25 kWh in a quarter hour is a 1 kW mean."""
    m, w, g = _toy()
    out = summarise(m, w, g)
    assert out.loc[0, "peak_kw"] == pytest.approx(1.0)


def test_a_night_touching_a_gap_is_not_covered():
    m, w, _ = _toy()
    gaps = pd.DataFrame([{
        "gap_start_utc": pd.Timestamp("2023-06-01T21:00Z"),
        "gap_end_utc": pd.Timestamp("2023-06-01T22:00Z"),
        "missing_intervals": 3,
    }])
    out = summarise(m, w, gaps)
    assert out.loc[0, "missing_intervals"] == 3
    assert not bool(out.loc[0, "covered"])


def test_real_data_night_totals_agree_with_pvoutput_before_the_fault(nights, repo_root):
    """Spec §7 test 1. For 2020-2021 the two independent sources agree to
    within 3%. This is the evidence the night-extraction method is sound, and
    it must keep passing."""
    pv = pd.read_csv(repo_root / "out" / "night_summary.csv", parse_dates=["date"])
    pv = pv[pv["covered"]].set_index("date")["night_wh"] / 1000
    me = nights[nights["covered"]].set_index("date")["import_kwh"]
    j = pd.concat({"pv": pv, "me": me}, axis=1).dropna()
    j = j[j.index.year.isin([2020, 2021])]
    ratio = j["pv"].sum() / j["me"].sum()
    assert 0.97 <= ratio <= 1.03, f"2020-21 agreement drifted: {ratio:.3f}"


def test_real_data_shows_the_known_divergence_after_the_fault(nights, repo_root):
    """Spec §7 test 2. 2023/2024/2025 shortfalls measured at 24.9/26.7/30.8%.
    Assert 20-35% per year — wide enough that 24.9 is not on a cliff edge,
    narrow enough that the fault must still be there."""
    pv = pd.read_csv(repo_root / "out" / "night_summary.csv", parse_dates=["date"])
    pv = pv[pv["covered"]].set_index("date")["night_wh"] / 1000
    me = nights[nights["covered"]].set_index("date")["import_kwh"]
    j = pd.concat({"pv": pv, "me": me}, axis=1).dropna()
    for year in (2023, 2024, 2025):
        y = j[j.index.year == year]
        short = 100 * (1 - y["pv"].sum() / y["me"].sum())
        assert 20.0 <= short <= 35.0, f"{year} shortfall {short:.1f}% outside 20-35"


def test_the_december_2022_step_is_visible(nights, repo_root):
    """Spec §7 test 3. November 2022 >= 0.95, December 2022 <= 0.85."""
    pv = pd.read_csv(repo_root / "out" / "night_summary.csv", parse_dates=["date"])
    pv = pv[pv["covered"]].set_index("date")["night_wh"] / 1000
    me = nights[nights["covered"]].set_index("date")["import_kwh"]
    j = pd.concat({"pv": pv, "me": me}, axis=1).dropna()
    j = j[j["me"] > 0.5]
    nov = (j[j.index.to_period("M") == "2022-11"]["pv"]
           / j[j.index.to_period("M") == "2022-11"]["me"]).median()
    dec = (j[j.index.to_period("M") == "2022-12"]["pv"]
           / j[j.index.to_period("M") == "2022-12"]["me"]).median()
    assert nov >= 0.95
    assert dec <= 0.85


def test_january_2024_outage_removes_nights_rather_than_shrinking_them(nights):
    """Four gaps remove most of 8-19 January 2024. Those nights must be
    excluded, not counted as unusually quiet midwinter nights."""
    jan = nights[(nights["date"] >= "2024-01-08") & (nights["date"] <= "2024-01-19")]
    assert len(jan) > 0
    assert not jan["covered"].all(), "the outage nights should not all be covered"


def test_ev_sensitivity_grid_has_a_row_per_combination(parts):
    g = ev_sensitivity(*parts)
    assert len(g) == 12
    assert set(g.columns) == {"power_kw", "hours", "n_ev", "median_ev_kwh", "median_rest_kwh"}


def test_the_meter_sees_more_ev_nights_than_pvoutput_did(nights):
    """Phase 2 found 72 EV nights from a signal missing much of the car. The
    meter sees the whole load, so at the same threshold it should find at
    least as many."""
    assert int(nights.loc[nights["covered"], "is_ev"].sum()) >= 72
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_meter_nights.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.meter_nights'`

- [ ] **Step 3: Implement**

Create `src/pvnight/meter_nights.py`:

```python
"""Per-night figures taken from the meter rather than from PVOutput.

At night there is no generation, so meter import *is* household consumption.
That equivalence is what lets this module answer the same question phase 2
answered, from a source that sees the whole house.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .meter import DT_HOURS

EV_POWER_KW = 2.0
EV_HOURS = 2.0


def _overlaps_a_gap(starts, ends, gaps: pd.DataFrame) -> np.ndarray:
    """Missing intervals inside each night. Zero when the frame is clean."""
    out = np.zeros(len(starts), dtype=int)
    if gaps.empty:
        return out
    gs = gaps["gap_start_utc"].to_numpy()
    ge = gaps["gap_end_utc"].to_numpy()
    n = gaps["missing_intervals"].to_numpy()
    for k, (a, b) in enumerate(zip(starts, ends)):
        hit = (gs < b) & (ge > a)
        out[k] = int(n[hit].sum())
    return out


def summarise(
    meter_df: pd.DataFrame,
    windows: pd.DataFrame,
    gaps: pd.DataFrame,
    ev_power_kw: float = EV_POWER_KW,
    ev_hours: float = EV_HOURS,
) -> pd.DataFrame:
    """One row per night, from the meter.

    `covered` means the night sits inside the meter's span and contains no
    missing intervals. There is no partial-coverage ratio as in phase 2: the
    meter either recorded an interval or it did not.
    """
    m = meter_df.sort_values("ts_utc").reset_index(drop=True)
    ts = m["ts_utc"].to_numpy()
    imp = m["import_kwh"].to_numpy()
    cum_i = np.concatenate([[0.0], imp.cumsum()])
    cum_e = np.concatenate([[0.0], m["export_kwh"].to_numpy().cumsum()])

    w = windows.dropna(subset=["night_start_utc", "night_end_utc"]).reset_index(drop=True)
    starts = w["night_start_utc"].to_numpy()
    ends = w["night_end_utc"].to_numpy()
    lo = np.searchsorted(ts, starts, side="left")
    hi = np.searchsorted(ts, ends, side="right")

    peak = np.zeros(len(w))
    hours_hi = np.zeros(len(w))
    for k, (a, b) in enumerate(zip(lo, hi)):
        seg = imp[a:b]
        if len(seg):
            peak[k] = seg.max() / DT_HOURS
            hours_hi[k] = (seg / DT_HOURS > ev_power_kw).sum() * DT_HOURS

    missing = _overlaps_a_gap(starts, ends, gaps)
    inside = (starts >= ts[0]) & (ends <= ts[-1])

    out = pd.DataFrame({
        "date": w["date"],
        "night_start_utc": w["night_start_utc"],
        "night_end_utc": w["night_end_utc"],
        "import_kwh": cum_i[hi] - cum_i[lo],
        "export_kwh": cum_e[hi] - cum_e[lo],
        "peak_kw": peak,
        "hours_above_2kw": hours_hi,
        "missing_intervals": missing,
    })
    out["is_ev"] = out["hours_above_2kw"] >= ev_hours
    out["covered"] = inside & (missing == 0) & (hi > lo)
    return out


def ev_sensitivity(
    meter_df: pd.DataFrame,
    windows: pd.DataFrame,
    gaps: pd.DataFrame,
    powers: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0),
    hours: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """How the EV split moves as the threshold moves.

    Republished because phase 2's rule was fitted to a signal missing much of
    the car; the threshold that separated a partial signal is not necessarily
    the one that separates a complete one.
    """
    rows = []
    for p in powers:
        base = summarise(meter_df, windows, gaps, ev_power_kw=p, ev_hours=min(hours))
        base = base[base["covered"]].rename(columns={"hours_above_2kw": "hours_above_p"})
        for h in hours:
            is_ev = base["hours_above_p"] >= h
            rows.append({
                "power_kw": p,
                "hours": h,
                "n_ev": int(is_ev.sum()),
                "median_ev_kwh": base.loc[is_ev, "import_kwh"].median(),
                "median_rest_kwh": base.loc[~is_ev, "import_kwh"].median(),
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_meter_nights.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/meter_nights.py tests/test_meter_nights.py
git commit -m "Add per-night meter figures and re-derived EV classifier

Pins the 2020-21 agreement, the 2023-25 divergence and the December 2022
step as regression tests: the fault is a fact about the data."
```

---

### Task 3: Both-bounds simulation at meter resolution

**Files:**
- Modify: `src/pvnight/battery.py` (add `dt_hours` keyword to `simulate`)
- Create: `src/pvnight/meter_battery.py`
- Test: `tests/test_meter_battery.py`

**Interfaces:**
- Consumes: `battery.BatterySpec`, `battery.simulate`, `battery.sweep`'s metric shape, `battery.elbow_capacity`, `battery.benefit_share_pct`, `battery.convergence_readings`; `meter.DT_HOURS`.
- Produces:
  - `battery.simulate(net_wh, night_mask, nonev_mask, month_idx, spec, dt_hours=DT_HOURS)` — additive keyword.
  - `meter_battery.bound_signals(meter_df) -> tuple[np.ndarray, np.ndarray]` — `(net_wh, gross_wh)`; `net` is `(export − import) * 1000`, `gross` is the same but with import and export treated as separable within the interval.
  - `meter_battery.sweep_bounds(meter_df, nights_df, capacities_kwh, power_kws=(2.5, 3.0, 3.7)) -> pd.DataFrame` — phase 2's sweep columns plus a `bound` column of `"charge_first"` or `"discharge_first"`.

**On the two bounds.** A 15-minute interval can contain both import and export (13.5% do). The `net` signal collapses them, so a battery never sees the export it could have stored — an understatement. The `gross` signal presents the export for charging and the import for discharging in the same interval, which a real battery could partly do — an overstatement. The truth is between; both are run and neither is called the answer.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_meter_battery.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pvnight.battery import DT_HOURS, BatterySpec, simulate
from pvnight.meter_battery import bound_signals, sweep_bounds


def test_dt_hours_actually_changes_the_power_cap():
    """If the parameter were decorative, the same series would give the same
    answer at both resolutions. A 3 kW cap passes 250 Wh in five minutes and
    750 Wh in fifteen."""
    net = np.array([50_000.0, -5000.0])
    z = np.zeros(2, bool)
    mi = np.zeros(2, int)
    spec = BatterySpec(np.array([20.0]), power_kw=3.0, round_trip=1.0)
    five = simulate(net, z, z, mi, spec, dt_hours=5 / 60)
    fifteen = simulate(net, z, z, mi, spec, dt_hours=0.25)
    assert five.discharge_wh[0] == pytest.approx(3000.0 * 5 / 60)
    assert fifteen.discharge_wh[0] == pytest.approx(3000.0 * 0.25)
    assert fifteen.discharge_wh[0] > five.discharge_wh[0]


def test_simulate_without_dt_hours_is_unchanged():
    """Phase 2's numbers must not move."""
    net = np.array([1000.0, -400.0, -400.0])
    z = np.zeros(3, bool)
    mi = np.zeros(3, int)
    spec = BatterySpec(np.array([5.0]), power_kw=3.0)
    a = simulate(net, z, z, mi, spec)
    b = simulate(net, z, z, mi, spec, dt_hours=DT_HOURS)
    assert a.grid_import_wh[0] == pytest.approx(b.grid_import_wh[0])
    assert a.discharge_wh[0] == pytest.approx(b.discharge_wh[0])


def _toy_meter():
    ts = pd.date_range("2023-06-01", periods=200, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    imp = np.abs(rng.normal(0.10, 0.05, 200))
    exp = np.abs(rng.normal(0.30, 0.20, 200))
    return pd.DataFrame({"ts_utc": ts, "import_kwh": imp, "export_kwh": exp})


def test_gross_never_gives_a_worse_result_than_net():
    """The two bounds are ordered by construction: gross offers the battery
    strictly more to work with."""
    m = _toy_meter()
    n = pd.DataFrame([{
        "date": m.ts_utc.iloc[0].date(),
        "night_start_utc": m.ts_utc.iloc[0], "night_end_utc": m.ts_utc.iloc[-1],
        "is_ev": False, "covered": True,
    }])
    out = sweep_bounds(m, n, np.arange(0.0, 6.1, 1.0), power_kws=(3.0,))
    net = out[out.bound == "net"].sort_values("capacity_kwh")
    gross = out[out.bound == "gross"].sort_values("capacity_kwh")
    assert (gross.grid_import_kwh_yr.to_numpy()
            <= net.grid_import_kwh_yr.to_numpy() + 1e-9).all()


def test_sweep_bounds_labels_both_bounds():
    m = _toy_meter()
    n = pd.DataFrame([{
        "date": m.ts_utc.iloc[0].date(),
        "night_start_utc": m.ts_utc.iloc[0], "night_end_utc": m.ts_utc.iloc[-1],
        "is_ev": False, "covered": True,
    }])
    out = sweep_bounds(m, n, np.array([0.0, 5.0]), power_kws=(3.0,))
    assert set(out["bound"].unique()) == {"net", "gross"}
    assert len(out) == 4


def test_bound_signals_differ_only_where_both_flows_are_present():
    m = pd.DataFrame({
        "ts_utc": pd.date_range("2023-06-01", periods=3, freq="15min", tz="UTC"),
        "import_kwh": [0.0, 0.5, 0.2],
        "export_kwh": [0.4, 0.0, 0.3],
    })
    net, gross = bound_signals(m)
    assert net[0] == pytest.approx(gross[0])      # export only
    assert net[1] == pytest.approx(gross[1])      # import only
    assert gross[2] != pytest.approx(net[2])      # both present
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_meter_battery.py -v`
Expected: FAIL — `TypeError: simulate() got an unexpected keyword argument 'dt_hours'`

- [ ] **Step 3: Add the keyword to `battery.simulate`**

In `src/pvnight/battery.py`, change the signature and the one line that uses the constant:

```python
def simulate(
    net_wh: np.ndarray,
    night_mask: np.ndarray,
    nonev_mask: np.ndarray,
    month_idx: np.ndarray,
    spec: BatterySpec,
    dt_hours: float = DT_HOURS,
) -> SimResult:
```

and replace `limit_wh = spec.power_kw * 1000.0 * DT_HOURS` with:

```python
    # The interval length is a parameter because the meter data is 15-minute
    # where PVOutput is 5-minute. The power cap converts to an energy limit
    # per interval, so it is the one place resolution genuinely bites.
    limit_wh = spec.power_kw * 1000.0 * dt_hours
```

Change nothing else in that file.

- [ ] **Step 4: Implement `meter_battery.py`**

Create `src/pvnight/meter_battery.py`:

```python
"""Battery simulation driven by the meter, at both bounds of the
within-interval ambiguity.

A 15-minute interval can record both import and export, and 13.5% of them do.
Collapsing them to a net figure hides export the battery could have stored;
treating them as fully separable credits it with more than it could certainly
have done. Both are run, and the pair is the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import BatterySpec, simulate
from .meter import DT_HOURS


def bound_signals(meter_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(net_wh, gross_wh)` per interval, in Wh.

    `net` is export minus import, the conservative reading. `gross` keeps the
    larger flow but adds back the part that cancelled, so a battery is offered
    the export it could have captured before the import arrived.
    """
    m = meter_df.sort_values("ts_utc")
    imp = m["import_kwh"].to_numpy(dtype=float) * 1000.0
    exp = m["export_kwh"].to_numpy(dtype=float) * 1000.0
    net = exp - imp
    both = np.minimum(imp, exp)
    gross = net + np.sign(net) * both
    gross = np.where(net == 0.0, exp, gross)
    return net, gross


def _masks(meter_df: pd.DataFrame, nights_df: pd.DataFrame):
    m = meter_df.sort_values("ts_utc")
    ts = m["ts_utc"].to_numpy()
    night = np.zeros(len(m), bool)
    nonev = np.zeros(len(m), bool)
    n = nights_df.dropna(subset=["night_start_utc", "night_end_utc"])
    lo = np.searchsorted(ts, n["night_start_utc"].to_numpy(), "left")
    hi = np.searchsorted(ts, n["night_end_utc"].to_numpy(), "right")
    for a, b, is_ev, covered in zip(lo, hi, n["is_ev"], n["covered"]):
        if not covered:
            continue
        night[a:b] = True
        if not is_ev:
            nonev[a:b] = True
    return night, nonev, pd.DatetimeIndex(m["ts_utc"]).month.to_numpy() - 1


def sweep_bounds(
    meter_df: pd.DataFrame,
    nights_df: pd.DataFrame,
    capacities_kwh: np.ndarray,
    power_kws: tuple[float, ...] = (2.5, 3.0, 3.7),
    round_trip: float = 0.90,
    usable_fraction: float = 0.90,
) -> pd.DataFrame:
    """Phase 2's sweep metrics, once per bound."""
    m = meter_df.sort_values("ts_utc")
    night, nonev, month = _masks(m, nights_df)
    span = (m["ts_utc"].iloc[-1] - m["ts_utc"].iloc[0]).total_seconds()
    years = span / (365.25 * 24 * 3600)
    signals = dict(zip(("net", "gross"), bound_signals(m)))

    frames = []
    for name, sig in signals.items():
        night_load = -sig[night & (sig < 0)].sum()
        nonev_load = -sig[nonev & (sig < 0)].sum()
        night_load = night_load if night_load > 0 else np.nan
        nonev_load = nonev_load if nonev_load > 0 else np.nan
        for p in power_kws:
            spec = BatterySpec(np.asarray(capacities_kwh, float), p,
                               round_trip, usable_fraction)
            r = simulate(sig, night, nonev, month, spec, dt_hours=DT_HOURS)
            usable = np.where(spec.usable_wh > 0, spec.usable_wh, np.nan)
            df = pd.DataFrame({
                "capacity_kwh": r.capacities_kwh,
                "power_kw": p,
                "bound": name,
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
            df["nonev_marginal_kwh_per_kwh"] = (
                -df["nonev_night_grid_import_kwh_yr"].diff() / step)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_meter_battery.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 6: Confirm phase 2 is untouched**

Run: `uv run pytest tests/test_battery.py tests/test_sweep.py tests/test_analyze_night.py -q`
Expected: PASS, unchanged counts. If any phase-2 number moved, stop and report it — the `dt_hours` default must reproduce the old behaviour exactly.

- [ ] **Step 7: Commit**

```bash
git add src/pvnight/battery.py src/pvnight/meter_battery.py tests/test_meter_battery.py
git commit -m "Simulate the meter signal at both within-interval bounds

battery.simulate gains an additive dt_hours keyword; phase 2's default
behaviour is unchanged and its tests still pass."
```

---

### Task 4: The comparison report

**Files:**
- Create: `src/pvnight/compare_report.py`
- Test: `tests/test_compare_report.py`

**Interfaces:**
- Consumes: `pvnight.report`'s `_svg`, `SERIES`, `FURNITURE`, `STYLE`; `pvnight.night_report`'s `TABLE_CSS`; `battery.elbow_capacity`, `battery.benefit_share_pct`, `battery.convergence_readings`.
- Produces: `compare_report.build_html(meter_nights, pv_nights, meter_sweep, pv_sweep, sensitivity, monthly_ratio, gaps) -> str`.

**Before writing chart code, load the `dataviz` skill, then `artifact-design`.** Phase 1's palette in `report.py` is already validated against the skill's `palette.md`; import it, do not re-pick.

**`monthly_ratio`** is built by Task 5: columns `month` (a `Period`), `pv_kwh`, `meter_kwh`, `ratio`.

Six charts:

1. `chart_visible_fraction(monthly_ratio)` — the monthly ratio through the whole record, with the November→December 2022 step annotated. This is the evidence for everything else.
2. `chart_night_distribution(meter_nights, pv_nights)` — night-energy CDFs, both sources, restricted to the overlap.
3. `chart_bounds(meter_sweep, elbow_net, elbow_gross)` — night grid import against capacity for both bounds, the band between them shaded.
4. `chart_marginal(meter_sweep)` — marginal return for both bounds, non-EV series.
5. `chart_comparison(meter_sweep, pv_sweep)` — the two analyses' non-EV curves on one axis, phase 2's drawn dashed and labelled superseded.
6. `chart_coverage(gaps, meter_nights)` — nights excluded per month, so the January 2024 outage is visible rather than silently absent.

**The page must, and each is asserted by a test:**

- Lead with the meter figures and state that phase 2's consumption channel was faulty from December 2022. The words "superseded" and "December 2022" must both appear.
- Report the capacity as a **range across the two bounds**, never a single number.
- State the measured resolution penalty from Task 5.
- Carry the word "heuristic" where the EV rule is introduced, above the republished sensitivity table.
- Note that January 2024 nights are excluded and how many.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compare_report.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pvnight.compare_report import build_html


@pytest.fixture
def toy():
    d = pd.date_range("2021-01-01", periods=400, freq="D")
    mn = pd.DataFrame({"date": d, "import_kwh": np.linspace(2, 30, 400),
                       "export_kwh": 0.0, "peak_kw": 1.0,
                       "hours_above_2kw": 0.0, "is_ev": False,
                       "missing_intervals": 0, "covered": True})
    pn = pd.DataFrame({"date": d, "night_wh": np.linspace(2, 25, 400) * 1000,
                       "covered": True, "is_ev": False})
    caps = np.arange(0.0, 20.1, 0.5)
    def mk(bound, scale):
        return pd.DataFrame({
            "capacity_kwh": caps, "power_kw": 3.0, "bound": bound,
            "grid_import_kwh_yr": 3000 * scale * np.exp(-caps / 6),
            "pv_export_kwh_yr": 2000 * np.exp(-caps / 8),
            "night_grid_import_kwh_yr": 1500 * scale * np.exp(-caps / 5),
            "nonev_night_grid_import_kwh_yr": 1200 * scale * np.exp(-caps / 5),
            "night_self_sufficiency_pct": 100 * (1 - np.exp(-caps / 5)),
            "nonev_night_self_sufficiency_pct": 100 * (1 - np.exp(-caps / 5)),
            "cycles_per_yr": np.r_[np.nan, 200 / caps[1:]],
            "marginal_kwh_per_kwh": np.r_[np.nan, -np.diff(3000 * scale * np.exp(-caps / 6))],
            "nonev_marginal_kwh_per_kwh": np.r_[np.nan, -np.diff(1200 * scale * np.exp(-caps / 5))],
        })
    ms = pd.concat([mk("net", 1.0), mk("gross", 0.92)], ignore_index=True)
    ps = mk("net", 0.75).drop(columns=["bound"])
    sens = pd.DataFrame({"power_kw": np.repeat([1.5, 2.0, 2.5, 3.0], 3),
                         "hours": np.tile([1.0, 2.0, 3.0], 4),
                         "n_ev": np.arange(12) + 60,
                         "median_ev_kwh": np.linspace(20, 35, 12),
                         "median_rest_kwh": np.linspace(5, 6, 12)})
    mr = pd.DataFrame({"month": pd.period_range("2020-01", "2025-12", freq="M")})
    mr["pv_kwh"] = 100.0
    mr["meter_kwh"] = np.where(mr.month < pd.Period("2022-12"), 102.0, 130.0)
    mr["ratio"] = mr.pv_kwh / mr.meter_kwh
    gaps = pd.DataFrame({
        "gap_start_utc": pd.to_datetime(["2024-01-08T23:00Z"]),
        "gap_end_utc": pd.to_datetime(["2024-01-14T08:45Z"]),
        "missing_intervals": [518]})
    return mn, pn, ms, ps, sens, mr, gaps


def test_page_has_six_charts_and_no_rasters(toy):
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    assert html.count("<svg") == 6
    assert "<image" not in html
    assert "data:image" not in html


def test_page_omits_the_document_wrapper(toy):
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    for tag in ["<!doctype", "<html", "<head>", "<body>"]:
        assert tag not in html.lower()


def test_page_labels_the_superseded_analysis(toy):
    """Side-by-side publication is only safe if the page says which is which."""
    low = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12).lower()
    assert "superseded" in low
    assert "december 2022" in low


def test_page_reports_a_range_across_the_two_bounds(toy):
    low = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12).lower()
    assert "bound" in low
    assert "net" in low and "gross" in low


def test_page_states_the_measured_resolution_penalty(toy):
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    assert "4.2" in html
    assert "15-minute" in html or "15 minute" in html


def test_page_discloses_the_january_2024_exclusions(toy):
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    assert "12" in html
    assert "January 2024" in html or "2024-01" in html


def test_page_calls_the_ev_rule_a_heuristic(toy):
    low = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12).lower()
    assert "heuristic" in low
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compare_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.compare_report'`

- [ ] **Step 3: Implement**

Create `src/pvnight/compare_report.py`. Import the shared style helpers rather than redefining them:

```python
"""The side-by-side page: meter-based figures, with phase 2 shown and labelled."""

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
```

Write the six charts named above, each returning `_svg(fig)`, following the
idiom already used in `night_report.py`: `figsize=(9, 4)` or `(9, 3.5)`,
`ax.legend(frameon=False, labelcolor=FURNITURE, fontsize=8)`, colours from
`SERIES`, axis furniture in `FURNITURE`. For `chart_bounds`, shade between the
two bounds with `ax.fill_between(..., alpha=0.2)` so the band reads as the
answer rather than either edge.

Then `build_html(meter_nights, pv_nights, meter_sweep, pv_sweep, sensitivity,
monthly_ratio, gaps, resolution_penalty_pct, excluded_nights)` assembling
`<title>Night Consumption, Measured at the Meter</title>`, `STYLE`,
`TABLE_CSS`, a stat row, the six chart cards, the republished EV sensitivity
table, and — per spec §4.4, which names four threshold-free readings, not
two — the convergence table (`battery.convergence_readings`), the
benefit-share column (`battery.benefit_share_pct`) and the elbow-stability
table (`battery.elbow_stability`), all applied to the `net` bound. Import
`elbow_stability` alongside the others.

The prose must carry, in its own words: that the meter figures lead and phase
2's are **superseded** because its consumption channel was faulty from
**December 2022**; the capacity as a range between the `net` and `gross`
bounds with both named; the measured `resolution_penalty_pct` against the
**15-minute** interval; the count of `excluded_nights` around **January
2024**; and the word **heuristic** where the EV rule is introduced.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_compare_report.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/compare_report.py tests/test_compare_report.py
git commit -m "Add the meter-versus-PVOutput comparison page"
```

---

### Task 5: Entry point, resolution penalty and outputs

**Files:**
- Create: `analyze_meter.py`
- Modify: `README.md`
- Test: `tests/test_analyze_meter.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `analyze_meter.run(repo_root, out_dir) -> dict` writing `out/meter_night_summary.csv`, `out/meter_battery_sweep.csv`, `out/meter_report.html`, returning `n_nights`, `n_ev_nights`, `median_night_kwh`, `elbow_net_kwh`, `elbow_gross_kwh`, `resolution_penalty_pct`, `excluded_nights_jan_2024`, `pv_shortfall_2025_pct`.

**The resolution penalty** is measured, not asserted: load phase 2's 5-minute PVOutput frame, build its net signal, sum each consecutive three samples into 15-minute totals, run the same capacity sweep at both resolutions, and report the percentage difference in `nonev_night_grid_import_kwh_yr` at the 5-minute elbow. Both resolutions genuinely exist for that dataset, so this is measured on real data.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_meter.py`:

```python
import pandas as pd
import pytest

from analyze_meter import run


def test_run_writes_all_three_outputs(tmp_path, repo_root):
    s = run(repo_root, tmp_path)

    nights = pd.read_csv(tmp_path / "meter_night_summary.csv")
    sweep = pd.read_csv(tmp_path / "meter_battery_sweep.csv")
    assert len(nights) > 2000
    assert set(sweep["bound"].unique()) == {"net", "gross"}
    assert set(sweep["power_kw"].unique()) == {2.5, 3.0, 3.7}
    assert (tmp_path / "meter_report.html").read_text().count("<svg") == 6

    assert s["n_nights"] > 2000
    assert 0 < s["elbow_net_kwh"] <= 30
    assert 0 < s["elbow_gross_kwh"] <= 30
    assert s["elbow_gross_kwh"] <= s["elbow_net_kwh"] + 1e-9


def test_the_meter_night_median_exceeds_pvoutputs(tmp_path, repo_root):
    """The whole point: PVOutput was missing load, so the meter's median
    night must be the larger figure."""
    s = run(repo_root, tmp_path)
    pv = pd.read_csv(repo_root / "out" / "night_summary.csv")
    pv_median = pv[pv["covered"]]["night_wh"].median() / 1000
    assert s["median_night_kwh"] > pv_median


def test_resolution_penalty_is_measured_and_plausible(tmp_path, repo_root):
    """Averaging hides peaks, so the coarser run should look no worse than
    the finer one, and the gap should be single-digit percent."""
    s = run(repo_root, tmp_path)
    assert -1.0 <= s["resolution_penalty_pct"] <= 25.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_analyze_meter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analyze_meter'`

- [ ] **Step 3: Implement `analyze_meter.py`**

Create `analyze_meter.py` at the repository root, orchestrating: `load_meter`
→ `find_gaps` → read `out/solar_windows.csv` → `meter_nights.summarise` →
`meter_battery.sweep_bounds` over `np.arange(0.0, 30.01, 0.5)` →
`elbow_capacity` per bound → `meter_nights.ev_sensitivity` → the monthly ratio
against `out/night_summary.csv` → the resolution-penalty measurement described
above → `compare_report.build_html` → write the two CSVs and the HTML →
return the summary dict.

Use `Path(__file__).parent` for the default `repo_root` under `__main__`, and
resolve `out/solar_windows.csv` and `out/night_summary.csv` from `repo_root`,
never from a data directory — phase 3's spec §5 records that phase 2 was bitten
by deriving an output path from an input path.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_analyze_meter.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Run the pipeline and the whole suite**

Run: `uv run python analyze_meter.py`
Expected: the summary prints; `out/` gains the two CSVs and `meter_report.html`.

Run: `uv run pytest -q`
Expected: PASS. 121 from phases 1-2 plus 8 + 9 + 5 + 7 + 3 = 153.

- [ ] **Step 6: Update the README**

Add a section covering what `analyze_meter.py` does, how to run it, what the
two new CSVs contain, and — stated plainly — that PVOutput's consumption
channel undercounts from December 2022, that the meter-based figures
supersede phase 2's, and that phase 2's output is retained for comparison
only. Quote the measured 2025 shortfall. Keep the README's existing short,
factual tone.

- [ ] **Step 7: Commit**

```bash
git add analyze_meter.py README.md tests/test_analyze_meter.py out/meter_night_summary.csv out/meter_battery_sweep.csv
git commit -m "Add the meter-based entry point and outputs"
```

---

## Verification

Before reporting complete, invoke `superpowers:verification-before-completion`.
Run `uv run pytest` and paste the real summary line; run `uv run python
analyze_meter.py` and quote the printed figures. Do not describe any number as
confirmed unless it appears in output you have seen.

## Notes for the implementer

- **Runtime.** The meter sweep is 2 bounds × 3 power caps × 233,358 intervals
  against a 61-element capacity vector. Expect a minute or two. The
  resolution-penalty measurement re-runs phase 2's data twice more. Do not
  coarsen the sweep to speed it up without saying so.
- **Phase 2's numbers must not move.** The only change to its code is an
  additive keyword with the old value as default. Task 3 step 6 checks this
  explicitly; if it fails, stop rather than adjusting phase 2's tests.
- **The gaps are not noise.** Five of seven fall in January 2024 and remove
  most of 8-19 January. Nights touching them are excluded, and the report says
  how many. Treating them as low-consumption nights would understate midwinter
  demand — the season that already dominates the answer.
- **If a test fails and you believe the assertion is wrong rather than the
  code**, do not adjust it. Report it with evidence — implementers on this
  project have correctly overturned the plan author twelve times.
