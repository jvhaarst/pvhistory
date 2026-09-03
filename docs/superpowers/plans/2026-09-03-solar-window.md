# Solar Window Determination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, for every calendar date, the window in which solar generation is physically possible at this site, so that night can be defined as its complement.

**Architecture:** Six years of PVOutput 5-minute history are loaded into a UTC-indexed frame. For each solar day the first and last generating samples are found, and the sun's geometric elevation at those instants is computed with pvlib. Those elevations are pooled across years in a circular ±10-day window, reduced to a 5th percentile, and smoothed with a 2-harmonic Fourier fit, giving θ_start(φ) and θ_end(φ). Converting a threshold back to a clock time is done by evaluating elevation on a one-minute UTC grid and finding the crossing.

**Tech Stack:** Python ≥3.11, uv, pandas, pyarrow, pvlib, numpy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-solar-window-design.md` — read it before starting. The plan argues from the spec; the numbered "established facts" in spec §2.1 are referenced throughout and explain *why* several non-obvious choices are made.

## Global Constraints

These apply to every task. Values are copied verbatim from the spec.

- **Site:** latitude `51.98`, longitude `5.80`, timezone `Europe/Amsterdam`.
- **Geometric elevation only.** Use pvlib's `elevation` column, never `apparent_elevation`. pvlib applies no refraction correction below the horizon, so `apparent_elevation` has a kink in its derivative at 0° and 36% of our events lie below it (spec fact 11).
- **UTC internally, local only for grouping and display.** Localize once at load, `tz_convert("UTC")`, and compute everything in UTC. The only authoritative use of local time is the `solar_date` grouping key (spec §5).
- **`sun_rise_set_transit_spa` must be given local noon.** Local midnight silently returns the *previous* day's sunrise (spec fact 10).
- **Never pass these files through `lzma`.** Despite the `.xz` suffix they are plain Parquet; `pandas.read_parquet` reads them directly (spec §2).
- **Null generation values are zero**, not missing (spec fact 2).
- **Sunrise elevation constant:** `-0.8358` degrees.
- **Model parameters:** pooling half-width 10 days, percentile 5.0, 2 harmonics, reference year 2024.
- Run everything through `uv`. Tests: `uv run pytest`.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project definition, deps, pytest config |
| `src/pvnight/config.py` | Site constants and model parameters. No logic. |
| `src/pvnight/loader.py` | Parquet → tidy UTC frame + `generating` flag |
| `src/pvnight/solar.py` | All pvlib contact: elevation, azimuth, sunrise/sunset, UTC day grid, elevation crossings |
| `src/pvnight/events.py` | Per solar day: first/last light instants. Timestamps only, no astronomy. |
| `src/pvnight/envelope.py` | Year angle, circular pooling, Fourier fit, and applying the model to build the window table |
| `src/pvnight/report.py` | Charts and HTML page |
| `analyze.py` | Entry point: runs the pipeline, writes CSVs and the report |
| `tests/` | One test module per source module, plus `conftest.py` |

---

### Task 1: Project scaffolding, config, and loader

**Files:**
- Create: `pyproject.toml`, `src/pvnight/__init__.py`, `src/pvnight/config.py`, `src/pvnight/loader.py`
- Test: `tests/conftest.py`, `tests/test_loader.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pvnight.config` module constants: `LATITUDE: float`, `LONGITUDE: float`, `SITE_TZ: str`, `DATA_GLOB: str`, `SUNRISE_ELEVATION_DEG: float`, `POOL_HALF_WIDTH_DAYS: int`, `PERCENTILE: float`, `N_HARMONICS: int`, `REFERENCE_YEAR: int`
  - `pvnight.loader.generating_flag(df: pd.DataFrame) -> pd.Series` — takes a frame with columns `solar_date`, `power_gen_w`, `power_avg_w`, `energy_gen_wh` sorted chronologically; returns a boolean Series named `generating`.
  - `pvnight.loader.load(data_dir: Path) -> pd.DataFrame` — columns `ts_utc` (tz-aware UTC), `solar_date` (`datetime.date`), `power_gen_w`, `power_avg_w`, `energy_gen_wh`, `power_cons_w`, `energy_cons_wh` (all float64), `generating` (bool). Sorted by `ts_utc`, index reset.

- [ ] **Step 1: Create the project skeleton**

Create `pyproject.toml`:

```toml
[project]
name = "pvnight"
version = "0.1.0"
description = "Determine the daily solar-possible window from PVOutput history"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=16",
    "pvlib>=0.11",
    "numpy>=1.26",
    "matplotlib>=3.8",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pvnight"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# analyze.py lives at the repo root, outside the installed package, so the
# root must be importable for tests/test_analyze.py.
pythonpath = ["."]
filterwarnings = ["ignore::DeprecationWarning"]
```

Create an empty `src/pvnight/__init__.py`.

Create `src/pvnight/config.py`:

```python
"""Site constants and model parameters. No logic lives here."""

LATITUDE = 51.98
LONGITUDE = 5.80
SITE_TZ = "Europe/Amsterdam"

DATA_GLOB = "pvoutput_gethistory.*.parquet.xz"

# Geometric elevation of the sun at SPA sunrise/sunset, in degrees.
# Measured against pvlib 0.15.2: -0.8359 to -0.8350 across a year.
SUNRISE_ELEVATION_DEG = -0.8358

# Model parameters (spec section 4.2).
POOL_HALF_WIDTH_DAYS = 10
PERCENTILE = 5.0
N_HARMONICS = 2
REFERENCE_YEAR = 2024  # leap, so all 366 day-of-year slots exist
```

Run: `uv sync`
Expected: a `.venv` is created and dependencies resolve.

- [ ] **Step 2: Write the failing loader tests**

Create `tests/conftest.py`:

```python
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def loaded(data_dir):
    from pvnight.loader import load

    return load(data_dir)
```

Create `tests/test_loader.py`:

```python
import datetime as dt

import pandas as pd
import pytest

from pvnight.loader import generating_flag, load


def test_utc_index_is_monotonic_and_unique(loaded):
    """The single assertion that catches the entire DST bug class."""
    ts = loaded["ts_utc"]
    assert ts.is_monotonic_increasing
    assert not ts.duplicated().any()


def test_spring_forward_day_has_23_hours_of_samples(loaded):
    """2023-03-26 is a genuine 23-hour local day: 276 five-minute slots."""
    day = loaded[loaded["solar_date"] == dt.date(2023, 3, 26)]
    assert len(day) == 276


def test_fall_back_day_has_288_slots_and_no_duplicate_instants(loaded):
    """Fall-back days hold 288 rows, not the 300 a true 25-hour day would
    need. One hour is unrecoverable (spec fact 5); what matters is that the
    rows we do have map to distinct UTC instants."""
    day = loaded[loaded["solar_date"] == dt.date(2023, 10, 29)]
    assert len(day) == 288
    assert not day["ts_utc"].duplicated().any()


def test_expected_columns_and_dtypes(loaded):
    assert str(loaded["ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert loaded["generating"].dtype == bool
    for col in ["power_gen_w", "power_avg_w", "energy_gen_wh",
                "power_cons_w", "energy_cons_wh"]:
        assert loaded[col].dtype == "float64"


def _frame(rows):
    """rows: list of (solar_date, power_gen_w, power_avg_w, energy_gen_wh)."""
    return pd.DataFrame(
        rows, columns=["solar_date", "power_gen_w", "power_avg_w", "energy_gen_wh"]
    )


def test_generating_flag_is_true_on_positive_power():
    df = _frame([(dt.date(2023, 6, 1), 0.0, 0.0, 0.0),
                 (dt.date(2023, 6, 1), 12.0, 0.0, 0.0)])
    assert generating_flag(df).tolist() == [False, True]


def test_generating_flag_is_true_on_positive_average_power():
    """Some years report Average Power when Instantaneous Power is absent."""
    df = _frame([(dt.date(2023, 6, 1), 0.0, 0.0, 0.0),
                 (dt.date(2023, 6, 1), 0.0, 9.0, 0.0)])
    assert generating_flag(df).tolist() == [False, True]


def test_generating_flag_detects_rising_cumulative_energy():
    """2020-2021 write nulls for power at night but still accumulate energy."""
    df = _frame([(dt.date(2023, 6, 1), 0.0, 0.0, 100.0),
                 (dt.date(2023, 6, 1), 0.0, 0.0, 140.0),
                 (dt.date(2023, 6, 1), 0.0, 0.0, 140.0)])
    assert generating_flag(df).tolist() == [False, True, False]


def test_generating_flag_does_not_leak_across_the_day_boundary():
    """Cumulative energy resets each day. Grouping by solar_date means the
    first sample of a new day is never compared against the previous day."""
    df = _frame([(dt.date(2023, 6, 1), 0.0, 0.0, 5000.0),
                 (dt.date(2023, 6, 2), 0.0, 0.0, 0.0),
                 (dt.date(2023, 6, 2), 0.0, 0.0, 30.0)])
    assert generating_flag(df).tolist() == [False, False, True]


def test_load_rejects_a_directory_with_no_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.loader'`

- [ ] **Step 4: Implement the loader**

Create `src/pvnight/loader.py`:

```python
"""Load PVOutput history exports into a tidy, UTC-indexed frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_GLOB, SITE_TZ


def generating_flag(df: pd.DataFrame) -> pd.Series:
    """True where the installation was producing.

    Three independent signals are OR-ed because the null-versus-zero
    convention changes across years (spec fact 2): explicit power in either
    power column, or an increase in the within-day cumulative energy counter.
    Grouping the diff by ``solar_date`` matters — the counter resets at
    midnight, so a global diff would compare across the day boundary.
    """
    power_on = (df["power_gen_w"] > 0) | (df["power_avg_w"] > 0)
    energy_rose = df.groupby("solar_date")["energy_gen_wh"].diff().fillna(0.0) > 0
    return (power_on | energy_rose).rename("generating")


def load(data_dir: Path) -> pd.DataFrame:
    """Read every yearly export in ``data_dir`` into one frame.

    Timestamps are localized once and immediately converted to UTC; every
    downstream computation works in UTC. ``solar_date`` keeps the local
    calendar date, which is the correct grouping key for a solar day.
    """
    paths = sorted(Path(data_dir).glob(DATA_GLOB))
    if not paths:
        raise FileNotFoundError(f"no files matching {DATA_GLOB!r} in {data_dir}")

    # These are plain Parquet despite the .xz suffix — do not use lzma.
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    naive = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"], format="%Y%m%d %H:%M"
    )
    local = naive.dt.tz_localize(
        SITE_TZ,
        nonexistent="shift_forward",  # spring-forward gap
        ambiguous=False,              # arbitrary; made inconsequential, spec 5.1
    )

    df = pd.DataFrame(
        {
            "ts_utc": local.dt.tz_convert("UTC"),
            "solar_date": pd.to_datetime(raw["Date"], format="%Y%m%d").dt.date,
            "power_gen_w": raw["Instantaneous Power"].astype("float64").fillna(0.0),
            "power_avg_w": raw["Average Power"].astype("float64").fillna(0.0),
            "energy_gen_wh": raw["Energy Generation"].astype("float64"),
            "power_cons_w": raw["Power Consumption"].astype("float64"),
            "energy_cons_wh": raw["Energy Consumption"].astype("float64"),
        }
    ).sort_values("ts_utc", kind="stable").reset_index(drop=True)

    df["generating"] = generating_flag(df)
    return df
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_loader.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/pvnight tests
git commit -m "Add project scaffolding and PVOutput loader

Localizes once and converts straight to UTC, keeping the local date only
as the solar-day grouping key. The generating flag ORs three signals
because the null-versus-zero convention changes across years."
```

---

### Task 2: Solar geometry

**Files:**
- Create: `src/pvnight/solar.py`
- Test: `tests/test_solar.py`

**Interfaces:**
- Consumes: `pvnight.config` constants from Task 1.
- Produces:
  - `solar.elevation(ts_utc: pd.DatetimeIndex) -> pd.Series` — geometric elevation in degrees.
  - `solar.azimuth(ts_utc: pd.DatetimeIndex) -> pd.Series` — degrees clockwise from north.
  - `solar.sun_rise_set(dates: Sequence[datetime.date]) -> pd.DataFrame` — indexed by date, columns `sunrise_utc`, `sunset_utc`, both tz-aware UTC.
  - `solar.day_grid_utc(local_date: datetime.date, freq: str = "1min") -> pd.DatetimeIndex` — UTC instants spanning that local day.
  - `solar.crossings(local_date, theta_start_deg, theta_end_deg) -> tuple[pd.Timestamp, pd.Timestamp]` — first and last minute at or above the thresholds; `pd.NaT` if never reached.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_solar.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd

from pvnight import solar
from pvnight.config import SUNRISE_ELEVATION_DEG


def test_day_grid_length_reflects_true_local_day_length():
    """Built in UTC, the grid spans the real day: 23h, 24h or 25h."""
    assert len(solar.day_grid_utc(dt.date(2023, 3, 26))) == 1380
    assert len(solar.day_grid_utc(dt.date(2023, 6, 15))) == 1440
    assert len(solar.day_grid_utc(dt.date(2023, 10, 29))) == 1500


def test_day_grid_is_utc_and_strictly_increasing():
    grid = solar.day_grid_utc(dt.date(2023, 10, 29))
    assert str(grid.tz) == "UTC"
    assert grid.is_monotonic_increasing
    assert not grid.duplicated().any()


def test_sun_rise_set_returns_the_requested_day_not_the_previous_one():
    """Regression guard for spec fact 10: pvlib given local midnight returns
    the previous day's sunrise. sun_rise_set must pass local noon."""
    dates = [dt.date(2023, 6, 21), dt.date(2023, 12, 21)]
    rs = solar.sun_rise_set(dates)
    for d in dates:
        assert rs.loc[d, "sunrise_utc"].tz_convert("Europe/Amsterdam").date() == d
        assert rs.loc[d, "sunset_utc"].tz_convert("Europe/Amsterdam").date() == d


def test_sunrise_is_before_sunset():
    rs = solar.sun_rise_set([dt.date(2023, 1, 15), dt.date(2023, 7, 15)])
    assert (rs["sunrise_utc"] < rs["sunset_utc"]).all()


def test_crossings_agree_with_pvlib_sunrise_sunset():
    """Cross-validation: the elevation-crossing solver and pvlib's SPA
    rise/set routine are independent code paths and must give one answer.

    The grid has one-minute resolution, so the first minute at or above the
    threshold falls in [sunrise, sunrise + 60s), and symmetrically at dusk.
    """
    dates = [dt.date(2023, 1, 1) + dt.timedelta(days=n) for n in range(0, 365, 7)]
    dates += [dt.date(2023, 3, 26), dt.date(2023, 10, 29),
              dt.date(2023, 6, 21), dt.date(2023, 12, 21)]
    rs = solar.sun_rise_set(dates)

    for d in dates:
        start, end = solar.crossings(d, SUNRISE_ELEVATION_DEG, SUNRISE_ELEVATION_DEG)
        assert start is not pd.NaT and end is not pd.NaT
        lead = (start - rs.loc[d, "sunrise_utc"]).total_seconds()
        lag = (rs.loc[d, "sunset_utc"] - end).total_seconds()
        assert 0 <= lead < 60, f"{d}: start {lead}s after sunrise"
        assert 0 <= lag < 60, f"{d}: end {lag}s before sunset"


def test_crossings_return_nat_when_threshold_is_never_reached():
    start, end = solar.crossings(dt.date(2023, 12, 21), 80.0, 80.0)
    assert start is pd.NaT and end is pd.NaT


def test_elevation_is_geometric_not_apparent():
    """Below the horizon pvlib applies no refraction, so the two agree; above
    it they diverge. This pins that we return the geometric column."""
    import pvlib

    from pvnight.config import LATITUDE, LONGITUDE

    idx = pd.DatetimeIndex(["2023-06-21T04:00", "2023-06-21T10:00"], tz="UTC")
    sp = pvlib.solarposition.get_solarposition(idx, LATITUDE, LONGITUDE)
    got = solar.elevation(idx)
    assert np.allclose(got.to_numpy(), sp["elevation"].to_numpy())
    assert not np.allclose(got.to_numpy(), sp["apparent_elevation"].to_numpy())


def test_elevation_peaks_around_solar_noon():
    grid = solar.day_grid_utc(dt.date(2023, 6, 21))
    el = solar.elevation(grid).to_numpy()
    peak = grid[int(np.argmax(el))].tz_convert("Europe/Amsterdam")
    assert 13 <= peak.hour <= 14
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_solar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.solar'`

- [ ] **Step 3: Implement the solar module**

Create `src/pvnight/solar.py`:

```python
"""All contact with pvlib lives here. Geometric elevation throughout."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pvlib

from .config import LATITUDE, LONGITUDE, SITE_TZ


def _position(ts_utc: pd.DatetimeIndex) -> pd.DataFrame:
    return pvlib.solarposition.get_solarposition(ts_utc, LATITUDE, LONGITUDE)


def elevation(ts_utc: pd.DatetimeIndex) -> pd.Series:
    """Geometric solar elevation in degrees.

    Deliberately not ``apparent_elevation``: pvlib applies no refraction
    correction below the horizon, which puts a kink in the apparent curve at
    0 degrees, and a third of our events sit below it (spec fact 11).
    """
    return _position(pd.DatetimeIndex(ts_utc))["elevation"]


def azimuth(ts_utc: pd.DatetimeIndex) -> pd.Series:
    """Solar azimuth in degrees clockwise from north."""
    return _position(pd.DatetimeIndex(ts_utc))["azimuth"]


def sun_rise_set(dates: Sequence[dt.date]) -> pd.DataFrame:
    """Sunrise and sunset in UTC for each local date.

    pvlib is given local *noon*. Given local midnight it converts to UTC
    first, lands on the previous UTC day, and returns the previous day's
    sunrise — a silent one-day shift (spec fact 10).
    """
    dates = list(dates)
    noon = pd.DatetimeIndex(
        [pd.Timestamp(d) + pd.Timedelta(hours=12) for d in dates]
    ).tz_localize(SITE_TZ)
    rst = pvlib.solarposition.sun_rise_set_transit_spa(noon, LATITUDE, LONGITUDE)
    return pd.DataFrame(
        {
            "sunrise_utc": pd.DatetimeIndex(rst["sunrise"]).tz_convert("UTC"),
            "sunset_utc": pd.DatetimeIndex(rst["sunset"]).tz_convert("UTC"),
        },
        index=pd.Index(dates, name="date"),
    )


def day_grid_utc(local_date: dt.date, freq: str = "1min") -> pd.DatetimeIndex:
    """UTC instants spanning one local day.

    Built in UTC so DST transitions simply make the day shorter or longer
    (1380 / 1440 / 1500 minutes) instead of producing a gap or duplicates.
    """
    start = pd.Timestamp(local_date).tz_localize(SITE_TZ)
    end = (pd.Timestamp(local_date) + pd.Timedelta(days=1)).tz_localize(SITE_TZ)
    return pd.date_range(
        start.tz_convert("UTC"), end.tz_convert("UTC"),
        freq=freq, inclusive="left", tz="UTC",
    )


def crossings(
    local_date: dt.date, theta_start_deg: float, theta_end_deg: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last minute of the local day at or above the thresholds.

    Elevation rises to a single midday peak and falls, so the first and last
    samples above a threshold are exactly the two crossings.
    """
    grid = day_grid_utc(local_date)
    el = elevation(grid).to_numpy()

    above_start = el >= theta_start_deg
    start = grid[int(np.argmax(above_start))] if above_start.any() else pd.NaT

    above_end = el >= theta_end_deg
    end = (
        grid[len(grid) - 1 - int(np.argmax(above_end[::-1]))]
        if above_end.any()
        else pd.NaT
    )
    return start, end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_solar.py -v`
Expected: PASS, 8 tests. The cross-validation test evaluates ~56 days × 1440 minutes and may take a few seconds.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/solar.py tests/test_solar.py
git commit -m "Add solar geometry module

Cross-validates the elevation-crossing solver against pvlib's independent
SPA rise/set routine. Guards the local-noon convention: given local
midnight pvlib returns the previous day's sunrise."
```

---

### Task 3: Per-day first and last light

**Files:**
- Create: `src/pvnight/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: the frame from `loader.load`.
- Produces: `events.first_last_light(df: pd.DataFrame) -> pd.DataFrame` — one row per solar day that generated, columns `solar_date` (`datetime.date`), `first_light_utc`, `last_light_utc` (tz-aware UTC), `n_generating` (int), `daily_yield_wh` (float).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events.py`:

```python
import datetime as dt

import pandas as pd

from pvnight.events import first_last_light


def _frame(rows):
    """rows: (ts_utc iso, solar_date, generating, energy_gen_wh)."""
    return pd.DataFrame(
        [
            {
                "ts_utc": pd.Timestamp(ts, tz="UTC"),
                "solar_date": d,
                "generating": g,
                "energy_gen_wh": e,
            }
            for ts, d, g, e in rows
        ]
    )


def test_picks_first_and_last_generating_sample():
    df = _frame([
        ("2023-06-01T03:00", dt.date(2023, 6, 1), False, 0.0),
        ("2023-06-01T04:00", dt.date(2023, 6, 1), True, 10.0),
        ("2023-06-01T12:00", dt.date(2023, 6, 1), True, 900.0),
        ("2023-06-01T20:00", dt.date(2023, 6, 1), False, 900.0),
    ])
    out = first_last_light(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["first_light_utc"] == pd.Timestamp("2023-06-01T04:00", tz="UTC")
    assert row["last_light_utc"] == pd.Timestamp("2023-06-01T12:00", tz="UTC")
    assert row["n_generating"] == 2
    assert row["daily_yield_wh"] == 900.0


def test_days_with_no_generation_are_absent():
    df = _frame([
        ("2023-01-01T10:00", dt.date(2023, 1, 1), False, 0.0),
        ("2023-01-02T10:00", dt.date(2023, 1, 2), True, 50.0),
    ])
    out = first_last_light(df)
    assert out["solar_date"].tolist() == [dt.date(2023, 1, 2)]


def test_real_data_matches_the_established_count(loaded):
    """Spec fact 4: 2,047 of 2,052 days recorded generation."""
    out = first_last_light(loaded)
    assert len(out) == 2047


def test_first_light_never_after_last_light(loaded):
    out = first_last_light(loaded)
    assert (out["first_light_utc"] <= out["last_light_utc"]).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.events'`

- [ ] **Step 3: Implement events**

Create `src/pvnight/events.py`:

```python
"""Reduce the sample-level frame to one first/last-light pair per solar day.

Timestamps only. Attaching solar geometry is the envelope module's job.
"""

from __future__ import annotations

import pandas as pd


def first_last_light(df: pd.DataFrame) -> pd.DataFrame:
    """One row per solar day on which the installation produced anything."""
    gen = df[df["generating"]]
    out = gen.groupby("solar_date").agg(
        first_light_utc=("ts_utc", "min"),
        last_light_utc=("ts_utc", "max"),
        n_generating=("ts_utc", "size"),
        daily_yield_wh=("energy_gen_wh", "max"),
    )
    return out.reset_index()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_events.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/events.py tests/test_events.py
git commit -m "Add per-day first/last light extraction"
```

---

### Task 4: Envelope fitting

**Files:**
- Create: `src/pvnight/envelope.py`
- Test: `tests/test_envelope.py`

**Interfaces:**
- Consumes: `events.first_last_light` output; `solar.elevation`; config constants.
- Produces:
  - `envelope.year_angle(ts: pd.DatetimeIndex) -> np.ndarray` — position in the year as radians in [0, 2π), leap-safe.
  - `envelope.attach_solar(events: pd.DataFrame) -> pd.DataFrame` — adds `el_first`, `el_last`, `phi_first`, `phi_last`.
  - `envelope.pooled_percentile(phi_events, values, phi_targets, half_width_days, q) -> tuple[np.ndarray, np.ndarray]` — returns percentile per target and the pooled sample count.
  - `envelope.pooled_year_count(phi_events, years, phi_targets, half_width_days) -> np.ndarray` — distinct calendar years contributing to each target.
  - `envelope.fit_fourier(phi, values, weights, n_harmonics) -> np.ndarray` — coefficient vector of length `2*n_harmonics + 1`.
  - `envelope.eval_fourier(coef, phi, n_harmonics) -> np.ndarray`
  - `envelope.Envelope` — dataclass with fields `coef_start`, `coef_end`, `n_harmonics`, `grid_phi`, `raw_start`, `raw_end`, `n_start`, `n_end`, `n_years_start`, `n_years_end`, and methods `theta_start(phi)`, `theta_end(phi)`, `samples_near(phi)`, `years_near(phi)`.
  - `envelope.fit(events_with_solar, half_width_days=..., q=..., n_harmonics=...) -> Envelope`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_envelope.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pvnight import envelope as env


def test_year_angle_is_zero_at_new_year_and_wraps_toward_two_pi():
    idx = pd.DatetimeIndex(
        ["2023-01-01T00:00", "2023-12-31T23:59"], tz="UTC"
    )
    phi = env.year_angle(idx)
    assert phi[0] == pytest.approx(0.0, abs=1e-9)
    assert phi[1] == pytest.approx(2 * np.pi, rel=1e-3)


def test_year_angle_is_leap_safe():
    """Mid-year in a leap year and a common year land at the same angle."""
    common = env.year_angle(pd.DatetimeIndex(["2023-07-02T12:00"], tz="UTC"))[0]
    leap = env.year_angle(pd.DatetimeIndex(["2024-07-02T00:00"], tz="UTC"))[0]
    assert abs(common - leap) < 0.02


def test_pooled_percentile_recovers_an_injected_value():
    phi_events = np.full(200, 1.0)
    values = np.linspace(0.0, 100.0, 200)
    out, counts = env.pooled_percentile(
        phi_events, values, np.array([1.0]), half_width_days=10, q=5.0
    )
    assert counts[0] == 200
    assert out[0] == pytest.approx(np.percentile(values, 5.0))


def test_pooled_percentile_window_wraps_the_year_boundary():
    """Events in late December must be pooled with a 1 January target."""
    late_december = 2 * np.pi * np.array([360.0, 362.0, 364.0]) / 365.25
    values = np.array([-1.0, -2.0, -3.0])
    out, counts = env.pooled_percentile(
        late_december, values, np.array([0.0]), half_width_days=10, q=50.0
    )
    assert counts[0] == 3
    assert out[0] == pytest.approx(-2.0)


def test_pooled_percentile_excludes_events_outside_the_window():
    phi = 2 * np.pi * np.array([0.0, 100.0]) / 365.25
    out, counts = env.pooled_percentile(
        phi, np.array([5.0, 99.0]), np.array([0.0]), half_width_days=10, q=50.0
    )
    assert counts[0] == 1
    assert out[0] == pytest.approx(5.0)


def test_fourier_fit_recovers_a_known_harmonic_exactly():
    phi = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    truth = 1.5 + 0.7 * np.cos(phi) - 0.3 * np.sin(phi)
    coef = env.fit_fourier(phi, truth, np.ones_like(phi), n_harmonics=2)
    assert np.allclose(env.eval_fourier(coef, phi, 2), truth, atol=1e-8)


def test_fourier_fit_is_periodic():
    phi = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    values = np.cos(phi) + 0.2 * np.sin(2 * phi)
    coef = env.fit_fourier(phi, values, np.ones_like(phi), n_harmonics=2)
    at_zero = env.eval_fourier(coef, np.array([0.0]), 2)
    at_two_pi = env.eval_fourier(coef, np.array([2 * np.pi]), 2)
    assert at_zero == pytest.approx(at_two_pi)


def test_fourier_weights_pull_the_fit_toward_heavily_weighted_points():
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    values = np.zeros_like(phi)
    values[0] = 10.0

    even = env.fit_fourier(phi, values, np.ones_like(phi), n_harmonics=2)
    weights = np.ones_like(phi)
    weights[0] = 500.0
    heavy = env.fit_fourier(phi, values, weights, n_harmonics=2)

    at_zero_even = env.eval_fourier(even, np.array([0.0]), 2)[0]
    at_zero_heavy = env.eval_fourier(heavy, np.array([0.0]), 2)[0]
    assert at_zero_heavy > at_zero_even


@pytest.fixture(scope="session")
def fitted(loaded):
    from pvnight.events import first_last_light

    ev = env.attach_solar(first_last_light(loaded))
    return env.fit(ev), ev


def test_attach_solar_reproduces_the_spec_seasonal_range(fitted):
    """Spec fact 8: December first-light threshold near -0.91 deg, April
    near -1.67. Tolerant bounds, but they would catch a switch to apparent
    elevation or a broken timestamp."""
    _, ev = fitted
    month = pd.to_datetime(ev["solar_date"]).dt.month
    dec = ev.loc[month == 12, "el_first"].quantile(0.05)
    apr = ev.loc[month == 4, "el_first"].quantile(0.05)
    assert -1.2 < dec < -0.6
    assert -1.9 < apr < -1.4
    assert dec > apr


def test_fitted_thresholds_are_finite_everywhere(fitted):
    model, _ = fitted
    phi = 2 * np.pi * np.arange(366) / 366
    assert np.isfinite(model.theta_start(phi)).all()
    assert np.isfinite(model.theta_end(phi)).all()


def test_fitted_thresholds_stay_near_the_horizon(fitted):
    """A physically sane envelope sits within a couple of degrees of 0."""
    model, _ = fitted
    phi = 2 * np.pi * np.arange(366) / 366
    assert np.abs(model.theta_start(phi)).max() < 3.0
    assert np.abs(model.theta_end(phi)).max() < 3.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_envelope.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.envelope'`

- [ ] **Step 3: Implement the envelope model**

Create `src/pvnight/envelope.py`:

```python
"""Fit the seasonal elevation threshold at which generation starts and stops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import solar
from .config import N_HARMONICS, PERCENTILE, POOL_HALF_WIDTH_DAYS

DAYS_PER_YEAR = 365.25


def year_angle(ts: pd.DatetimeIndex) -> np.ndarray:
    """Position within the calendar year, in radians.

    Uses the fraction of the actual year elapsed rather than an integer day
    number, so leap years need no special case for 29 February.
    """
    ts = pd.DatetimeIndex(ts)
    year_start = pd.to_datetime(
        pd.Index(ts.year).astype(str) + "-01-01", utc=True
    )
    next_year = pd.to_datetime(
        pd.Index(ts.year + 1).astype(str) + "-01-01", utc=True
    )
    frac = (ts - year_start) / (next_year - year_start)
    return 2 * np.pi * np.asarray(frac, dtype=float)


def attach_solar(events: pd.DataFrame) -> pd.DataFrame:
    """Add solar elevation and year angle for both ends of each day."""
    ev = events.dropna(subset=["first_light_utc", "last_light_utc"]).copy()
    first = pd.DatetimeIndex(ev["first_light_utc"])
    last = pd.DatetimeIndex(ev["last_light_utc"])
    ev["el_first"] = solar.elevation(first).to_numpy()
    ev["el_last"] = solar.elevation(last).to_numpy()
    ev["phi_first"] = year_angle(first)
    ev["phi_last"] = year_angle(last)
    return ev


def pooled_percentile(
    phi_events: np.ndarray,
    values: np.ndarray,
    phi_targets: np.ndarray,
    half_width_days: int = POOL_HALF_WIDTH_DAYS,
    q: float = PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile of ``values`` pooled over a circular window in the year.

    The window wraps the year boundary, so 1 January pools with late
    December. Circular distance is taken via the complex argument, which
    handles the wrap without any modular arithmetic of our own.
    """
    half = 2 * np.pi * half_width_days / DAYS_PER_YEAR
    out = np.full(len(phi_targets), np.nan)
    counts = np.zeros(len(phi_targets), dtype=int)

    for i, target in enumerate(phi_targets):
        distance = np.abs(np.angle(np.exp(1j * (phi_events - target))))
        selected = distance <= half
        counts[i] = int(selected.sum())
        if counts[i]:
            out[i] = np.percentile(values[selected], q)
    return out, counts


def pooled_year_count(
    phi_events: np.ndarray,
    years: np.ndarray,
    phi_targets: np.ndarray,
    half_width_days: int = POOL_HALF_WIDTH_DAYS,
) -> np.ndarray:
    """Distinct calendar years contributing to each target.

    Spec section 7 requires this alongside the raw sample count: 120 samples
    drawn from one year is a very different claim from 120 drawn from six.
    """
    half = 2 * np.pi * half_width_days / DAYS_PER_YEAR
    out = np.zeros(len(phi_targets), dtype=int)
    for i, target in enumerate(phi_targets):
        distance = np.abs(np.angle(np.exp(1j * (phi_events - target))))
        out[i] = len(np.unique(years[distance <= half]))
    return out


def _design(phi: np.ndarray, n_harmonics: int) -> np.ndarray:
    columns = [np.ones_like(phi)]
    for k in range(1, n_harmonics + 1):
        columns.append(np.cos(k * phi))
        columns.append(np.sin(k * phi))
    return np.column_stack(columns)


def fit_fourier(
    phi: np.ndarray, values: np.ndarray, weights: np.ndarray, n_harmonics: int
) -> np.ndarray:
    """Weighted least-squares truncated Fourier fit. Periodic by construction."""
    finite = np.isfinite(values) & (weights > 0)
    design = _design(phi[finite], n_harmonics)
    root_w = np.sqrt(weights[finite])
    coef, *_ = np.linalg.lstsq(
        design * root_w[:, None], values[finite] * root_w, rcond=None
    )
    return coef


def eval_fourier(coef: np.ndarray, phi: np.ndarray, n_harmonics: int) -> np.ndarray:
    return _design(np.asarray(phi, dtype=float), n_harmonics) @ coef


@dataclass
class Envelope:
    """The fitted seasonal thresholds, plus the raw values behind them."""

    coef_start: np.ndarray
    coef_end: np.ndarray
    n_harmonics: int
    grid_phi: np.ndarray
    raw_start: np.ndarray
    raw_end: np.ndarray
    n_start: np.ndarray
    n_end: np.ndarray
    n_years_start: np.ndarray
    n_years_end: np.ndarray

    def theta_start(self, phi) -> np.ndarray:
        return eval_fourier(self.coef_start, np.atleast_1d(phi), self.n_harmonics)

    def theta_end(self, phi) -> np.ndarray:
        return eval_fourier(self.coef_end, np.atleast_1d(phi), self.n_harmonics)

    def _nearest(self, phi) -> int:
        return int(np.argmin(np.abs(np.angle(np.exp(1j * (self.grid_phi - phi))))))

    def samples_near(self, phi) -> tuple[int, int]:
        """Pooled sample counts at the nearest grid point to ``phi``."""
        i = self._nearest(phi)
        return int(self.n_start[i]), int(self.n_end[i])

    def years_near(self, phi) -> tuple[int, int]:
        """Distinct contributing years at the nearest grid point to ``phi``."""
        i = self._nearest(phi)
        return int(self.n_years_start[i]), int(self.n_years_end[i])


def fit(
    events_with_solar: pd.DataFrame,
    half_width_days: int = POOL_HALF_WIDTH_DAYS,
    q: float = PERCENTILE,
    n_harmonics: int = N_HARMONICS,
) -> Envelope:
    """Fit theta_start and theta_end over a 366-point year grid."""
    grid = 2 * np.pi * np.arange(366) / 366

    raw_start, n_start = pooled_percentile(
        events_with_solar["phi_first"].to_numpy(),
        events_with_solar["el_first"].to_numpy(),
        grid, half_width_days, q,
    )
    raw_end, n_end = pooled_percentile(
        events_with_solar["phi_last"].to_numpy(),
        events_with_solar["el_last"].to_numpy(),
        grid, half_width_days, q,
    )

    years = pd.to_datetime(events_with_solar["solar_date"]).dt.year.to_numpy()
    n_years_start = pooled_year_count(
        events_with_solar["phi_first"].to_numpy(), years, grid, half_width_days
    )
    n_years_end = pooled_year_count(
        events_with_solar["phi_last"].to_numpy(), years, grid, half_width_days
    )

    return Envelope(
        coef_start=fit_fourier(grid, raw_start, n_start.astype(float), n_harmonics),
        coef_end=fit_fourier(grid, raw_end, n_end.astype(float), n_harmonics),
        n_harmonics=n_harmonics,
        grid_phi=grid,
        raw_start=raw_start,
        raw_end=raw_end,
        n_start=n_start,
        n_end=n_end,
        n_years_start=n_years_start,
        n_years_end=n_years_end,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_envelope.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/envelope.py tests/test_envelope.py
git commit -m "Add seasonal elevation-threshold envelope fit

Pools first/last-light elevations across years in a circular window,
takes the 5th percentile, and smooths with a 2-harmonic Fourier fit that
is periodic by construction."
```

---

### Task 5: Window table and the containment regression

**Files:**
- Modify: `src/pvnight/envelope.py` (append `build_windows` and `thresholds_table`)
- Test: `tests/test_windows.py`

**Interfaces:**
- Consumes: `Envelope` from Task 4, `solar.crossings`, `solar.sun_rise_set`, `solar.day_grid_utc`.
- Produces:
  - `envelope.thresholds_table(model: Envelope) -> pd.DataFrame` — 366 rows, columns `doy`, `theta_start_deg`, `theta_end_deg`, `theta_start_raw_deg`, `theta_end_raw_deg`, `n_samples_start`, `n_samples_end`, `n_years`.
  - `envelope.build_windows(model, start_date, end_date, observed_dates) -> pd.DataFrame` — one row per date, columns per spec §7.2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_windows.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pvnight import envelope as env
from pvnight.events import first_last_light


@pytest.fixture(scope="session")
def pipeline(loaded):
    ev = env.attach_solar(first_last_light(loaded))
    model = env.fit(ev)
    observed = set(ev["solar_date"])
    windows = env.build_windows(
        model, dt.date(2020, 1, 1), dt.date(2026, 12, 31), observed
    )
    return model, ev, windows


def test_thresholds_table_has_366_rows(pipeline):
    model, _, _ = pipeline
    table = env.thresholds_table(model)
    assert len(table) == 366
    assert table["doy"].tolist() == list(range(1, 367))
    assert table["theta_start_deg"].notna().all()


def test_windows_cover_every_date_once(pipeline):
    _, _, w = pipeline
    expected = pd.date_range("2020-01-01", "2026-12-31", freq="D")
    assert len(w) == len(expected)
    assert not w["date"].duplicated().any()


def test_window_opens_before_it_closes(pipeline):
    _, _, w = pipeline
    assert (w["solar_start_utc"] < w["solar_end_utc"]).all()


def test_local_columns_carry_an_explicit_offset(pipeline):
    _, _, w = pipeline
    sample = w["solar_start_local"].iloc[180]
    assert sample.endswith("+02:00") or sample.endswith("+01:00")


def test_night_links_one_day_to_the_next(pipeline):
    _, _, w = pipeline
    assert (w["night_start_utc"][:-1].to_numpy()
            == w["solar_end_utc"][:-1].to_numpy()).all()
    assert (w["night_end_utc"][:-1].to_numpy()
            == w["solar_start_utc"][1:].to_numpy()).all()
    assert pd.isna(w["night_end_utc"].iloc[-1])


def test_night_duration_reflects_real_elapsed_time_across_dst(pipeline):
    """The spring-forward night is an hour shorter than its neighbours and
    the fall-back night an hour longer. Naive local arithmetic would report
    all three as equal."""
    _, _, w = pipeline
    w = w.set_index("date")
    spring = w.loc[dt.date(2023, 3, 25), "night_duration_h"]
    before_spring = w.loc[dt.date(2023, 3, 24), "night_duration_h"]
    autumn = w.loc[dt.date(2023, 10, 28), "night_duration_h"]
    before_autumn = w.loc[dt.date(2023, 10, 27), "night_duration_h"]
    assert spring == pytest.approx(before_spring - 1.0, abs=0.15)
    assert autumn == pytest.approx(before_autumn + 1.0, abs=0.15)


def test_dst_hour_missing_flags_exactly_the_fall_back_dates(pipeline):
    _, _, w = pipeline
    flagged = set(w.loc[w["dst_hour_missing"], "date"])
    assert dt.date(2023, 10, 29) in flagged
    assert dt.date(2024, 10, 27) in flagged
    assert dt.date(2023, 3, 26) not in flagged
    assert len(flagged) == 7  # one per year, 2020-2026


def test_extrapolated_marks_dates_with_no_observation(pipeline):
    _, _, w = pipeline
    w = w.set_index("date")
    assert w.loc[dt.date(2020, 1, 15), "extrapolated"]
    assert w.loc[dt.date(2026, 6, 1), "extrapolated"]
    assert not w.loc[dt.date(2023, 6, 1), "extrapolated"]


def test_window_contains_observed_first_light_on_95_percent_of_days(pipeline):
    """The core regression guard. A 5th-percentile threshold implies about
    95% containment by construction; a materially lower figure means the fit
    has drifted."""
    _, ev, w = pipeline
    m = w.merge(ev, left_on="date", right_on="solar_date")
    assert (m["first_light_utc"] >= m["solar_start_utc"]).mean() >= 0.95


def test_window_contains_observed_last_light_on_95_percent_of_days(pipeline):
    _, ev, w = pipeline
    m = w.merge(ev, left_on="date", right_on="solar_date")
    assert (m["last_light_utc"] <= m["solar_end_utc"]).mean() >= 0.95


def test_days_outside_the_window_miss_it_only_narrowly(pipeline):
    """Where generation does fall outside, it should be minutes, not hours."""
    _, ev, w = pipeline
    m = w.merge(ev, left_on="date", right_on="solar_date")
    early = m.loc[m["first_light_utc"] < m["solar_start_utc"]]
    violation_min = (
        (early["solar_start_utc"] - early["first_light_utc"])
        .dt.total_seconds() / 60
    )
    assert violation_min.median() < 10.0


def test_n_years_reflects_thin_early_coverage(pipeline):
    """2020 contributes nothing before 20 May, so a January target draws on
    five years while a July target draws on six (spec section 8)."""
    _, _, w = pipeline
    w = w.set_index("date")
    assert w.loc[dt.date(2023, 1, 15), "n_years"] == 5
    assert w.loc[dt.date(2023, 7, 15), "n_years"] == 6


def test_no_window_is_undefined_at_this_latitude(pipeline):
    _, _, w = pipeline
    assert not w["window_undefined"].any()


def test_start_offset_stays_in_a_physically_sane_band(pipeline):
    """Spec section 9 test 7: within -30 to +60 minutes of sunrise."""
    _, _, w = pipeline
    assert w["start_offset_min"].between(-30, 60).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_windows.py -v`
Expected: FAIL — `AttributeError: module 'pvnight.envelope' has no attribute 'build_windows'`

- [ ] **Step 3: Append the table builders to `envelope.py`**

Add to the end of `src/pvnight/envelope.py`:

```python
def thresholds_table(model: Envelope) -> pd.DataFrame:
    """The model itself, one row per day of a leap reference year."""
    return pd.DataFrame(
        {
            "doy": np.arange(1, 367),
            "theta_start_deg": model.theta_start(model.grid_phi),
            "theta_end_deg": model.theta_end(model.grid_phi),
            "theta_start_raw_deg": model.raw_start,
            "theta_end_raw_deg": model.raw_end,
            "n_samples_start": model.n_start,
            "n_samples_end": model.n_end,
            "n_years": np.minimum(model.n_years_start, model.n_years_end),
        }
    )


def _local_iso(ts: pd.Timestamp) -> str | None:
    """ISO-8601 in site-local time with an explicit UTC offset."""
    if pd.isna(ts):
        return None
    return ts.tz_convert(SITE_TZ).isoformat()


def build_windows(
    model: Envelope,
    start_date: dt.date,
    end_date: dt.date,
    observed_dates: set[dt.date],
) -> pd.DataFrame:
    """Apply the fitted model to every date in the range."""
    dates = [d.date() for d in pd.date_range(start_date, end_date, freq="D")]
    rise_set = solar.sun_rise_set(dates)

    rows = []
    for d in dates:
        phi = year_angle(
            pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=12)])
        )[0]
        theta_start = float(model.theta_start(phi)[0])
        theta_end = float(model.theta_end(phi)[0])
        start, end = solar.crossings(d, theta_start, theta_end)
        n_start, n_end = model.samples_near(phi)
        years_start, years_end = model.years_near(phi)

        # A 25-hour local day is a fall-back day, so an hour is missing.
        grid_len = len(solar.day_grid_utc(d))

        rows.append(
            {
                "date": d,
                "solar_start_utc": start,
                "solar_end_utc": end,
                "solar_start_local": _local_iso(start),
                "solar_end_local": _local_iso(end),
                "sunrise_utc": rise_set.loc[d, "sunrise_utc"],
                "sunset_utc": rise_set.loc[d, "sunset_utc"],
                "theta_start_deg": theta_start,
                "theta_end_deg": theta_end,
                "n_samples": min(n_start, n_end),
                "n_years": min(years_start, years_end),
                "dst_hour_missing": grid_len == 1500,
                "extrapolated": d not in observed_dates,
                # Spec section 8: not expected at this latitude, but flagged
                # rather than assumed away.
                "window_undefined": pd.isna(start) or pd.isna(end),
            }
        )

    w = pd.DataFrame(rows)
    w["start_offset_min"] = (
        w["solar_start_utc"] - w["sunrise_utc"]
    ).dt.total_seconds() / 60
    w["end_offset_min"] = (
        w["solar_end_utc"] - w["sunset_utc"]
    ).dt.total_seconds() / 60
    w["night_start_utc"] = w["solar_end_utc"]
    w["night_end_utc"] = w["solar_start_utc"].shift(-1)
    w["night_duration_h"] = (
        w["night_end_utc"] - w["night_start_utc"]
    ).dt.total_seconds() / 3600
    return w
```

Add the imports this needs at the top of `envelope.py`:

```python
import datetime as dt
from .config import SITE_TZ
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_windows.py -v`
Expected: PASS, 14 tests. Building 2,557 daily windows evaluates ~3.7M grid points and takes on the order of a minute.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS, 48 tests.

- [ ] **Step 6: Commit**

```bash
git add src/pvnight/envelope.py tests/test_windows.py
git commit -m "Add window table with night intervals and containment guards

Night duration is a UTC subtraction, so the spring-forward night comes
out an hour shorter and the fall-back night an hour longer, which the
tests assert directly."
```

---

### Task 6: Charts and HTML report

**Files:**
- Create: `src/pvnight/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Envelope`, the events frame, and the windows frame.
- Produces: `report.build_html(model, events, windows, loaded) -> str` — a complete HTML fragment (no `<html>`/`<head>`/`<body>` wrapper, per the Artifact contract) with a `<title>`, a `<style>` block, and five inline SVG charts.

**Before writing any chart code, load the `dataviz` skill, then `artifact-design`.** They govern the palette, chart forms, and the theme-aware token structure. The chart list below is the spec's requirement; those skills decide how the charts look.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report.py`:

```python
import datetime as dt

import pytest

from pvnight import envelope as env
from pvnight.events import first_last_light
from pvnight.report import build_html


@pytest.fixture(scope="session")
def artifacts(loaded):
    ev = env.attach_solar(first_last_light(loaded))
    model = env.fit(ev)
    windows = env.build_windows(
        model, dt.date(2023, 1, 1), dt.date(2023, 12, 31), set(ev["solar_date"])
    )
    return model, ev, windows


def test_html_has_a_title_and_five_charts(artifacts, loaded):
    model, ev, w = artifacts
    html = build_html(model, ev, w, loaded)
    assert "<title>" in html
    assert html.count("<svg") == 5


def test_html_omits_the_document_wrapper(artifacts, loaded):
    """The Artifact host supplies doctype, html, head and body."""
    model, ev, w = artifacts
    html = build_html(model, ev, w, loaded)
    for tag in ["<!doctype", "<html", "<head>", "<body>"]:
        assert tag not in html.lower()


def test_html_defines_light_and_dark_palettes(artifacts, loaded):
    model, ev, w = artifacts
    html = build_html(model, ev, w, loaded)
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="dark"]' in html
    assert '[data-theme="light"]' in html


def test_azimuth_chart_carries_the_season_confounding_caveat(artifacts, loaded):
    """Spec fact 7 requires this chart be labelled a shading screen, not a
    horizon profile, so no reader mistakes the winter bins for obstruction."""
    model, ev, w = artifacts
    html = build_html(model, ev, w, loaded)
    assert "season" in html.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.report'`

- [ ] **Step 3: Implement the report**

Create `src/pvnight/report.py`. Structure it as one function per chart, each returning an SVG string, plus `build_html` that assembles them.

Use this helper so every chart renders identically and with a transparent background (the page's theme shows through):

```python
"""Charts and the HTML report page."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import envelope as env  # noqa: E402

# Neutral grey for axis furniture: legible on both light and dark grounds.
FURNITURE = "#8a8f98"


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
```

Add the shared palette and helpers below the `_svg` helper:

```python
# Placeholder categorical palette. The dataviz skill supplies the real one —
# swap these three values, keep the names.
SERIES = ["#3b82f6", "#f59e0b", "#10b981"]


def _reference_year(windows: pd.DataFrame) -> int:
    """The most completely represented year in the window table."""
    return int(pd.to_datetime(windows["date"]).dt.year.value_counts().idxmax())


def _local_hours(ts) -> np.ndarray:
    """Local hour of day as a float, for plotting against day of year."""
    local = pd.DatetimeIndex(ts).tz_convert(SITE_TZ)
    return local.hour + local.minute / 60 + local.second / 3600


def _doy(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(list(dates))).dt.dayofyear.to_numpy()
```

Import what these need at the top of the module:

```python
from . import solar
from .config import SITE_TZ, SUNRISE_ELEVATION_DEG
```

Now the five charts:

```python
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
    ax.set_title("Shading screen, not a horizon profile", fontsize=10)
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
        grid.to_numpy(), aspect="auto", origin="lower", cmap="viridis",
        interpolation="nearest",
        extent=[1, 366, grid.index.min() - 0.5, grid.index.max() + 0.5],
    )
    ax.set_yticks(list(grid.index))
    ax.set_yticklabels([str(y) for y in grid.index])
    ax.set_xlabel("day of year")
    return _svg(fig)
```

Finally `build_html`. The palette is defined on bare `:root`, redefined under
`prefers-color-scheme: dark` guarded by `:not([data-theme="light"])`, and again
under `[data-theme="dark"]` so an explicit toggle wins in both directions:

```python
STYLE = """
<style>
:root {
  --bg: #fbfbfa; --surface: #ffffff; --ink: #1f2328;
  --muted: #6b7280; --line: #e5e7eb;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16181d; --surface: #1e2128; --ink: #e8eaed;
    --muted: #9aa0a6; --line: #2f333b;
  }
}
:root[data-theme="dark"] {
  --bg: #16181d; --surface: #1e2128; --ink: #e8eaed;
  --muted: #9aa0a6; --line: #2f333b;
}
body { background: var(--bg); color: var(--ink);
       font: 15px/1.6 ui-sans-serif, system-ui, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
.lede { color: var(--muted); margin: 0 0 28px; }
.card { background: var(--surface); border: 1px solid var(--line);
        border-radius: 10px; padding: 18px; margin: 0 0 20px; }
.card h2 { font-size: 15px; margin: 0 0 6px; }
.card p { color: var(--muted); font-size: 13px; margin: 0 0 12px; }
.chart { overflow-x: auto; }
.stats { display: flex; flex-wrap: wrap; gap: 22px; margin: 0 0 28px; }
.stat b { display: block; font-size: 20px; }
.stat span { color: var(--muted); font-size: 12px; }
</style>
"""


def build_html(model, events: pd.DataFrame, windows: pd.DataFrame,
               loaded: pd.DataFrame) -> str:
    """Assemble the report. No document wrapper — the host supplies it."""
    grid_phi = model.grid_phi
    theta_start = model.theta_start(grid_phi)
    nights = windows["night_duration_h"].dropna()

    stats = [
        (f"{theta_start.min():.2f} to {theta_start.max():.2f}",
         "fitted start threshold (degrees)"),
        (f"{windows['start_offset_min'].median():.1f} min",
         "median start offset from sunrise"),
        (f"{nights.min():.1f} to {nights.max():.1f} h", "night length range"),
        (f"{len(events)}", "days with observed generation"),
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
         "A flat line means no obstruction. Read this only as a screen: the "
         "azimuth bins mix seasons, since morning azimuth 130 degrees is "
         "reached both at midwinter sunrise and in summer mid-morning, so "
         "this is not a horizon profile.",
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
        "<title>Solar Window</title>"
        + STYLE
        + '<main><h1>When the sun can reach the panels</h1>'
        + '<p class="lede">Six years of PVOutput history, reduced to a daily '
        + "window in which generation is possible. Everything outside it is "
        + "night.</p>"
        + f'<div class="stats">{stat_html}</div>'
        + card_html
        + "</main>"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pvnight/report.py tests/test_report.py
git commit -m "Add charts and HTML report

The azimuth chart is labelled a shading screen rather than a horizon
profile, since its bins mix seasons."
```

---

### Task 7: Entry point and publication

**Files:**
- Create: `analyze.py`, `README.md`
- Test: `tests/test_analyze.py`

**Interfaces:**
- Consumes: every module above.
- Produces: `analyze.run(data_dir: Path, out_dir: Path) -> dict` — writes `solar_thresholds.csv`, `solar_windows.csv` and `report.html` into `out_dir`, returning a summary dict with keys `n_days_observed`, `n_windows`, `median_start_offset_min`, `shortest_night_h`, `longest_night_h`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze.py`:

```python
import pandas as pd

from analyze import run


def test_run_writes_all_three_outputs(tmp_path, data_dir):
    summary = run(data_dir, tmp_path)

    thresholds = pd.read_csv(tmp_path / "solar_thresholds.csv")
    windows = pd.read_csv(tmp_path / "solar_windows.csv")
    assert len(thresholds) == 366
    assert len(windows) == len(pd.date_range("2020-01-01", "2026-12-31"))
    assert (tmp_path / "report.html").read_text().count("<svg") == 5

    assert summary["n_days_observed"] == 2047
    assert 0 < summary["shortest_night_h"] < summary["longest_night_h"] < 24


def test_windows_csv_timestamps_are_unambiguous(tmp_path, data_dir):
    run(data_dir, tmp_path)
    w = pd.read_csv(tmp_path / "solar_windows.csv")
    assert w["solar_start_local"].str.contains(r"\+0[12]:00").all()
    assert w["solar_start_utc"].str.endswith("+00:00").all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analyze'`

- [ ] **Step 3: Implement the entry point**

Create `analyze.py` at the repository root:

```python
"""Run the full pipeline: load history, fit the envelope, write the outputs."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from pvnight import envelope as env
from pvnight import report
from pvnight.events import first_last_light
from pvnight.loader import load

START_DATE = dt.date(2020, 1, 1)
END_DATE = dt.date(2026, 12, 31)


def run(data_dir: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load(Path(data_dir))
    events = env.attach_solar(first_last_light(samples))
    model = env.fit(events)
    windows = env.build_windows(
        model, START_DATE, END_DATE, set(events["solar_date"])
    )

    env.thresholds_table(model).to_csv(out_dir / "solar_thresholds.csv", index=False)
    windows.to_csv(out_dir / "solar_windows.csv", index=False)
    (out_dir / "report.html").write_text(
        report.build_html(model, events, windows, samples)
    )

    nights = windows["night_duration_h"].dropna()
    return {
        "n_days_observed": len(events),
        "n_windows": len(windows),
        "median_start_offset_min": float(windows["start_offset_min"].median()),
        "shortest_night_h": float(nights.min()),
        "longest_night_h": float(nights.max()),
    }


if __name__ == "__main__":
    summary = run(Path(__file__).parent, Path(__file__).parent / "out")
    for key, value in summary.items():
        print(f"{key}: {value}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Run the pipeline and the full suite**

Run: `uv run python analyze.py`
Expected: the summary prints, and `out/` holds the two CSVs and `report.html`.

Run: `uv run pytest`
Expected: PASS, all 54 tests.

- [ ] **Step 6: Write the README**

Create `README.md` covering: what the repository does in two sentences, how to run it (`uv run python analyze.py`), what the two CSVs contain, and a pointer to the spec for the reasoning. State plainly that night consumption statistics are not yet built and that `night_start_utc`/`night_end_utc` are the join keys for that work.

- [ ] **Step 7: Publish the report**

Read `out/report.html` in full, then publish it with the Artifact tool, passing a favicon and a one-sentence description. Hand the user the URL.

- [ ] **Step 8: Commit**

```bash
git add analyze.py README.md tests/test_analyze.py out/solar_thresholds.csv out/solar_windows.csv
git commit -m "Add pipeline entry point and generated outputs"
```

---

## Verification

Before reporting the work complete, invoke `superpowers:verification-before-completion`. Concretely: run `uv run pytest` and paste the real summary line; run `uv run python analyze.py` and quote the printed figures. Do not describe any number as confirmed unless it appears in command output you have actually seen.

## Notes for the implementer

- **Runtime.** `build_windows` over seven years evaluates roughly 3.7 million solar positions. A minute or two is expected. Do not "optimise" this by coarsening the grid below one minute without saying so — the crossing test depends on that resolution.
- **The five charts are a spec requirement**, not a suggestion. The report test asserts the count.
- **If a test fails in a way that suggests the model is wrong rather than the code**, stop and raise it rather than loosening the assertion. The containment and sanity-band tests exist precisely to catch that, and weakening them would defeat their purpose.
