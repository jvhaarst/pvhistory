# Tariff Economics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the battery analysis from energy to euros using the 2027 Greenchoice tariff, so the recommended capacity comes from a real exchange rate instead of an invented threshold or a geometric artifact.

**Architecture:** A new `tariff.py` parses the price file and maps every UTC interval to a price band via local wall-clock time. `battery.simulate` gains one additive `band_idx` keyword that accumulates import and export per band — the same pattern its existing `month_idx` uses, so no second dispatch loop is written. A new `economics.py` prices the resulting flows, derives break-even costs, and measures a perfect-foresight arbitrage ceiling. The report and entry point gain euro cards and keys.

**Tech Stack:** Python via `uv`, pandas 3.x, numpy, matplotlib. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-05-tariff-economics-design.md`

## Global Constraints

- Run everything through `uv`: `uv run pytest`, `uv run python analyze_meter.py`.
- UTC internally throughout. Local time (`Europe/Amsterdam`, `config.SITE_TZ`) appears **only** in `tariff.band_index`, and only via `tz_convert` — never arithmetic on naive local timestamps.
- Never assert an exact pandas datetime resolution; pandas 3.x defaults to microseconds.
- `battery.simulate` with `band_idx=None` must be **bit-identical** to today. Phases 2 and 3 results must not move.
- **A second dispatch loop must not be written.** Extend the existing one.
- Prices are exact decimals from the file: Normaal 0.30566 / 0.07049 / 0.08050; Dal 0.27939 / 0.05964 / 0.06965; SuperDal 0.18225 / 0.01951 / 0.02951.
- Summer is 1 April – 30 September; winter is 1 October – 31 March. Calendar dates, unrelated to DST.
- Bands must tile all 24 hours in each season, with no gap and no overlap. Assert it.
- 2020–2025 are full calendar years; **2026 ends 3 September and must be labelled partial, never annualised silently.**
- Standing charges, taxes, degradation and discounting are out of scope and must be stated on the page, not silently omitted.
- Every number in prose must be computed from the frame beside it. This project has shipped prose contradicting its own data seven times.
- `data/meterdata/` and `data/Greenchoice - Variabele kosten - 2027` are the user's personal files: untracked, not git-ignored. **Never `git add` them; never use `git add -A`.** Stage only the exact paths each task lists.

---

### Task 1: The tariff model

**Files:**
- Create: `src/pvnight/tariff.py`
- Test: `tests/test_tariff.py`

**Interfaces:**
- Consumes: `pvnight.config.SITE_TZ` (`"Europe/Amsterdam"`).
- Produces:
  - `TARIFF_FILE = "data/Greenchoice - Variabele kosten - 2027"`
  - `@dataclass(frozen=True) class Tariff` with fields `bands: pd.DataFrame` and `hour_map: np.ndarray`
  - `Tariff.bands` columns: `band` (str), `levering_eur_kwh`, `terugleverkosten_eur_kwh`, `vergoeding_eur_kwh` (all float). Row order **is** the band index.
  - `Tariff.hour_map`: shape `(2, 24)` int, row 0 = winter, row 1 = summer, value = row index into `bands`.
  - `Tariff.net_export_eur_kwh` -> `np.ndarray`, `vergoeding − terugleverkosten` per band.
  - `Tariff.index_for(ts_utc: pd.Series | pd.DatetimeIndex) -> np.ndarray` of band indices.
  - `load_tariff(path: Path) -> Tariff`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tariff.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvnight.tariff import TARIFF_FILE, load_tariff

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tariff():
    return load_tariff(REPO / TARIFF_FILE)


def test_the_three_bands_carry_the_prices_from_the_file(tariff):
    b = tariff.bands.set_index("band")
    assert b.loc["Normaal", "levering_eur_kwh"] == pytest.approx(0.30566)
    assert b.loc["Normaal", "terugleverkosten_eur_kwh"] == pytest.approx(0.07049)
    assert b.loc["Normaal", "vergoeding_eur_kwh"] == pytest.approx(0.08050)
    assert b.loc["Dal", "levering_eur_kwh"] == pytest.approx(0.27939)
    assert b.loc["SuperDal", "levering_eur_kwh"] == pytest.approx(0.18225)


def test_net_export_is_one_cent_in_every_band(tariff):
    """The finding the whole phase rests on: terugleverkosten claw back all
    but a cent of the vergoeding, so there is no net metering left."""
    assert tariff.net_export_eur_kwh == pytest.approx(0.01, abs=2e-5)


def test_the_bands_tile_every_hour_of_both_seasons(tariff):
    assert tariff.hour_map.shape == (2, 24)
    assert (tariff.hour_map >= 0).all(), "an hour was left unassigned"


def test_the_same_clock_hour_changes_band_across_the_season_boundary(tariff):
    """31 March 14:00 local is Dal; 1 April 14:00 local is SuperDal.

    Both are 12:00 UTC in 2026 — the season, not the clock, decides.
    """
    ts = pd.DatetimeIndex(["2026-03-31 12:00", "2026-04-01 12:00"], tz="UTC")
    names = tariff.bands["band"].to_numpy()[tariff.index_for(ts)]
    assert list(names) == ["Dal", "SuperDal"]


def test_a_winter_afternoon_is_not_superdal(tariff):
    ts = pd.DatetimeIndex(["2026-01-15 11:00"], tz="UTC")   # 12:00 local
    assert tariff.bands["band"].to_numpy()[tariff.index_for(ts)][0] == "Dal"


def test_evening_peak_is_normaal_in_both_seasons(tariff):
    ts = pd.DatetimeIndex(["2026-01-17 17:30", "2026-07-17 16:30"], tz="UTC")
    names = tariff.bands["band"].to_numpy()[tariff.index_for(ts)]
    assert list(names) == ["Normaal", "Normaal"]


def test_every_interval_of_a_dst_spring_forward_day_gets_a_band(tariff):
    """2026-03-29: local jumps 02:00 -> 03:00, so local hour 2 never occurs.

    Assigning bands by naive local arithmetic rather than tz_convert is the
    kind of thing that produces a silently wrong hour here.
    """
    ts = pd.date_range("2026-03-29", periods=96, freq="15min", tz="UTC")
    idx = tariff.index_for(ts)
    assert len(idx) == 96
    assert (idx >= 0).all()
    local_hours = set(pd.DatetimeIndex(ts).tz_convert("Europe/Amsterdam").hour)
    assert 2 not in local_hours, "fixture assumption: 02:00 local is skipped"


def test_every_interval_of_a_dst_fall_back_day_gets_a_band(tariff):
    """2026-10-25: local hour 2 occurs twice. Both are Dal (00:00-07:00)."""
    ts = pd.date_range("2026-10-25", periods=96, freq="15min", tz="UTC")
    names = tariff.bands["band"].to_numpy()[tariff.index_for(ts)]
    local = pd.DatetimeIndex(ts).tz_convert("Europe/Amsterdam")
    twos = [n for n, h in zip(names, local.hour) if h == 2]
    assert len(twos) == 8, "fixture assumption: 02:00 local repeats"
    assert set(twos) == {"Dal"}


def test_a_malformed_file_raises_rather_than_returning_half_a_tariff(tmp_path):
    bad = tmp_path / "tariff.txt"
    bad.write_text("Zomer (1 april t/m 30 september)\n"
                   "    Normaal zomer (07:00 - 10:00)\n")
    with pytest.raises(ValueError, match="tile"):
        load_tariff(bad)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tariff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.tariff'`

- [ ] **Step 3: Implement `src/pvnight/tariff.py`**

```python
"""The electricity tariff, parsed from the supplier's own file.

Every euro figure in this project traces back to this module. The file gives
three prices per band, not one: what a kWh costs to import (`levering`), what
it costs to export (`terugleverkosten`), and what exporting pays
(`vergoeding`). The last two nearly cancel — net feed-in is EUR 0.01/kWh in
every band — which is why storing a kWh is worth 18-30x exporting it, and the
whole reason a battery can pay for itself here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SITE_TZ

TARIFF_FILE = "data/Greenchoice - Variabele kosten - 2027"

_SEASON_RE = re.compile(r"^(Zomer|Winter)\s*\(")
_BAND_RE = re.compile(r"^\s+(Normaal|Dal|SuperDal)\s+(zomer|winter)\s*\((.+)\)\s*$")
_PRICE_RE = re.compile(
    r"^\s+(Leveringskosten|Terugleverkosten|Terugleververgoeding)"
    r"\s+€\s*([0-9]+,[0-9]+)\s+per kWh\s*$")
_RANGE_RE = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")

_PRICE_COL = {
    "Leveringskosten": "levering_eur_kwh",
    "Terugleverkosten": "terugleverkosten_eur_kwh",
    "Terugleververgoeding": "vergoeding_eur_kwh",
}


@dataclass(frozen=True)
class Tariff:
    """Band prices, and the map from (season, local hour) to a band."""

    bands: pd.DataFrame
    hour_map: np.ndarray

    @property
    def net_export_eur_kwh(self) -> np.ndarray:
        """What a kWh exported actually earns, per band.

        Reported as a single derived array rather than left to callers,
        because the two components nearly cancelling is the finding.
        """
        return (self.bands["vergoeding_eur_kwh"].to_numpy()
                - self.bands["terugleverkosten_eur_kwh"].to_numpy())

    def index_for(self, ts_utc) -> np.ndarray:
        """Band index for each UTC timestamp.

        The one place this project converts to local time. Band boundaries
        are wall-clock Amsterdam, so 07:00-10:00 moves against UTC twice a
        year; `tz_convert` handles both transition days, and no naive local
        arithmetic happens anywhere. An hour misassigned between Normaal and
        SuperDal is a 12.3 cent/kWh error landing systematically in summer
        afternoons.
        """
        local = pd.DatetimeIndex(pd.Series(ts_utc).to_numpy()).tz_convert(SITE_TZ) \
            if not isinstance(ts_utc, pd.DatetimeIndex) else ts_utc.tz_convert(SITE_TZ)
        summer = ((local.month >= 4) & (local.month <= 9)).astype(int)
        return self.hour_map[summer, local.hour]


def _parse_ranges(text: str) -> list[tuple[int, int]]:
    """`"07:00 - 10:00 en 17:00 - 22:00"` -> `[(7, 10), (17, 22)]`.

    An end of `00:00` means midnight at the end of the day, so it becomes 24.
    """
    out = []
    for h1, m1, h2, m2 in _RANGE_RE.findall(text):
        if m1 != "00" or m2 != "00":
            raise ValueError(f"tariff: non-hourly band boundary in {text!r}")
        start, end = int(h1), int(h2)
        out.append((start, 24 if end == 0 else end))
    if not out:
        raise ValueError(f"tariff: no time ranges in {text!r}")
    return out


def load_tariff(path: Path) -> Tariff:
    """Parse the supplier's file into a `Tariff`.

    Strict by choice: a file that does not tile both seasons raises rather
    than yielding a tariff with silent holes, because a hole would price some
    energy at zero and no test of the euro figures would look wrong.
    """
    rows: list[dict] = []
    season = None
    cur: dict | None = None

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _SEASON_RE.match(line)
        if m:
            season = "summer" if m.group(1) == "Zomer" else "winter"
            cur = None
            continue
        m = _BAND_RE.match(line)
        if m:
            cur = {"season": season, "band": m.group(1),
                   "hours": _parse_ranges(m.group(3))}
            rows.append(cur)
            continue
        m = _PRICE_RE.match(line)
        if m:
            if cur is None:
                raise ValueError("tariff: a price line preceded any band")
            cur[_PRICE_COL[m.group(1)]] = float(m.group(2).replace(",", "."))

    if not rows:
        raise ValueError(f"tariff: no bands parsed from {path}")

    parsed = pd.DataFrame(rows)
    missing = [c for c in _PRICE_COL.values() if c not in parsed.columns]
    if missing or parsed[list(_PRICE_COL.values())].isna().any().any():
        raise ValueError(f"tariff: bands are missing prices {missing}")

    bands = (parsed.groupby("band", as_index=False)[list(_PRICE_COL.values())]
             .first()
             .sort_values("levering_eur_kwh", ascending=False)
             .reset_index(drop=True))
    band_row = {b: i for i, b in enumerate(bands["band"])}

    hour_map = np.full((2, 24), -1, dtype=int)
    for r in rows:
        s = 1 if r["season"] == "summer" else 0
        for start, end in r["hours"]:
            for h in range(start, end):
                if hour_map[s, h] != -1:
                    raise ValueError(
                        f"tariff: hour {h} of {r['season']} is in two bands")
                hour_map[s, h] = band_row[r["band"]]

    if (hour_map < 0).any():
        holes = [(s, h) for s in (0, 1) for h in range(24) if hour_map[s, h] < 0]
        raise ValueError(f"tariff: bands do not tile the day, holes at {holes}")

    return Tariff(bands=bands, hour_map=hour_map)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tariff.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Confirm the band mix on the real record**

Run:

```bash
uv run python -c "
from pathlib import Path
from pvnight import meter, tariff
t = tariff.load_tariff(Path('.') / tariff.TARIFF_FILE)
m = meter.load_meter(Path('.') / meter.METER_SUBDIR)
i = t.index_for(m['ts_utc'])
import numpy as np
for k, n in zip(t.bands['band'], np.bincount(i, minlength=len(t.bands))):
    print(f'{k:9} {n:7d} intervals  {100*n/len(i):5.1f}%')
"
```

Expected: all three bands present, percentages summing to 100, no band at 0.
Record the actual output in your report — it is the first evidence the band
map works on real timestamps rather than on fixtures.

- [ ] **Step 6: Commit**

```bash
git add src/pvnight/tariff.py tests/test_tariff.py
git commit -m "Parse the supplier tariff and map intervals to price bands"
```

---

### Task 2: Band-resolved flows from the simulator

**Files:**
- Modify: `src/pvnight/battery.py` (the `SimResult` dataclass and `simulate`)
- Test: `tests/test_battery.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1 — this task is independent of the tariff.
- Produces:
  - `simulate(net_wh, night_mask, nonev_mask, month_idx, spec, dt_hours=DT_HOURS, band_idx=None, n_bands=None) -> SimResult`
  - `SimResult.band_grid_import_wh: np.ndarray | None` — shape `(n_bands, n_caps)`
  - `SimResult.band_export_wh: np.ndarray | None` — shape `(n_bands, n_caps)`
  - Both are `None` when `band_idx is None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_battery.py`:

```python
def test_band_flows_are_absent_unless_asked_for():
    net = np.array([1000.0, -1000.0])
    z = np.zeros(2, bool)
    mi = np.zeros(2, int)
    r = simulate(net, z, z, mi, BatterySpec(np.array([5.0]), power_kw=3.0))
    assert r.band_grid_import_wh is None
    assert r.band_export_wh is None


def test_band_flows_reconcile_with_the_totals():
    """Splitting a flow by band must not create or destroy energy.

    This is the guard that matters: a band map with a hole, or an off-by-one
    index, would still produce plausible-looking euro figures.
    """
    rng = np.random.default_rng(0)
    net = rng.normal(0, 800, 500)
    z = np.zeros(500, bool)
    mi = np.zeros(500, int)
    bands = rng.integers(0, 3, 500)
    spec = BatterySpec(np.arange(0.0, 6.1, 1.5), power_kw=3.0)
    r = simulate(net, z, z, mi, spec, band_idx=bands, n_bands=3)

    assert r.band_grid_import_wh.shape == (3, len(spec.capacities_kwh))
    assert r.band_grid_import_wh.sum(axis=0) == pytest.approx(r.grid_import_wh)
    assert r.band_export_wh.sum(axis=0) == pytest.approx(r.export_wh)


def test_asking_for_band_flows_does_not_change_any_other_result():
    """The additive keyword must be exactly that.

    Phases 2 and 3 both rest on this dispatch loop, and their published
    figures must not move because a later phase wanted a new accumulator.
    """
    rng = np.random.default_rng(1)
    net = rng.normal(0, 800, 400)
    nm = rng.random(400) < 0.4
    em = nm & (rng.random(400) < 0.5)
    mi = rng.integers(0, 12, 400)
    spec = BatterySpec(np.arange(0.0, 9.1, 1.5), power_kw=3.0)

    plain = simulate(net, nm, em, mi, spec)
    banded = simulate(net, nm, em, mi, spec, band_idx=rng.integers(0, 3, 400),
                      n_bands=3)

    for field in ("grid_import_wh", "export_wh", "charge_wh", "discharge_wh",
                  "night_grid_import_wh", "nonev_grid_import_wh",
                  "final_stored_wh"):
        assert getattr(plain, field) == pytest.approx(getattr(banded, field)), field
    assert plain.monthly_discharge_wh == pytest.approx(banded.monthly_discharge_wh)


def test_n_bands_defaults_to_the_highest_index_present():
    net = np.array([1000.0, -1000.0])
    z = np.zeros(2, bool)
    mi = np.zeros(2, int)
    r = simulate(net, z, z, mi, BatterySpec(np.array([5.0]), power_kw=3.0),
                 band_idx=np.array([0, 1]))
    assert r.band_grid_import_wh.shape[0] == 2


def test_a_band_index_out_of_range_raises_rather_than_wrapping():
    """Negative or oversized indices must not silently land in another band."""
    net = np.array([1000.0, -1000.0])
    z = np.zeros(2, bool)
    mi = np.zeros(2, int)
    with pytest.raises(ValueError, match="band_idx"):
        simulate(net, z, z, mi, BatterySpec(np.array([5.0]), power_kw=3.0),
                 band_idx=np.array([0, 5]), n_bands=3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_battery.py -k band -v`
Expected: FAIL — `AttributeError: 'SimResult' object has no attribute 'band_grid_import_wh'`

- [ ] **Step 3: Add the fields to `SimResult`**

In `src/pvnight/battery.py`, add two fields to the end of the `SimResult`
dataclass (it is `@dataclass`, not frozen, and these have defaults so every
existing construction site keeps working):

```python
    band_grid_import_wh: np.ndarray | None = None
    band_export_wh: np.ndarray | None = None
```

- [ ] **Step 4: Extend `simulate`**

Change the signature to:

```python
def simulate(
    net_wh: np.ndarray,
    night_mask: np.ndarray,
    nonev_mask: np.ndarray,
    month_idx: np.ndarray,
    spec: BatterySpec,
    dt_hours: float = DT_HOURS,
    band_idx: np.ndarray | None = None,
    n_bands: int | None = None,
) -> SimResult:
```

Extend the docstring with:

```
    `band_idx` is optional and additive: when given, import and export are
    additionally accumulated per tariff band, using the same pattern
    `month_idx` already uses for discharge. With it None the results are
    bit-identical to a run without it, which phases 2 and 3 depend on.
```

Before the loop, beside the other accumulators:

```python
    band_import = band_export = None
    if band_idx is not None:
        band_idx = np.asarray(band_idx)
        if n_bands is None:
            n_bands = int(band_idx.max()) + 1
        if band_idx.min() < 0 or band_idx.max() >= n_bands:
            raise ValueError(
                f"band_idx out of range: [{band_idx.min()}, {band_idx.max()}] "
                f"against n_bands={n_bands}")
        band_import = np.zeros((n_bands, n))
        band_export = np.zeros((n_bands, n))
```

Inside the loop, in the `e > 0.0` branch, immediately after
`export += e - accept`:

```python
            if band_export is not None:
                band_export[band_idx[t]] += e - accept
```

and in the `elif e < 0.0` branch, immediately after `grid += shortfall`:

```python
            if band_import is not None:
                band_import[band_idx[t]] += shortfall
```

Add to the `SimResult(...)` construction:

```python
        band_grid_import_wh=band_import,
        band_export_wh=band_export,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_battery.py -v`
Expected: PASS, all tests including the five new ones.

- [ ] **Step 6: Verify phases 2 and 3 have not moved**

Run: `uv run pytest -q`
Expected: PASS. 175 existing plus 9 (Task 1) plus 5 (this task) = **189**.

Then confirm the published figures are unchanged:

```bash
uv run python analyze_night.py | head -20
```

Expected: phase 2's elbow still 8.0 and `recommend_capacity` still 7.5. If
either has moved, **stop and report it** — the keyword was supposed to be
additive.

- [ ] **Step 7: Commit**

```bash
git add src/pvnight/battery.py tests/test_battery.py
git commit -m "Accumulate import and export per tariff band, optionally"
```

---

### Task 3: Pricing the sweep

**Files:**
- Create: `src/pvnight/economics.py`
- Test: `tests/test_economics.py`

**Interfaces:**
- Consumes: `tariff.Tariff` (Task 1); `battery.simulate(..., band_idx=, n_bands=)` and `SimResult.band_grid_import_wh` / `.band_export_wh` (Task 2); `meter_battery.bound_signals`, `meter_battery._masks`, `meter.DT_HOURS`; `battery.BatterySpec`.
- Produces:
  - `FULL_YEARS_END = pd.Timestamp("2026-01-01", tz="UTC")` — the first instant not in a full calendar year of this record.
  - `year_band_index(ts_utc, tariff) -> tuple[np.ndarray, list[int], int]` returning the combined index, the year list, and `n_bands`.
  - `price_capacities(meter_df, nights_df, capacities_kwh, tariff, power_kw=3.0, round_trip=0.90, usable_fraction=0.90) -> pd.DataFrame` with one row per (capacity, year) and columns `capacity_kwh`, `year`, `is_full_year`, `import_kwh`, `export_kwh`, `cost_eur`.
  - `savings(priced: pd.DataFrame) -> pd.DataFrame` with `capacity_kwh`, `year`, `is_full_year`, `cost_eur`, `saving_eur`.
  - `break_even_eur_per_kwh(saving_eur_per_yr: float, capacity_kwh: float, years: float) -> float`
  - `payback_years(saving_eur_per_yr, capacity_kwh, eur_per_kwh, fixed_cost_eur) -> float`
  - `recommend_capacity_eur(annual: pd.DataFrame, eur_per_kwh, fixed_cost_eur, years) -> float`
  - `blended_value_eur_per_kwh(priced: pd.DataFrame, capacity_kwh: float) -> float`
  - `derived_threshold_kwh_per_kwh(eur_per_kwh, years, blended_value) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_economics.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pvnight import economics
from pvnight.tariff import Tariff


@pytest.fixture
def toy_tariff():
    """Two bands at round prices, so euro answers are checkable by hand."""
    bands = pd.DataFrame({
        "band": ["Peak", "Off"],
        "levering_eur_kwh": [0.30, 0.20],
        "terugleverkosten_eur_kwh": [0.07, 0.05],
        "vergoeding_eur_kwh": [0.08, 0.06],
    })
    hour_map = np.zeros((2, 24), dtype=int)
    hour_map[:, 0:12] = 1          # Off overnight and morning
    hour_map[:, 12:24] = 0         # Peak afternoon and evening
    return Tariff(bands=bands, hour_map=hour_map)


def test_net_export_is_the_difference_not_the_headline_rate(toy_tariff):
    assert toy_tariff.net_export_eur_kwh == pytest.approx([0.01, 0.01])


def test_break_even_inverts_payback():
    """At the break-even price, payback must equal the horizon exactly."""
    price = economics.break_even_eur_per_kwh(600.0, capacity_kwh=9.0, years=10.0)
    assert economics.payback_years(600.0, 9.0, price, 0.0) == pytest.approx(10.0)


def test_break_even_is_saving_times_years_over_capacity():
    assert economics.break_even_eur_per_kwh(450.0, 9.0, 10.0) == pytest.approx(500.0)


def test_payback_includes_the_fixed_cost():
    assert economics.payback_years(500.0, 10.0, 400.0, 1500.0) == pytest.approx(11.0)


def test_recommend_picks_the_capacity_with_the_best_net_position():
    """Net position, not the largest saving: a bigger battery always saves
    more energy, so an unpriced rule would recommend the largest one swept."""
    annual = pd.DataFrame({
        "capacity_kwh": [5.0, 10.0, 20.0],
        "saving_eur": [400.0, 600.0, 650.0],
    })
    # over 10 years at 300 EUR/kWh: 4000-1500=2500, 6000-3000=3000, 6500-6000=500
    assert economics.recommend_capacity_eur(annual, 300.0, 0.0, 10.0) == 10.0


def test_recommend_returns_zero_when_nothing_pays_back():
    annual = pd.DataFrame({"capacity_kwh": [5.0, 10.0],
                           "saving_eur": [50.0, 60.0]})
    assert economics.recommend_capacity_eur(annual, 900.0, 0.0, 10.0) == 0.0


def test_derived_threshold_replaces_the_invented_fifty():
    """EUR 500/kWh over 10 years at EUR 0.28/kWh blended needs ~179 kWh/yr."""
    got = economics.derived_threshold_kwh_per_kwh(500.0, 10.0, 0.28)
    assert got == pytest.approx(500.0 / (10.0 * 0.28))
    assert got > 50.0


def _toy_meter(hours=48):
    """Alternating export by day and import by night, on the toy bands."""
    ts = pd.date_range("2024-06-01", periods=hours * 4, freq="15min", tz="UTC")
    local_hour = ts.tz_convert("Europe/Amsterdam").hour
    exporting = (local_hour >= 12)
    return pd.DataFrame({
        "ts_utc": ts,
        "import_kwh": np.where(exporting, 0.0, 0.25),
        "export_kwh": np.where(exporting, 0.5, 0.0),
    })


def _toy_nights(meter_df):
    d = meter_df["ts_utc"].dt.tz_convert("Europe/Amsterdam").dt.date
    rows = []
    for day in sorted(set(d)):
        base = pd.Timestamp(day, tz="Europe/Amsterdam").tz_convert("UTC")
        rows.append({"date": pd.Timestamp(day), "night_start_utc": base,
                     "night_end_utc": base + pd.Timedelta(hours=6),
                     "is_ev": False, "covered": True})
    return pd.DataFrame(rows)


def test_zero_capacity_reproduces_the_bill_computed_straight_from_the_meter(toy_tariff):
    """The anchor. cost(0) must equal the meter's own arithmetic, or every
    euro figure downstream is measuring the simulator rather than the house.
    """
    m = _toy_meter()
    priced = economics.price_capacities(m, _toy_nights(m), np.array([0.0]),
                                        toy_tariff)
    got = priced["cost_eur"].sum()

    b = toy_tariff.index_for(m["ts_utc"])
    lev = toy_tariff.bands["levering_eur_kwh"].to_numpy()[b]
    net = toy_tariff.net_export_eur_kwh[b]
    want = float((m["import_kwh"] * lev - m["export_kwh"] * net).sum())
    assert got == pytest.approx(want, rel=1e-9)


def test_a_battery_lowers_the_bill_and_the_saving_is_positive(toy_tariff):
    m = _toy_meter()
    priced = economics.price_capacities(m, _toy_nights(m), np.array([0.0, 5.0]),
                                        toy_tariff)
    s = economics.savings(priced)
    zero = s.loc[s.capacity_kwh == 0.0, "saving_eur"].sum()
    five = s.loc[s.capacity_kwh == 5.0, "saving_eur"].sum()
    assert zero == pytest.approx(0.0)
    assert five > 0.0


def test_the_partial_year_is_flagged_rather_than_annualised(toy_tariff):
    """2026 ends on 3 September in this record. Treating it as a full year
    would understate EUR/yr by a third and nothing would look wrong."""
    m = _toy_meter()
    m = pd.concat([m, m.assign(ts_utc=m["ts_utc"] + pd.DateOffset(years=2))],
                  ignore_index=True)
    priced = economics.price_capacities(m, _toy_nights(m), np.array([0.0]),
                                        toy_tariff)
    assert set(priced["year"]) == {2024, 2026}
    assert priced.loc[priced.year == 2024, "is_full_year"].all()
    assert not priced.loc[priced.year == 2026, "is_full_year"].any()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_economics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pvnight.economics'`

- [ ] **Step 3: Implement `src/pvnight/economics.py`**

```python
"""The battery in euros.

Every capacity recommendation this project published before this module was
energy-only, which meant trading a stock (kWh bought) against a flow (kWh/yr
saved) with no exchange rate. Two attempts to supply one failed: an invented
50 kWh/yr threshold, removed at the user's instruction, and a geometric elbow
that turned out to drift with the sweep's truncation. A tariff supplies a
real rate, so this module is where the recommendation stops being arbitrary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import BatterySpec, simulate
from .meter import DT_HOURS
from .meter_battery import _masks, bound_signals
from .tariff import Tariff

# The meter record ends 3 September 2026, so 2026 is not a full calendar year
# and must never be annualised. Anything at or after this instant is partial.
FULL_YEARS_END = pd.Timestamp("2026-01-01", tz="UTC")


def year_band_index(ts_utc, tariff: Tariff) -> tuple[np.ndarray, list[int], int]:
    """One index encoding both the calendar year and the tariff band.

    Per-year figures could be had by simulating each year separately, but the
    battery's state of charge carries across a year boundary and phase 3's
    spec is explicit that the timeline is never broken up. Encoding
    `year * n_bands + band` into a single accumulator index gets per-year
    costs out of one continuous simulation instead.
    """
    idx = pd.DatetimeIndex(pd.Series(ts_utc).to_numpy())
    years = sorted(set(int(y) for y in idx.year))
    pos = {y: i for i, y in enumerate(years)}
    n_bands = len(tariff.bands)
    band = tariff.index_for(idx)
    year_pos = np.array([pos[int(y)] for y in idx.year])
    return year_pos * n_bands + band, years, n_bands


def price_capacities(meter_df: pd.DataFrame, nights_df: pd.DataFrame,
                     capacities_kwh: np.ndarray, tariff: Tariff,
                     power_kw: float = 3.0, round_trip: float = 0.90,
                     usable_fraction: float = 0.90) -> pd.DataFrame:
    """Cost of the variable electricity bill, per capacity and per year.

    Uses the charge-first ordering. Phase 3 measured the two within-interval
    orderings to differ by under 0.06% of baseline import and to agree on the
    elbow, so the choice does not move the answer.
    """
    m = meter_df.sort_values("ts_utc").reset_index(drop=True)
    night_i, nonev_i, month_i = _masks(m, nights_df)
    sig, src = bound_signals(m)["charge_first"]

    combo, years, n_bands = year_band_index(m["ts_utc"], tariff)
    spec = BatterySpec(np.asarray(capacities_kwh, float), power_kw,
                       round_trip, usable_fraction)
    r = simulate(sig, night_i[src], nonev_i[src], month_i[src], spec,
                 dt_hours=DT_HOURS, band_idx=combo[src],
                 n_bands=len(years) * n_bands)

    lev = tariff.bands["levering_eur_kwh"].to_numpy()
    net_exp = tariff.net_export_eur_kwh
    rows = []
    for yi, year in enumerate(years):
        sl = slice(yi * n_bands, (yi + 1) * n_bands)
        imp = r.band_grid_import_wh[sl] / 1000.0        # (n_bands, n_caps)
        exp = r.band_export_wh[sl] / 1000.0
        cost = (imp * lev[:, None]).sum(axis=0) - (exp * net_exp[:, None]).sum(axis=0)
        for ci, cap in enumerate(spec.capacities_kwh):
            rows.append({
                "capacity_kwh": float(cap),
                "year": int(year),
                "is_full_year": bool(
                    pd.Timestamp(f"{year}-01-01", tz="UTC") < FULL_YEARS_END),
                "import_kwh": float(imp[:, ci].sum()),
                "export_kwh": float(exp[:, ci].sum()),
                "cost_eur": float(cost[ci]),
            })
    return pd.DataFrame(rows)


def savings(priced: pd.DataFrame) -> pd.DataFrame:
    """What each capacity saves against no battery, per year."""
    base = (priced[priced["capacity_kwh"] == 0.0]
            .set_index("year")["cost_eur"])
    if base.empty:
        raise ValueError(
            "savings: the sweep must include capacity 0.0 — it is the "
            "baseline every saving is measured against")
    out = priced.copy()
    out["saving_eur"] = out["year"].map(base).to_numpy() - out["cost_eur"]
    return out[["capacity_kwh", "year", "is_full_year", "cost_eur", "saving_eur"]]


def break_even_eur_per_kwh(saving_eur_per_yr: float, capacity_kwh: float,
                           years: float) -> float:
    """Installed cost per kWh at which this capacity exactly repays itself.

    Undiscounted, and degradation is not modelled — both stated on the page.
    """
    if capacity_kwh <= 0:
        return float("nan")
    return float(saving_eur_per_yr * years / capacity_kwh)


def payback_years(saving_eur_per_yr: float, capacity_kwh: float,
                  eur_per_kwh: float, fixed_cost_eur: float) -> float:
    if saving_eur_per_yr <= 0:
        return float("inf")
    return float((fixed_cost_eur + capacity_kwh * eur_per_kwh)
                 / saving_eur_per_yr)


def recommend_capacity_eur(annual: pd.DataFrame, eur_per_kwh: float,
                           fixed_cost_eur: float, years: float) -> float:
    """The capacity with the best undiscounted net position over `years`.

    Not the largest saving: a bigger battery always saves more energy, so a
    rule that maximised saving would always recommend the largest capacity
    swept. Returns 0.0 when nothing pays back, which is a real answer.
    """
    a = annual.sort_values("capacity_kwh")
    net = (a["saving_eur"].to_numpy() * years
           - fixed_cost_eur - a["capacity_kwh"].to_numpy() * eur_per_kwh)
    if net.max() <= 0:
        return 0.0
    return float(a["capacity_kwh"].to_numpy()[int(np.argmax(net))])


def blended_value_eur_per_kwh(priced: pd.DataFrame, capacity_kwh: float) -> float:
    """Euros saved per kWh of grid import the battery actually displaced.

    Measured from the band mix the battery displaces rather than assumed,
    because displaced energy is not spread evenly across the bands.
    """
    s = savings(priced)
    row = s[s["capacity_kwh"] == capacity_kwh]
    base = priced[priced["capacity_kwh"] == 0.0]["import_kwh"].sum()
    here = priced[priced["capacity_kwh"] == capacity_kwh]["import_kwh"].sum()
    displaced = base - here
    if displaced <= 0:
        return float("nan")
    return float(row["saving_eur"].sum() / displaced)


def derived_threshold_kwh_per_kwh(eur_per_kwh: float, years: float,
                                  blended_value: float) -> float:
    """The kWh/yr a marginal kWh of capacity must return to be worth buying.

    This is the number `battery.recommend_capacity` was always missing. Its
    default of 50 was a judgement call the user had removed from the report;
    this replaces it with an arithmetic consequence of a real price.
    """
    if years <= 0 or blended_value <= 0:
        return float("nan")
    return float(eur_per_kwh / (years * blended_value))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_economics.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Measure the real figures and record them**

Run:

```bash
uv run python -c "
from pathlib import Path
import numpy as np, pandas as pd
from pvnight import meter, meter_nights, tariff, economics
root = Path('.')
t = tariff.load_tariff(root / tariff.TARIFF_FILE)
w = pd.read_csv(root/'out/solar_windows.csv', parse_dates=['date','solar_start_utc','solar_end_utc','night_start_utc','night_end_utc'])
m = meter.load_meter(root / meter.METER_SUBDIR); g = meter.find_gaps(m)
n = meter_nights.summarise(m, w, g)
caps = np.arange(0.0, 30.01, 0.5)
p = economics.price_capacities(m, n, caps, t)
s = economics.savings(p)
full = s[s.is_full_year]
ann = full.groupby('capacity_kwh', as_index=False)['saving_eur'].mean()
print('no-battery bill, 2025:', round(p[(p.capacity_kwh==0)&(p.year==2025)].cost_eur.sum(), 2), 'EUR')
for c in (5.0, 9.0, 15.0, 30.0):
    row = ann[ann.capacity_kwh==c]
    sv = float(row.saving_eur.iloc[0])
    print(f'{c:5.1f} kWh: {sv:7.2f} EUR/yr  break-even {economics.break_even_eur_per_kwh(sv, c, 10):7.1f} EUR/kWh @10yr')
for price in (300.0, 500.0, 700.0):
    print(f'optimum @ {price:.0f} EUR/kWh, 10yr:', economics.recommend_capacity_eur(ann, price, 0.0, 10.0), 'kWh')
bv = economics.blended_value_eur_per_kwh(p, 9.0)
print('blended value:', round(bv, 4), 'EUR/kWh')
print('derived threshold @500/10yr:', round(economics.derived_threshold_kwh_per_kwh(500.0, 10.0, bv), 1), 'kWh/yr  (the removed invention was 50)')
"
```

**Check the spec's three predictions against this output and report each
explicitly as held or broken:**

1. Is the euro optimum **larger** than phase 3's 9.0 kWh? If it is smaller,
   say so loudly — the spec says treat that as a bug until proven otherwise.
2. Does the euro optimum stay put when the sweep top changes? Re-run
   `recommend_capacity_eur` on `caps` truncated to 15 and to 20 and compare.
3. Is the derived threshold near 180 kWh/yr, and above 50?

A broken prediction is a finding to report, not a number to quietly accept.

- [ ] **Step 6: Commit**

```bash
git add src/pvnight/economics.py tests/test_economics.py
git commit -m "Price the capacity sweep against the tariff"
```

---

### Task 4: The arbitrage ceiling

**Files:**
- Modify: `src/pvnight/economics.py` (append)
- Test: `tests/test_economics.py` (append)

**Interfaces:**
- Consumes: `tariff.Tariff`.
- Produces: `arbitrage_ceiling_eur_yr(meter_df, tariff, capacity_kwh, power_kw=3.0, round_trip=0.90, usable_fraction=0.90) -> dict` with keys `ceiling_eur_yr`, `best_spread_eur_kwh`, `usable_days`, `years`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_economics.py`:

```python
def test_arbitrage_ceiling_uses_the_widest_spread_available_in_a_day(toy_tariff):
    """Buy at the cheapest band, discharge at the dearest, once a day.

    Deliberately generous: it ignores that the battery is already busy doing
    self-consumption. A ceiling is allowed to be unreachable — that is what
    makes it a ceiling — but it must be labelled as one wherever it appears.
    """
    ts = pd.date_range("2024-06-01", periods=96 * 10, freq="15min", tz="UTC")
    m = pd.DataFrame({"ts_utc": ts, "import_kwh": 0.0, "export_kwh": 0.0})
    out = economics.arbitrage_ceiling_eur_yr(m, toy_tariff, capacity_kwh=10.0)

    # Toy bands: buy at 0.20, displace 0.30, round trip 0.90.
    assert out["best_spread_eur_kwh"] == pytest.approx(0.30 - 0.20 / 0.90)
    assert out["ceiling_eur_yr"] > 0.0


def test_arbitrage_ceiling_is_zero_when_every_band_costs_the_same():
    flat = Tariff(
        bands=pd.DataFrame({"band": ["A", "B"],
                            "levering_eur_kwh": [0.25, 0.25],
                            "terugleverkosten_eur_kwh": [0.05, 0.05],
                            "vergoeding_eur_kwh": [0.06, 0.06]}),
        hour_map=np.zeros((2, 24), dtype=int),
    )
    ts = pd.date_range("2024-06-01", periods=96 * 5, freq="15min", tz="UTC")
    m = pd.DataFrame({"ts_utc": ts, "import_kwh": 0.0, "export_kwh": 0.0})
    out = economics.arbitrage_ceiling_eur_yr(m, flat, capacity_kwh=10.0)
    assert out["ceiling_eur_yr"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_economics.py -k arbitrage -v`
Expected: FAIL — `AttributeError: module 'pvnight.economics' has no attribute 'arbitrage_ceiling_eur_yr'`

- [ ] **Step 3: Implement it**

Append to `src/pvnight/economics.py`:

```python
def arbitrage_ceiling_eur_yr(meter_df: pd.DataFrame, tariff: Tariff,
                             capacity_kwh: float, power_kw: float = 3.0,
                             round_trip: float = 0.90,
                             usable_fraction: float = 0.90) -> dict:
    """An upper bound on what price-aware grid charging could add.

    Not a recommendation and not a dispatch policy. It answers one question:
    is a price-aware phase worth building at all?

    The bound is deliberately loose in three ways, all of which push it up,
    so a small number here is genuinely conclusive while a large one only
    means "worth investigating":

    - it assumes one full cycle every day at the day's widest band spread;
    - it ignores that the battery is already occupied storing solar surplus;
    - it uses perfect foresight, which no policy has.

    Hindsight is legitimate for a bound and illegitimate for a
    recommendation, so every caller must label it.
    """
    idx = pd.DatetimeIndex(meter_df["ts_utc"])
    band = tariff.index_for(idx)
    lev = tariff.bands["levering_eur_kwh"].to_numpy()[band]

    local_day = idx.tz_convert(pd.DatetimeIndex(idx).tz).normalize()
    d = pd.DataFrame({"day": local_day, "price": lev})
    per_day = d.groupby("day")["price"].agg(["min", "max"])

    # Buying loses the round trip; displacing does not.
    spread = per_day["max"] - per_day["min"] / round_trip
    spread = spread.clip(lower=0.0)

    eta = float(np.sqrt(round_trip))
    usable_kwh = capacity_kwh * usable_fraction
    # A cycle is also capped by what the inverter can move in a day.
    movable_kwh = min(usable_kwh, power_kw * 24.0 * eta)

    total = float((spread * movable_kwh).sum())
    span_s = (idx.max() - idx.min()).total_seconds()
    years = span_s / (365.25 * 24 * 3600)
    return {
        "ceiling_eur_yr": total / years if years > 0 else float("nan"),
        "best_spread_eur_kwh": float(spread.max()) if len(spread) else 0.0,
        "usable_days": int((spread > 0).sum()),
        "years": float(years),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_economics.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Measure it on the real record**

Run:

```bash
uv run python -c "
from pathlib import Path
from pvnight import meter, tariff, economics
root = Path('.')
t = tariff.load_tariff(root / tariff.TARIFF_FILE)
m = meter.load_meter(root / meter.METER_SUBDIR)
print(economics.arbitrage_ceiling_eur_yr(m, t, capacity_kwh=9.0))
"
```

Report the ceiling in euros per year, and say plainly whether it is large
enough to justify a price-aware phase 5 given that it is an over-estimate.

- [ ] **Step 6: Commit**

```bash
git add src/pvnight/economics.py tests/test_economics.py
git commit -m "Bound what price-aware grid charging could add"
```

---

### Task 5: Report, entry point, outputs and README

**Files:**
- Modify: `src/pvnight/compare_report.py`
- Modify: `analyze_meter.py`
- Modify: `README.md`
- Test: `tests/test_compare_report.py` (append), `tests/test_analyze_meter.py` (modify the chart count)

**Interfaces:**
- Consumes: everything above.
- Produces: `analyze_meter.run` gains summary keys `no_battery_cost_eur`, `annual_saving_eur`, `euro_optimum_kwh`, `breakeven_eur_per_kwh`, `derived_threshold_kwh_per_kwh`, `arbitrage_ceiling_eur_yr`; and writes `out/meter_tariff_bands.csv` and `out/meter_economics.csv`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compare_report.py`:

```python
def test_the_tariff_card_states_the_measured_net_export(monkeypatch, toy):
    """The 18-30x spread is the whole economic case, so it must be computed
    from the band table rather than written beside it."""
    bands = pd.DataFrame({
        "band": ["Normaal", "Dal", "SuperDal"],
        "levering_eur_kwh": [0.30566, 0.27939, 0.18225],
        "terugleverkosten_eur_kwh": [0.07049, 0.05964, 0.01951],
        "vergoeding_eur_kwh": [0.08050, 0.06965, 0.02951],
    })
    html = compare_report.tariff_card(bands)
    assert "0.010" in html          # the derived net export
    assert "30.5" in html           # the derived Normaal multiplier
    assert "0.30566" in html


def test_the_tariff_card_moves_with_the_prices(monkeypatch):
    """Halve the import price and the multiplier must halve with it."""
    bands = pd.DataFrame({
        "band": ["Only"],
        "levering_eur_kwh": [0.10],
        "terugleverkosten_eur_kwh": [0.04],
        "vergoeding_eur_kwh": [0.05],
    })
    html = compare_report.tariff_card(bands)
    assert "10.0" in html           # 0.10 / 0.01
    assert "30.5" not in html
```

Modify `tests/test_analyze_meter.py`: the SVG assertion becomes

```python
    # Six analysis charts, two winter-wall charts, two euro charts.
    assert (tmp_path / "meter_report.html").read_text().count("<svg") == 10
    assert (tmp_path / "meter_economics.csv").exists()
    assert (tmp_path / "meter_tariff_bands.csv").exists()

    assert s["no_battery_cost_eur"] > 0.0
    assert s["annual_saving_eur"] > 0.0
    assert s["derived_threshold_kwh_per_kwh"] > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compare_report.py -k tariff -v`
Expected: FAIL — `AttributeError: module 'pvnight.compare_report' has no attribute 'tariff_card'`

- [ ] **Step 3: Add the report pieces**

In `src/pvnight/compare_report.py`, add:

```python
def tariff_card(bands: pd.DataFrame) -> str:
    """The tariff table and what it implies, both read from the frame.

    The 18-30x multiplier is the economic case for the whole phase, so it is
    computed here rather than quoted: change the prices and the sentence
    changes with them.
    """
    net = (bands["vergoeding_eur_kwh"] - bands["terugleverkosten_eur_kwh"])
    ratio = bands["levering_eur_kwh"] / net
    head = "".join(f"<th>{h}</th>" for h in
                   ["band", "import €/kWh", "export cost", "export paid",
                    "net export", "×"])
    rows = "".join(
        f"<tr><td>{b}</td><td>{lv:.5f}</td><td>{tk:.5f}</td>"
        f"<td>{vg:.5f}</td><td>{n:.5f}</td><td>{r:.1f}×</td></tr>"
        for b, lv, tk, vg, n, r in zip(
            bands["band"], bands["levering_eur_kwh"],
            bands["terugleverkosten_eur_kwh"], bands["vergoeding_eur_kwh"],
            net, ratio))
    return (
        f'<p class="sub">Exporting a kWh nets <strong>€{net.mean():.3f}</strong>'
        f" — the feed-in charge claws back all but a fraction of the feed-in "
        f"payment, so there is no net metering left in this contract. The same "
        f"kWh stored and used later is worth <strong>{ratio.min():.1f}× to "
        f"{ratio.max():.1f}×</strong> that. This is the entire economic case "
        "for a battery here.</p>"
        f'<div class="chart"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>")


def chart_savings(annual: pd.DataFrame) -> str:
    """Euros saved per year against capacity."""
    fig, ax = plt.subplots(figsize=(9, 4))
    a = annual.sort_values("capacity_kwh")
    ax.plot(a["capacity_kwh"], a["saving_eur"], lw=2, color=SERIES[0])
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel("€ saved per year")
    return _svg(fig)


def chart_break_even(annual: pd.DataFrame, years: float) -> str:
    """Installed €/kWh each capacity must beat to repay itself."""
    fig, ax = plt.subplots(figsize=(9, 4))
    a = annual[annual["capacity_kwh"] > 0].sort_values("capacity_kwh")
    ax.plot(a["capacity_kwh"], a["break_even_eur_per_kwh"], lw=2,
            color=SERIES[1])
    ax.set_xlabel("battery capacity (kWh, nameplate)")
    ax.set_ylabel(f"break-even €/kWh over {years:.0f} years")
    return _svg(fig)
```

Extend `build_html`'s signature with, after `coverable_pct`:

```python
                tariff_bands: pd.DataFrame | None = None,
                annual_savings: pd.DataFrame | None = None,
                economics_summary: dict | None = None,
                horizon_years: float = 10.0,
```

and add the euro cards immediately **after** the capacity cards, so the
energy answer is read first and the euro answer is read as its refinement:

```python
        *([] if tariff_bands is None else [
            ("What a kWh is actually worth", tariff_card(tariff_bands), "")]),
        *([] if annual_savings is None or economics_summary is None else [
            ("What the battery saves, in euros",
             _euro_prose(economics_summary, horizon_years),
             chart_savings(annual_savings)),
            ("What it would have to cost",
             "The installed price per kWh at which each capacity exactly "
             f"repays itself in {horizon_years:.0f} years, undiscounted and "
             "ignoring degradation. Read your own answer off it when a quote "
             "arrives. Standing charges, network tariffs and taxes are "
             "excluded throughout: they do not change with battery size, so "
             "they cancel in every comparison here.",
             chart_break_even(annual_savings, horizon_years))]),
```

Note the first entry has an empty SVG string — `build_html` already renders
`<div class="chart">{svg}</div>`, and an empty string there is a card with a
table and no chart, which is what the tariff card needs.

Add the prose helper:

```python
def _euro_prose(s: dict, years: float) -> str:
    """The euro finding, every number read from the summary dict."""
    return (
        f"Under this tariff the variable electricity bill without a battery "
        f"comes to <strong>€{s['no_battery_cost_eur']:,.0f}</strong> in the "
        f"most recent full year. The euro-optimal capacity is "
        f"<strong>{s['euro_optimum_kwh']:.1f} kWh</strong>, saving "
        f"<strong>€{s['annual_saving_eur']:,.0f}</strong> a year, which is a "
        f"break-even installed cost of "
        f"<strong>€{s['breakeven_eur_per_kwh']:,.0f}/kWh</strong> over "
        f"{years:.0f} years. Phase 2 published a 50 kWh/yr rule of thumb that "
        "was never derived from anything and was removed; the same rule "
        f"derived from this tariff is <strong>"
        f"{s['derived_threshold_kwh_per_kwh']:.0f} kWh/yr</strong>. "
        f"A perfect-foresight upper bound on price-aware grid charging would "
        f"add at most <strong>€{s['arbitrage_ceiling_eur_yr']:,.0f}</strong> a "
        "year on top — a ceiling, not a policy, and unreachable by any real "
        "dispatch.")
```

- [ ] **Step 4: Wire the entry point**

In `analyze_meter.py`, add to the imports `tariff` and `economics`, add
`HORIZON_YEARS = 10.0` and `EXAMPLE_EUR_PER_KWH = 500.0` beside the other
constants, and inside `run()` after the winter-wall block:

```python
    tar = tariff.load_tariff(repo_root / tariff.TARIFF_FILE)
    priced = economics.price_capacities(meter_df, nights_df, CAPACITIES, tar)
    saved = economics.savings(priced)
    full = saved[saved["is_full_year"]]
    annual = (full.groupby("capacity_kwh", as_index=False)["saving_eur"].mean())
    annual["break_even_eur_per_kwh"] = [
        economics.break_even_eur_per_kwh(s, c, HORIZON_YEARS)
        for s, c in zip(annual["saving_eur"], annual["capacity_kwh"])]

    euro_opt = economics.recommend_capacity_eur(
        annual, EXAMPLE_EUR_PER_KWH, 0.0, HORIZON_YEARS)
    blended = economics.blended_value_eur_per_kwh(priced, euro_opt or 9.0)
    latest_full = int(full["year"].max())
    econ = {
        "no_battery_cost_eur": float(priced[
            (priced.capacity_kwh == 0.0) & (priced.year == latest_full)
        ]["cost_eur"].sum()),
        "euro_optimum_kwh": float(euro_opt),
        "annual_saving_eur": float(
            annual.loc[annual.capacity_kwh == euro_opt, "saving_eur"].sum()),
        "breakeven_eur_per_kwh": float(
            annual.loc[annual.capacity_kwh == euro_opt,
                       "break_even_eur_per_kwh"].sum()),
        "derived_threshold_kwh_per_kwh":
            economics.derived_threshold_kwh_per_kwh(
                EXAMPLE_EUR_PER_KWH, HORIZON_YEARS, blended),
        "arbitrage_ceiling_eur_yr": economics.arbitrage_ceiling_eur_yr(
            meter_df, tar, euro_opt or 9.0)["ceiling_eur_yr"],
    }
```

Pass `tariff_bands=tar.bands`, `annual_savings=annual`,
`economics_summary=econ`, `horizon_years=HORIZON_YEARS` into `build_html`;
write `tar.bands.to_csv(out_dir / "meter_tariff_bands.csv", index=False)` and
`saved.to_csv(out_dir / "meter_economics.csv", index=False)`; and merge
`econ` into the returned summary dict.

- [ ] **Step 5: Run everything**

Run: `uv run pytest -q`
Expected: PASS. **203** — 175 today, plus 9 (Task 1), 5 (Task 2), 10
(Task 3), 2 (Task 4) and 2 here. Report the real number from the output if
it differs rather than assuming this one.

Run: `uv run python analyze_meter.py`
Expected: the summary prints with the new euro keys; `out/` gains
`meter_economics.csv` and `meter_tariff_bands.csv`.

- [ ] **Step 6: Update the README**

Add a `## Battery economics under the 2027 tariff` section covering: that net
feed-in is €0.01/kWh so there is no net metering left; the 18–30× spread and
that it is the whole case for storage; the euro-optimal capacity and how it
compares to phase 3's 9.0 kWh energy answer; the break-even curve and how to
read it against a quote; the derived kWh/yr threshold against the removed 50;
the arbitrage ceiling labelled as a bound; and the excluded items — standing
charges, taxes, degradation, discounting — each named rather than omitted.
Quote only numbers you have seen in the output of Step 5.

- [ ] **Step 7: Commit**

```bash
git add src/pvnight/compare_report.py analyze_meter.py README.md \
        tests/test_compare_report.py tests/test_analyze_meter.py \
        out/meter_economics.csv out/meter_tariff_bands.csv
git commit -m "Publish the battery's value in euros"
```

---

## Verification

Before reporting any task complete, invoke `superpowers:verification-before-completion`.
Run `uv run pytest` and paste the real summary line. Do not describe any
number as confirmed unless it appears in output you have seen.

## Notes for the implementer

- **The tariff file is the user's personal data.** It is untracked and not
  git-ignored, like `data/meterdata/`. Never `git add` it, and never use
  `git add -A`.
- **Runtime.** `price_capacities` runs the 61-capacity sweep once over
  233,358 intervals with two extra accumulators. Expect a minute or two. Do
  not coarsen the sweep to speed it up without saying so.
- **Phase 2 and 3 figures must not move.** Task 2's keyword is additive and
  Task 2 Step 6 checks this explicitly. If phase 2's elbow is no longer 8.0
  or `recommend_capacity` no longer 7.5, stop rather than adjusting their
  tests.
- **The spec records three predictions** (§12). Task 3 Step 5 checks them.
  Report each as held or broken; a broken prediction is a finding, not a
  number to reconcile away.
- **If a test fails and you believe the assertion is wrong rather than the
  code**, do not adjust it. Report it with evidence — implementers on this
  project have correctly overturned the plan author more than a dozen times,
  including on the framing of the two battery bounds and on a resolution
  penalty whose sign was an artifact.
