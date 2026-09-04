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
