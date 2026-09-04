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


def test_charge_first_never_gives_a_worse_result_than_discharge_first():
    """The two orderings bracket what a battery could bridge within an
    interval, so the favourable ordering must never import more."""
    m = _toy_meter()
    n = pd.DataFrame([{
        "date": m.ts_utc.iloc[0].date(),
        "night_start_utc": m.ts_utc.iloc[0], "night_end_utc": m.ts_utc.iloc[-1],
        "is_ev": False, "covered": True,
    }])
    out = sweep_bounds(m, n, np.arange(0.0, 6.1, 1.0), power_kws=(3.0,))
    charge_first = out[out.bound == "charge_first"].sort_values("capacity_kwh")
    discharge_first = out[out.bound == "discharge_first"].sort_values("capacity_kwh")
    assert (charge_first.grid_import_kwh_yr.to_numpy()
            <= discharge_first.grid_import_kwh_yr.to_numpy() + 1e-9).all()


def test_both_orderings_reproduce_the_measured_import_at_zero_capacity():
    """The property that makes these a bracket rather than two different
    questions. A formulation that collapses each interval to export-minus-
    import fails this: it reports only the residual, understating the
    baseline the house actually paid for."""
    m = _toy_meter()
    n = pd.DataFrame([{
        "date": m.ts_utc.iloc[0].date(),
        "night_start_utc": m.ts_utc.iloc[0], "night_end_utc": m.ts_utc.iloc[-1],
        "is_ev": False, "covered": True,
    }])
    out = sweep_bounds(m, n, np.array([0.0]), power_kws=(3.0,))
    measured_kwh = m["import_kwh"].sum()
    years = (m.ts_utc.iloc[-1] - m.ts_utc.iloc[0]).total_seconds() / (365.25 * 24 * 3600)
    for bound in ("charge_first", "discharge_first"):
        got = out.loc[out.bound == bound, "grid_import_kwh_yr"].iloc[0] * years
        assert got == pytest.approx(measured_kwh, rel=1e-6)


def test_sweep_bounds_labels_both_bounds():
    m = _toy_meter()
    n = pd.DataFrame([{
        "date": m.ts_utc.iloc[0].date(),
        "night_start_utc": m.ts_utc.iloc[0], "night_end_utc": m.ts_utc.iloc[-1],
        "is_ev": False, "covered": True,
    }])
    out = sweep_bounds(m, n, np.array([0.0, 5.0]), power_kws=(3.0,))
    assert set(out["bound"].unique()) == {"charge_first", "discharge_first"}
    assert len(out) == 4


def test_charge_first_and_discharge_first_present_flows_as_separate_steps():
    """No single number can express "charge 0.3 and discharge 0.5", so
    neither ordering tries: each interval becomes two steps, in opposite
    order for the two bounds."""
    m = pd.DataFrame({
        "ts_utc": pd.date_range("2023-06-01", periods=3, freq="15min", tz="UTC"),
        "import_kwh": [0.0, 0.5, 0.2],
        "export_kwh": [0.4, 0.3, 0.0],
    })
    sigs = bound_signals(m)
    charge_first, cf_idx = sigs["charge_first"]
    discharge_first, df_idx = sigs["discharge_first"]

    assert len(charge_first) == 6
    assert len(discharge_first) == 6
    assert charge_first.tolist() == pytest.approx([400.0, 0.0, 300.0, -500.0, 0.0, -200.0])
    assert discharge_first.tolist() == pytest.approx([0.0, 400.0, -500.0, 300.0, -200.0, 0.0])
    assert cf_idx.tolist() == [0, 0, 1, 1, 2, 2]
    assert df_idx.tolist() == [0, 0, 1, 1, 2, 2]
