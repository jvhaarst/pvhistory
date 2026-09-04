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
