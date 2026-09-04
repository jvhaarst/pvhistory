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


def test_uncovered_nights_are_excluded_from_the_masks():
    """`covered` is the contract that a night's energy is gap-free. If
    build_masks ignored it, gap-corrupted nights would re-enter the night
    metrics through the back door."""
    s = pd.DataFrame({
        "ts_utc": pd.to_datetime(
            ["2023-06-01T21:00", "2023-06-02T21:00"], utc=True
        ),
        "power_gen_w": [0.0, 0.0],
        "power_cons_w": [100.0, 100.0],
    })
    n = pd.DataFrame([
        {"date": pd.Timestamp("2023-06-01"),
         "night_start_utc": pd.Timestamp("2023-06-01T20:00", tz="UTC"),
         "night_end_utc": pd.Timestamp("2023-06-01T23:00", tz="UTC"),
         "is_ev": False, "covered": False},
        {"date": pd.Timestamp("2023-06-02"),
         "night_start_utc": pd.Timestamp("2023-06-02T20:00", tz="UTC"),
         "night_end_utc": pd.Timestamp("2023-06-02T23:00", tz="UTC"),
         "is_ev": False, "covered": True},
    ])
    night, nonev, _ = build_masks(s, n)
    assert night.tolist() == [False, True]
    assert nonev.tolist() == [False, True]


def test_sweep_survives_a_frame_where_every_night_is_uncovered(recwarn):
    """With no covered night there is no night-time deficit, so the
    self-sufficiency denominators are zero. That must yield NaN deliberately,
    not a RuntimeWarning and an incidental NaN."""
    idx = pd.date_range("2023-06-01", periods=500, freq="5min", tz="UTC")
    s = pd.DataFrame({"ts_utc": idx, "power_gen_w": 1000.0, "power_cons_w": 400.0})
    n = pd.DataFrame([{
        "date": idx[0].date(),
        "night_start_utc": idx[0], "night_end_utc": idx[-1],
        "is_ev": False, "covered": False,
    }])
    out = sweep(s, n, capacities_kwh=np.array([0.0, 5.0]), power_kws=(3.0,))
    assert len(out) == 2
    assert out["night_self_sufficiency_pct"].isna().all()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


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


def test_recommend_returns_the_capacity_beyond_which_return_never_recovers():
    df = pd.DataFrame({
        "capacity_kwh": [0.0, 1.0, 2.0, 3.0, 4.0],
        "power_kw": 3.0,
        "marginal_kwh_per_kwh": [np.nan, 900.0, 800.0, 700.0, 600.0],
        "nonev_marginal_kwh_per_kwh": [np.nan, 200.0, 120.0, 40.0, 10.0],
    })
    assert recommend_capacity(df, power_kw=3.0, threshold_kwh_per_kwh=50.0) == 3.0
    assert recommend_capacity(df, power_kw=3.0, threshold_kwh_per_kwh=50.0,
                              scenario="all") == 4.0


def test_recommend_ignores_a_low_marginal_return_at_tiny_capacity():
    """The real curve rises before it falls: a 0.5 kWh battery empties within
    minutes of sunset, so its first half-kWh buys little, and marginal value
    peaks near 2.5 kWh. A first-crossing rule returns 0.5 here, which is
    degenerate. The answer must be 4.0 — the first capacity past the last one
    still earning its keep."""
    df = pd.DataFrame({
        "capacity_kwh": [0.0, 0.5, 1.5, 2.5, 3.5, 4.0, 5.0],
        "power_kw": 3.0,
        "marginal_kwh_per_kwh": np.nan,
        "nonev_marginal_kwh_per_kwh": [np.nan, 36.8, 113.3, 134.0, 60.0, 41.3, 13.6],
    })
    assert recommend_capacity(df, power_kw=3.0, threshold_kwh_per_kwh=50.0) == 4.0
