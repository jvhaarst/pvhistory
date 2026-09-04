import numpy as np
import pandas as pd
import pytest

from pvnight.battery import (
    benefit_share_pct,
    convergence_readings,
    build_masks,
    elbow_capacity,
    elbow_stability,
    net_wh,
    recommend_capacity,
    sweep,
)


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


def test_recommend_falls_back_when_the_curve_never_meets_the_threshold():
    """A curve entirely below the threshold means even the first kWh does not
    pay. Recommend the smallest capacity rather than silently the largest."""
    df = pd.DataFrame({
        "capacity_kwh": [0.0, 1.0, 2.0],
        "power_kw": 3.0,
        "marginal_kwh_per_kwh": np.nan,
        "nonev_marginal_kwh_per_kwh": [np.nan, 10.0, 5.0],
    })
    assert recommend_capacity(df, power_kw=3.0, threshold_kwh_per_kwh=50.0) == 1.0


def test_recommend_returns_the_largest_capacity_when_return_never_falls_off():
    """If the marginal return is still above the threshold at the top of the
    sweep, the sweep was too narrow — return its maximum rather than inventing
    a capacity beyond what was simulated."""
    df = pd.DataFrame({
        "capacity_kwh": [0.0, 1.0, 2.0],
        "power_kw": 3.0,
        "marginal_kwh_per_kwh": np.nan,
        "nonev_marginal_kwh_per_kwh": [np.nan, 900.0, 800.0],
    })
    assert recommend_capacity(df, power_kw=3.0, threshold_kwh_per_kwh=50.0) == 2.0


def test_benefit_share_runs_from_zero_to_one_hundred():
    """Share of the benefit an infinitely large battery could deliver.
    Threshold-free: it needs no judgement call, only the curve's endpoints."""
    s = pd.DataFrame({
        "capacity_kwh": [0.0, 1.0, 2.0, 3.0],
        "power_kw": 3.0,
        "nonev_night_grid_import_kwh_yr": [100.0, 60.0, 30.0, 20.0],
    })
    out = benefit_share_pct(s, power_kw=3.0)
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[-1] == pytest.approx(100.0)
    # 100 -> 60 is 40 of the 80 total achievable
    assert out.iloc[1] == pytest.approx(50.0)


def test_elbow_finds_the_bend_without_any_threshold():
    """Maximum perpendicular distance from the chord joining the endpoints.
    On an L-shaped curve the elbow is the corner."""
    s = pd.DataFrame({
        "capacity_kwh": [0.0, 1.0, 2.0, 3.0, 4.0],
        "power_kw": 3.0,
        "nonev_night_grid_import_kwh_yr": [100.0, 40.0, 20.0, 14.0, 10.0],
    })
    assert elbow_capacity(s, power_kw=3.0) == 1.0


def test_elbow_is_flat_curve_safe():
    """A straight line has no bend; return the largest capacity rather than
    an arbitrary interior point."""
    s = pd.DataFrame({
        "capacity_kwh": [0.0, 1.0, 2.0],
        "power_kw": 3.0,
        "nonev_night_grid_import_kwh_yr": [100.0, 50.0, 0.0],
    })
    assert elbow_capacity(s, power_kw=3.0) == 2.0


def test_elbow_and_benefit_share_on_the_real_sweep(real_sweep):
    """Measured: the parameter-free elbow lands at 8.0 kWh, half a step above
    the 50 kWh/yr threshold's 7.5, and 7.5 kWh captures 84.2% of everything an
    infinite battery could achieve."""
    assert elbow_capacity(real_sweep, power_kw=3.0) == pytest.approx(8.0)
    share = benefit_share_pct(real_sweep, power_kw=3.0)
    d = real_sweep[real_sweep.power_kw == 3.0].sort_values("capacity_kwh").reset_index(drop=True)
    at_7_5 = float(share[np.isclose(d.capacity_kwh, 7.5)].iloc[0])
    assert at_7_5 == pytest.approx(84.2, abs=0.5)


@pytest.fixture(scope="session")
def real_sweep():
    """The committed sweep, so the threshold-free figures are pinned to data."""
    return pd.read_csv("out/battery_sweep.csv")


def test_elbow_stability_reports_its_own_sweep_dependence():
    """The elbow is not parameter-free: it drifts with where the sweep is
    truncated, settling only once the top end is flat. The report must be
    able to show that rather than claim independence it does not have."""
    s = pd.DataFrame({
        "capacity_kwh": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "power_kw": 3.0,
        "nonev_night_grid_import_kwh_yr": [100.0, 40.0, 22.0, 16.0, 13.0, 12.0],
    })
    out = elbow_stability(s, power_kw=3.0, tops=(2.0, 3.0, 5.0))
    assert list(out.columns) == ["sweep_top_kwh", "elbow_kwh", "top_end_marginal"]
    assert len(out) == 3
    assert out["elbow_kwh"].notna().all()


def test_elbow_stability_on_the_real_sweep(real_sweep):
    """Measured: 6.0 kWh on a 0-10 sweep, settling to 8.0 by 0-25."""
    out = elbow_stability(real_sweep, power_kw=3.0, tops=(10.0, 20.0, 25.0, 30.0))
    got = dict(zip(out.sweep_top_kwh, out.elbow_kwh))
    assert got[10.0] == pytest.approx(6.0)
    assert got[25.0] == pytest.approx(8.0)
    assert got[30.0] == pytest.approx(8.0)


def test_convergence_readings_names_each_method_and_its_answer():
    """No single geometric reading is authoritative; the table exists to show
    they disagree, and by how little."""
    s = pd.DataFrame({
        "capacity_kwh": np.arange(0.0, 6.1, 0.5),
        "power_kw": 3.0,
        "nonev_night_grid_import_kwh_yr": 100 * np.exp(-np.arange(0.0, 6.1, 0.5) / 2),
    })
    s["nonev_marginal_kwh_per_kwh"] = -s.nonev_night_grid_import_kwh_yr.diff() / s.capacity_kwh.diff()
    out = convergence_readings(s, power_kw=3.0)
    assert list(out.columns) == ["method", "capacity_kwh", "what_it_measures"]
    assert len(out) == 5
    assert out["capacity_kwh"].notna().all()
    assert out["method"].is_unique


def test_convergence_readings_on_the_real_sweep(real_sweep):
    """Measured: the five readings land between 2.5 and 9.0, with the four
    candidate sizes clustered in 6.5-9.0."""
    out = convergence_readings(real_sweep, power_kw=3.0).set_index("method")
    got = out["capacity_kwh"]
    assert got["inflection of the import curve"] == pytest.approx(2.5)
    assert got["steepest collapse of marginal return"] == pytest.approx(6.5)
    assert got["inflection of the marginal curve"] == pytest.approx(7.0)
    assert got["elbow, full sweep"] == pytest.approx(8.0)
    assert got["elbow, measured from the inflection"] == pytest.approx(9.0)
    candidates = got.drop("inflection of the import curve")
    assert candidates.min() == pytest.approx(6.5)
    assert candidates.max() == pytest.approx(9.0)
