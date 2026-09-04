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


def test_a_mid_day_energy_gap_is_not_silently_zeroed():
    """A NaN energy reading is unrecorded, not zero, and the real increment
    on the row after the gap must not be discarded either."""
    s = _samples([
        ("2023-06-01T20:00", dt.date(2023, 6, 1), 100.0, 0.0),
        ("2023-06-01T20:05", dt.date(2023, 6, 1), 100.0, 100.0),
        ("2023-06-01T20:10", dt.date(2023, 6, 1), 100.0, np.nan),
        ("2023-06-01T20:15", dt.date(2023, 6, 1), 100.0, 250.0),
    ])
    inc = consumption_increments(s)
    assert inc.iloc[0] == 0.0            # first of day, artefact of diff()
    assert inc.iloc[1] == 100.0
    assert np.isnan(inc.iloc[2])         # genuine gap, NOT zero
    assert np.isnan(inc.iloc[3])         # increment across the gap is unknown

    w = _one_night_window("2023-06-01T20:00", "2023-06-01T20:20")
    out = summarise_nights(s, w, min_coverage=0.0)
    assert out.loc[0, "missing_increments"] == 2
    assert not bool(out.loc[0, "covered"])   # a gap disqualifies the night


@pytest.fixture(scope="session")
def real_nights(loaded, repo_root):
    w = pd.read_csv(
        repo_root / "out" / "solar_windows.csv",
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


def test_the_null_safe_fix_does_not_move_the_published_figures(real_nights):
    """Every energy_cons_wh null in this dataset coincides with a
    power_cons_w null, so the stricter rule is inert here. Pin that, so a
    future data drop that breaks the correlation shows up as a test failure
    rather than as a silently different answer."""
    c = real_nights[real_nights["covered"]]
    c = c[(c["date"] >= "2020-05-20") & (c["date"] <= "2025-12-30")]
    assert len(c) == 2031
    assert c["night_wh"].median() / 1000 == pytest.approx(4.85, abs=0.05)
    assert int(c["is_ev"].sum()) == 72


def test_the_ev_heuristic_has_a_known_false_positive_rate(real_nights):
    """Spec fact 7 says the car arrived in 2022, measured with a stricter
    ">=1h above 5 kW" rule. The rule `is_ev` actually implements (">=2h above
    2 kW", fact 6) also catches 6 pre-2022 winter evenings — a 3.5-5 kW load
    for about two hours, plausibly a heat pump or a dryer. None of them shows
    the fast-charge signature, and they are materially smaller than real EV
    nights, so the heuristic's imperfection is bounded and visible rather
    than tuned away."""
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


def test_ev_sensitivity_grid_has_a_row_per_combination(loaded, repo_root):
    w = pd.read_csv(
        repo_root / "out" / "solar_windows.csv",
        parse_dates=["date", "night_start_utc", "night_end_utc"],
    )
    g = ev_sensitivity(loaded, w)
    assert len(g) == 12
    assert set(g.columns) == {"power_w", "hours", "n_ev", "median_ev_kwh", "median_rest_kwh"}
    chosen = g[(g.power_w == 2000.0) & (g.hours == 2.0)].iloc[0]
    assert chosen.n_ev == 72
