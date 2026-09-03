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
