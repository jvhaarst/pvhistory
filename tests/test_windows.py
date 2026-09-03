import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pvnight import envelope as env
from pvnight.config import SITE_TZ
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


def test_night_duration_is_immune_to_dst_transitions(pipeline):
    """Night is real elapsed time between two astronomical events, so a
    clock change cannot alter it — the sequence runs smooth through both
    transitions. Naive local wall-clock arithmetic, by contrast, injects a
    spurious +1h at spring-forward and -1h at fall-back. That contrast is
    why every duration in this table is a UTC subtraction.
    """
    _, _, w = pipeline
    ws = w.set_index("date")

    spring = [ws.loc[dt.date(2023, 3, d), "night_duration_h"] for d in (23, 24, 25, 26)]
    assert all(-0.2 < s < 0 for s in np.diff(spring)), f"step across spring-forward: {spring}"

    autumn = [ws.loc[dt.date(2023, 10, d), "night_duration_h"] for d in (27, 28, 29)]
    assert all(0 < s < 0.2 for s in np.diff(autumn)), f"step across fall-back: {autumn}"

    row = ws.loc[dt.date(2023, 3, 25)]
    start = pd.Timestamp(row["night_start_utc"]).tz_convert(SITE_TZ).tz_localize(None)
    end = pd.Timestamp(row["night_end_utc"]).tz_convert(SITE_TZ).tz_localize(None)
    naive_h = (end - start).total_seconds() / 3600
    assert naive_h == pytest.approx(row["night_duration_h"] + 1.0, abs=0.05)


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


def test_window_contains_observed_first_light_on_93_percent_of_days(pipeline):
    """The core regression guard. Measured containment is 94.77%: a pooled
    +/-10-day percentile can't hit exactly 95% per day because of seasonal
    drift inside the window, so the floor is set below the measurement with
    margin rather than at the nominal 95%. A materially lower figure means
    the fit has drifted."""
    _, ev, w = pipeline
    m = w.merge(ev, left_on="date", right_on="solar_date")
    assert (m["first_light_utc"] >= m["solar_start_utc"]).mean() >= 0.93


def test_window_contains_observed_last_light_on_93_percent_of_days(pipeline):
    """Measured containment is 94.19%; see test_..._first_light_... above."""
    _, ev, w = pipeline
    m = w.merge(ev, left_on="date", right_on="solar_date")
    assert (m["last_light_utc"] <= m["solar_end_utc"]).mean() >= 0.93


def test_window_contains_both_ends_on_88_percent_of_days(pipeline):
    """Measured 89.30%. Two independent ~94% endpoints multiply out to
    roughly 89%, so this is the honest joint figure, guarded with margin."""
    _, ev, w = pipeline
    m = w.merge(ev, left_on="date", right_on="solar_date")
    both = (m["first_light_utc"] >= m["solar_start_utc"]) & (m["last_light_utc"] <= m["solar_end_utc"])
    assert both.mean() >= 0.88


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


def test_end_offset_uses_sunset_not_sunrise(pipeline):
    """Guards a copy-paste slip in the structurally near-identical offset
    columns. Measured range is -6.712 .. +0.890 min; computing this against
    sunrise instead of sunset would put it hours away."""
    _, _, w = pipeline
    assert w["end_offset_min"].between(-30, 60).all()
    assert w["end_offset_min"].notna().all()


def test_window_opens_before_sunrise_and_closes_near_sunset(pipeline):
    """The panels wake on diffuse light slightly before geometric sunrise,
    so the window always opens early; it closes within a minute of sunset."""
    _, _, w = pipeline
    assert (w["solar_start_utc"] <= w["sunrise_utc"]).all()
    assert (w["solar_end_utc"] <= w["sunset_utc"] + pd.Timedelta(minutes=1)).all()


def test_evening_threshold_sits_above_the_morning_threshold(pipeline):
    """Spec fact 8 as a hard invariant: measured theta_start -1.628..-1.036,
    theta_end -0.965..-0.148, so the evening threshold is higher on every
    day of the table. A sign error or a swapped pair would break this."""
    _, _, w = pipeline
    assert w["theta_start_deg"].notna().all()
    assert w["theta_end_deg"].notna().all()
    assert w["theta_start_deg"].abs().max() < 3.0
    assert w["theta_end_deg"].abs().max() < 3.0
    assert (w["theta_end_deg"] > w["theta_start_deg"]).all()


def test_both_local_columns_carry_an_explicit_offset(pipeline):
    """solar_start_local was already covered; solar_end_local was not."""
    _, _, w = pipeline
    assert w["solar_end_local"].str.contains(r"\+0[12]:00").all()


def test_every_date_draws_on_a_full_pooled_window(pipeline):
    """Measured 96..120 samples per date. A collapse here would mean the
    circular pooling window stopped wrapping the year boundary."""
    _, _, w = pipeline
    assert (w["n_samples"] >= 90).all()
