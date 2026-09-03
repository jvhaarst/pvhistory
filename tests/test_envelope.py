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
