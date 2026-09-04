import datetime as dt

import pytest

from pvnight import envelope as env
from pvnight.events import first_last_light
from pvnight.report import build_html


@pytest.fixture(scope="session")
def artifacts(loaded):
    ev = env.attach_solar(first_last_light(loaded))
    model = env.fit(ev)
    windows = env.build_windows(
        model, dt.date(2023, 1, 1), dt.date(2023, 12, 31), set(ev["solar_date"])
    )
    return model, ev, windows


def test_html_has_a_title_and_five_charts(artifacts, loaded):
    model, ev, w = artifacts
    html = build_html(model, ev, w, loaded)
    assert "<title>" in html
    assert html.count("<svg") == 5


def test_html_omits_the_document_wrapper(artifacts, loaded):
    """The Artifact host supplies doctype, html, head and body."""
    model, ev, w = artifacts
    html = build_html(model, ev, w, loaded)
    for tag in ["<!doctype", "<html", "<head>", "<body>"]:
        assert tag not in html.lower()


def test_html_defines_light_and_dark_palettes(artifacts, loaded):
    model, ev, w = artifacts
    html = build_html(model, ev, w, loaded)
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="dark"]' in html
    assert '[data-theme="light"]' in html


def test_azimuth_chart_carries_the_season_confounding_caveat(artifacts, loaded):
    """Spec fact 7 requires this chart be labelled a shading screen, not a
    horizon profile, so no reader mistakes the winter bins for obstruction."""
    model, ev, w = artifacts
    html = build_html(model, ev, w, loaded)
    assert "season" in html.lower()
