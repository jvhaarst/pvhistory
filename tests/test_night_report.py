import numpy as np
import pandas as pd
import pytest

from pvnight.night_report import build_html


@pytest.fixture
def toy():
    nights = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=60, freq="D"),
        "night_wh": np.linspace(1000, 30000, 60),
        "peak_w": np.linspace(300, 9000, 60),
        "hours_above_2kw": np.linspace(0, 4, 60),
        "is_ev": [False] * 55 + [True] * 5,
        "covered": True,
        "coverage": 1.0,
    })
    caps = np.arange(0.0, 20.1, 1.0)
    sweep = pd.DataFrame({
        "capacity_kwh": np.tile(caps, 2),
        "power_kw": np.repeat([3.0, 3.7], len(caps)),
        "grid_import_kwh_yr": np.tile(3000 * np.exp(-caps / 6), 2),
        "pv_export_kwh_yr": np.tile(2000 * np.exp(-caps / 8), 2),
        "night_grid_import_kwh_yr": np.tile(1500 * np.exp(-caps / 5), 2),
        "nonev_night_grid_import_kwh_yr": np.tile(1200 * np.exp(-caps / 5), 2),
        "night_self_sufficiency_pct": np.tile(100 * (1 - np.exp(-caps / 5)), 2),
        "nonev_night_self_sufficiency_pct": np.tile(100 * (1 - np.exp(-caps / 5)), 2),
        "cycles_per_yr": np.tile(np.r_[0, 200 / caps[1:]], 2),
        "marginal_kwh_per_kwh": np.tile(np.r_[np.nan, np.diff(3000 * np.exp(-caps / 6)) * -1], 2),
        "nonev_marginal_kwh_per_kwh": np.tile(np.r_[np.nan, np.diff(1200 * np.exp(-caps / 5)) * -1], 2),
    })
    monthly = pd.DataFrame({
        "month": range(1, 13),
        "gen_kwh": [2.2, 4.6, 13.5, 25.2, 28.3, 32.7, 27.3, 24.9, 15.5, 5.7, 2.9, 1.7],
        "day_cons_kwh": [5.9, 5.8, 6.0, 7.5, 7.2, 8.0, 7.0, 6.6, 6.5, 5.9, 4.8, 5.4],
        "surplus_kwh": [-3.4, -0.3, 6.8, 15.9, 21.3, 23.9, 19.4, 17.4, 9.3, 0.0, -2.2, -3.5],
        "night_kwh": [9.1, 6.3, 5.5, 3.9, 2.8, 2.8, 2.8, 3.9, 5.0, 5.9, 8.3, 9.3],
        "above_cap_pct_ev": [30, 28, 25, 20, 18, 15, 16, 19, 22, 27, 31, 33],
        "above_cap_pct_nonev": [1.6, 1.5, 1.4, 1.2, 1.0, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.8],
        "discharge_kwh": [5, 20, 90, 160, 190, 200, 185, 170, 120, 30, 6, 3],
    })
    sens = pd.DataFrame({
        "power_w": np.repeat([1500.0, 2000.0, 2500.0, 3000.0], 3),
        "hours": np.tile([1.0, 2.0, 3.0], 4),
        "n_ev": [210, 80, 50, 200, 72, 42, 150, 65, 38, 120, 55, 30],
        "median_ev_kwh": np.linspace(12, 30, 12),
        "median_rest_kwh": np.linspace(4.4, 4.9, 12),
    })
    return nights, sweep, monthly, sens


def test_page_has_a_title_and_eight_charts(toy):
    html = build_html(*toy, recommended_kwh=7.0, coverable_pct=48.2,
                       negative_surplus_months=[11, 12, 1])
    assert "<title>" in html
    assert html.count("<svg") == 8


def test_page_omits_the_document_wrapper(toy):
    html = build_html(*toy, recommended_kwh=7.0, coverable_pct=48.2,
                       negative_surplus_months=[11, 12, 1])
    for tag in ["<!doctype", "<html", "<head>", "<body>"]:
        assert tag not in html.lower()


def test_page_embeds_no_raster_images(toy):
    """An embedded PNG trips the Artifact publish content scanner."""
    html = build_html(*toy, recommended_kwh=7.0, coverable_pct=48.2,
                       negative_surplus_months=[11, 12, 1])
    assert "<image" not in html
    assert "data:image" not in html


def test_page_defines_light_and_dark_palettes(toy):
    html = build_html(*toy, recommended_kwh=7.0, coverable_pct=48.2,
                       negative_surplus_months=[11, 12, 1])
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="dark"]' in html
    assert '[data-theme="light"]' in html


def test_page_states_the_winter_finding(toy):
    """The single most important caveat: no battery charges in midwinter."""
    html = build_html(*toy, recommended_kwh=7.0, coverable_pct=48.2,
                       negative_surplus_months=[11, 12, 1]).lower()
    assert "winter" in html
    assert "negative" in html or "cannot" in html


def test_page_labels_the_ev_rule_as_a_heuristic(toy):
    html = build_html(*toy, recommended_kwh=7.0, coverable_pct=48.2,
                       negative_surplus_months=[11, 12, 1]).lower()
    assert "heuristic" in html


def test_winter_prose_follows_the_data_not_a_hardcoded_string(toy):
    """The winter finding is the analysis's most important caveat. If the
    dataset is ever refreshed, the prose must move with it."""
    html = build_html(*toy, recommended_kwh=7.0, coverable_pct=61.5,
                       negative_surplus_months=[12])
    assert "61.5" in html
    assert "48.2" not in html
    assert "December" in html
    assert "November" not in html


def test_monthly_charts_do_not_trust_row_order(toy):
    nights, sweep, monthly, sens = toy
    shuffled = monthly.sample(frac=1.0, random_state=0).iloc[:9]
    html = build_html(nights, sweep, shuffled, sens, recommended_kwh=7.0,
                       coverable_pct=48.2, negative_surplus_months=[11, 12, 1])
    assert html.count("<svg") == 8


def test_marginal_caption_describes_the_last_crossing_not_the_first(toy):
    """The curve rises before it falls, so "first capacity below the
    threshold" is both wrong and degenerate. The page must not say it."""
    html = build_html(*toy, recommended_kwh=7.5, coverable_pct=48.2,
                      negative_surplus_months=[11, 12, 1])
    low = html.lower()
    assert "first capacity" not in low
    assert "never" in low and "judgement" in low
