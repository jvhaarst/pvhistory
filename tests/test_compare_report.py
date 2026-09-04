import numpy as np
import pandas as pd
import pytest

from pvnight.compare_report import build_html


@pytest.fixture
def toy():
    d = pd.date_range("2021-01-01", periods=400, freq="D")
    mn = pd.DataFrame({"date": d, "import_kwh": np.linspace(2, 30, 400),
                       "export_kwh": 0.0, "peak_kw": 1.0,
                       "hours_above_2kw": 0.0, "is_ev": False,
                       "missing_intervals": 0, "covered": True})
    pn = pd.DataFrame({"date": d, "night_wh": np.linspace(2, 25, 400) * 1000,
                       "covered": True, "is_ev": False})
    caps = np.arange(0.0, 20.1, 0.5)
    def mk(bound, scale):
        return pd.DataFrame({
            "capacity_kwh": caps, "power_kw": 3.0, "bound": bound,
            "grid_import_kwh_yr": 3000 * scale * np.exp(-caps / 6),
            "pv_export_kwh_yr": 2000 * np.exp(-caps / 8),
            "night_grid_import_kwh_yr": 1500 * scale * np.exp(-caps / 5),
            "nonev_night_grid_import_kwh_yr": 1200 * scale * np.exp(-caps / 5),
            "night_self_sufficiency_pct": 100 * (1 - np.exp(-caps / 5)),
            "nonev_night_self_sufficiency_pct": 100 * (1 - np.exp(-caps / 5)),
            "cycles_per_yr": np.r_[np.nan, 200 / caps[1:]],
            "marginal_kwh_per_kwh": np.r_[np.nan, -np.diff(3000 * scale * np.exp(-caps / 6))],
            "nonev_marginal_kwh_per_kwh": np.r_[np.nan, -np.diff(1200 * scale * np.exp(-caps / 5))],
        })
    ms = pd.concat([mk("charge_first", 1.0), mk("discharge_first", 0.92)], ignore_index=True)
    ps = mk("charge_first", 0.75).drop(columns=["bound"])
    sens = pd.DataFrame({"power_kw": np.repeat([1.5, 2.0, 2.5, 3.0], 3),
                         "hours": np.tile([1.0, 2.0, 3.0], 4),
                         "n_ev": np.arange(12) + 60,
                         "median_ev_kwh": np.linspace(20, 35, 12),
                         "median_rest_kwh": np.linspace(5, 6, 12)})
    mr = pd.DataFrame({"month": pd.period_range("2020-01", "2025-12", freq="M")})
    mr["pv_kwh"] = 100.0
    mr["meter_kwh"] = np.where(mr.month < pd.Period("2022-12"), 102.0, 130.0)
    mr["ratio"] = mr.pv_kwh / mr.meter_kwh
    gaps = pd.DataFrame({
        "gap_start_utc": pd.to_datetime(["2024-01-08T23:00Z"]),
        "gap_end_utc": pd.to_datetime(["2024-01-14T08:45Z"]),
        "missing_intervals": [518]})
    return mn, pn, ms, ps, sens, mr, gaps


def test_page_has_six_charts_and_no_rasters(toy):
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    assert html.count("<svg") == 6
    assert "<image" not in html
    assert "data:image" not in html


def test_page_omits_the_document_wrapper(toy):
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    for tag in ["<!doctype", "<html", "<head>", "<body>"]:
        assert tag not in html.lower()


def test_page_labels_the_superseded_analysis(toy):
    """Side-by-side publication is only safe if the page says which is which."""
    low = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12).lower()
    assert "superseded" in low
    assert "december 2022" in low


def test_page_reports_a_range_across_the_two_bounds(toy):
    low = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12).lower()
    assert "bound" in low
    assert "charge-first" in low or "charge_first" in low
    assert "discharge-first" in low or "discharge_first" in low


def test_page_states_the_measured_resolution_penalty(toy):
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    assert "4.2" in html
    assert "15-minute" in html or "15 minute" in html


def test_page_discloses_the_january_2024_exclusions(toy):
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    assert "12" in html
    assert "January 2024" in html or "2024-01" in html


def test_page_calls_the_ev_rule_a_heuristic(toy):
    low = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12).lower()
    assert "heuristic" in low


def test_gap_prose_follows_the_data_not_a_hardcoded_string(toy):
    """The report's factual claims must move with the frame they describe."""
    mn, pn, ms, ps, sens, mr, gaps = toy
    two = pd.DataFrame({
        "gap_start_utc": pd.to_datetime(["2025-03-04T01:00Z", "2025-03-09T02:00Z"]),
        "gap_end_utc": pd.to_datetime(["2025-03-04T04:00Z", "2025-03-09T05:00Z"]),
        "missing_intervals": [12, 12]})
    html = build_html(mn, pn, ms, ps, sens, mr, two,
                      resolution_penalty_pct=4.2, excluded_nights=3)
    assert "March 2025" in html
    assert "seven gaps" not in html.lower()
    assert "january 2024" not in html.lower()


def test_build_html_survives_an_empty_nights_frame(toy):
    mn, pn, ms, ps, sens, mr, gaps = toy
    empty = mn.iloc[0:0]
    html = build_html(empty, pn, ms, ps, sens, mr, gaps,
                      resolution_penalty_pct=4.2, excluded_nights=0)
    assert html.count("<svg") == 6
