import numpy as np
import pandas as pd
import pytest

from pvnight import compare_report
from pvnight.battery import elbow_capacity
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
    # A bare `"12" in html` passed on the capacity axis and on "Capped at
    # 12 kWh"; it never touched the exclusion count. Assert the count in the
    # sentence that actually reports it.
    html = build_html(*toy, resolution_penalty_pct=4.2, excluded_nights=12)
    assert "<strong>12</strong> nights from this analysis" in html
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


def _stability(elbows):
    """A stability table with a chosen elbow per truncation."""
    tops = [10.0, 15.0, 20.0, 25.0, 30.0][-len(elbows):]
    return pd.DataFrame({"sweep_top_kwh": tops, "elbow_kwh": elbows,
                         "top_end_marginal": [1.5] * len(elbows)})


def test_stability_caption_says_settled_only_when_the_last_two_rows_agree(monkeypatch):
    """The caption is derived from the table, not asserted beside it.

    The original sentence claimed the elbow stops moving once the top-end
    marginal goes flat, printed above a table where it moved on the very last
    row. Both branches are exercised here because a caption that is only ever
    checked against one data shape is how that defect survived.
    """
    monkeypatch.setattr(compare_report, "elbow_stability",
                        lambda *a, **k: _stability([8.0, 8.0]))
    settled = compare_report._elbow_stability_table(pd.DataFrame())
    assert "stopped moving with the sweep" in settled
    assert "not</strong> settled" not in settled

    monkeypatch.setattr(compare_report, "elbow_stability",
                        lambda *a, **k: _stability([8.5, 9.0]))
    moving = compare_report._elbow_stability_table(pd.DataFrame())
    assert "not</strong> settled" in moving
    assert "a longer sweep reads higher" in moving
    assert "8.5" in moving and "9.0" in moving


def test_the_headline_tile_discloses_an_unsettled_elbow_either_way(monkeypatch, toy):
    """The disclosure must survive a non-degenerate bracket.

    It was first written into the equal-elbows branch only, so a data shape
    where the two bounds disagree would silently drop a warning the code had
    already computed.
    """
    monkeypatch.setattr(compare_report, "elbow_stability",
                        lambda *a, **k: _stability([8.5, 9.0]))

    # Bend the discharge-first curve so the two bounds elbow apart. The stock
    # fixture is degenerate (both 7.0), which is precisely why the original
    # bug was invisible: the disclosure lived in the equal-elbows branch and
    # every test took that branch.
    args = list(toy)
    ms = args[2].copy()
    sel = ms["bound"] == "discharge_first"
    col = "nonev_night_grid_import_kwh_yr"
    ms.loc[sel, col] = (ms.loc[sel, col].to_numpy()
                        * np.exp(-ms.loc[sel, "capacity_kwh"].to_numpy() / 10.0))
    args[2] = ms
    d = ms[(ms["bound"] == "discharge_first") & (ms["power_kw"] == 3.0)]
    assert elbow_capacity(d, power_kw=3.0) != 7.0, "fixture no longer splits"

    html = build_html(*args, resolution_penalty_pct=0.0, excluded_nights=1)
    # Anchor to the TILE's own caption. Asserting the bare phrase matched the
    # stability card instead, and passed against the unfixed code.
    assert ("nameplate capacity, range across the charge-first / "
            "discharge-first bounds, low end — still climbing where the "
            "sweep stops") in html
