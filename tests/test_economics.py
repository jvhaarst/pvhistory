import numpy as np
import pandas as pd
import pytest

from pvnight import economics
from pvnight.tariff import Tariff


@pytest.fixture
def toy_tariff():
    """Two bands at round prices, so euro answers are checkable by hand."""
    bands = pd.DataFrame({
        "band": ["Peak", "Off"],
        "levering_eur_kwh": [0.30, 0.20],
        "terugleverkosten_eur_kwh": [0.07, 0.05],
        "vergoeding_eur_kwh": [0.08, 0.06],
    })
    hour_map = np.zeros((2, 24), dtype=int)
    hour_map[:, 0:12] = 1          # Off overnight and morning
    hour_map[:, 12:24] = 0         # Peak afternoon and evening
    return Tariff(bands=bands, hour_map=hour_map)


def test_net_export_is_the_difference_not_the_headline_rate(toy_tariff):
    assert toy_tariff.net_export_eur_kwh == pytest.approx([0.01, 0.01])


def test_break_even_inverts_payback():
    """At the break-even price, payback must equal the horizon exactly."""
    price = economics.break_even_eur_per_kwh(600.0, capacity_kwh=9.0, years=10.0)
    assert economics.payback_years(600.0, 9.0, price, 0.0) == pytest.approx(10.0)


def test_break_even_is_saving_times_years_over_capacity():
    assert economics.break_even_eur_per_kwh(450.0, 9.0, 10.0) == pytest.approx(500.0)


def test_payback_includes_the_fixed_cost():
    assert economics.payback_years(500.0, 10.0, 400.0, 1500.0) == pytest.approx(11.0)


def test_recommend_picks_the_capacity_with_the_best_net_position():
    """Net position, not the largest saving: a bigger battery always saves
    more energy, so an unpriced rule would recommend the largest one swept."""
    annual = pd.DataFrame({
        "capacity_kwh": [5.0, 10.0, 20.0],
        "saving_eur": [400.0, 600.0, 650.0],
    })
    # over 10 years at 300 EUR/kWh: 4000-1500=2500, 6000-3000=3000, 6500-6000=500
    assert economics.recommend_capacity_eur(annual, 300.0, 0.0, 10.0) == 10.0


def test_recommend_returns_zero_when_nothing_pays_back():
    annual = pd.DataFrame({"capacity_kwh": [5.0, 10.0],
                           "saving_eur": [50.0, 60.0]})
    assert economics.recommend_capacity_eur(annual, 900.0, 0.0, 10.0) == 0.0


def test_derived_threshold_replaces_the_invented_fifty():
    """EUR 500/kWh over 10 years at EUR 0.28/kWh blended needs ~179 kWh/yr."""
    got = economics.derived_threshold_kwh_per_kwh(500.0, 10.0, 0.28)
    assert got == pytest.approx(500.0 / (10.0 * 0.28))
    assert got > 50.0


def _toy_meter(hours=48):
    """Alternating export by day and import by night, on the toy bands."""
    ts = pd.date_range("2024-06-01", periods=hours * 4, freq="15min", tz="UTC")
    local_hour = ts.tz_convert("Europe/Amsterdam").hour
    exporting = local_hour >= 12
    return pd.DataFrame({
        "ts_utc": ts,
        "import_kwh": np.where(exporting, 0.0, 0.25),
        "export_kwh": np.where(exporting, 0.5, 0.0),
    })


def _toy_nights(meter_df):
    d = meter_df["ts_utc"].dt.tz_convert("Europe/Amsterdam").dt.date
    rows = []
    for day in sorted(set(d)):
        base = pd.Timestamp(day, tz="Europe/Amsterdam").tz_convert("UTC")
        rows.append({"date": pd.Timestamp(day), "night_start_utc": base,
                     "night_end_utc": base + pd.Timedelta(hours=6),
                     "is_ev": False, "covered": True})
    return pd.DataFrame(rows)


def test_zero_capacity_reproduces_the_bill_computed_straight_from_the_meter(toy_tariff):
    """The anchor. cost(0) must equal the meter's own arithmetic, or every
    euro figure downstream is measuring the simulator rather than the house.
    """
    m = _toy_meter()
    priced = economics.price_capacities(m, _toy_nights(m), np.array([0.0]),
                                        toy_tariff)
    got = priced["cost_eur"].sum()

    b = toy_tariff.index_for(m["ts_utc"])
    lev = toy_tariff.bands["levering_eur_kwh"].to_numpy()[b]
    net = toy_tariff.net_export_eur_kwh[b]
    want = float((m["import_kwh"] * lev - m["export_kwh"] * net).sum())
    assert got == pytest.approx(want, rel=1e-9)


def test_a_battery_lowers_the_bill_and_the_saving_is_positive(toy_tariff):
    m = _toy_meter()
    priced = economics.price_capacities(m, _toy_nights(m), np.array([0.0, 5.0]),
                                        toy_tariff)
    s = economics.savings(priced)
    zero = s.loc[s.capacity_kwh == 0.0, "saving_eur"].sum()
    five = s.loc[s.capacity_kwh == 5.0, "saving_eur"].sum()
    assert zero == pytest.approx(0.0)
    assert five > 0.0


def test_savings_refuses_a_sweep_with_no_baseline(toy_tariff):
    m = _toy_meter()
    priced = economics.price_capacities(m, _toy_nights(m), np.array([5.0]),
                                        toy_tariff)
    with pytest.raises(ValueError, match="capacity 0"):
        economics.savings(priced)


def _spanning_meter(start, end, freq="6h"):
    """A record covering an explicit span, for the full-year rule."""
    ts = pd.date_range(start, end, freq=freq, tz="UTC")
    return pd.DataFrame({"ts_utc": ts, "import_kwh": 0.5, "export_kwh": 0.5})


def test_full_years_are_read_from_the_span_not_a_hardcoded_cutoff():
    """A year counts only if the record covers it end to end.

    The first draft pinned this to a literal date, which admitted 2019 — the
    real meter record opens at 2019-12-31 23:15 UTC, so that "year" holds
    three intervals and a EUR 0.03 bill, and averaging across it diluted
    every saving by a seventh.
    """
    m = _spanning_meter("2019-12-31 23:15", "2021-09-03 22:00")
    assert economics.full_years(m["ts_utc"]) == {2020}


def test_the_partial_year_is_flagged_rather_than_annualised(toy_tariff):
    """A part-year must never be averaged in as though it were whole.

    2024 is covered end to end here; 2025 stops in June. Treating 2025 as a
    full year would understate EUR/yr by half and nothing would look wrong.
    """
    m = _spanning_meter("2024-01-01", "2025-06-30")
    nights = pd.DataFrame([{
        "date": pd.Timestamp("2024-06-01"),
        "night_start_utc": pd.Timestamp("2024-06-01 00:00", tz="UTC"),
        "night_end_utc": pd.Timestamp("2024-06-01 06:00", tz="UTC"),
        "is_ev": False, "covered": True,
    }])
    priced = economics.price_capacities(m, nights, np.array([0.0]), toy_tariff)
    assert set(priced["year"]) == {2024, 2025}
    assert priced.loc[priced.year == 2024, "is_full_year"].all()
    assert not priced.loc[priced.year == 2025, "is_full_year"].any()


def test_arbitrage_ceiling_uses_the_widest_spread_available_in_a_day(toy_tariff):
    """Buy at the cheapest band, discharge at the dearest, once a day.

    Deliberately generous: it ignores that the battery is already busy doing
    self-consumption. A ceiling is allowed to be unreachable — that is what
    makes it a ceiling — but it must be labelled as one wherever it appears.
    """
    ts = pd.date_range("2024-06-01", periods=96 * 10, freq="15min", tz="UTC")
    m = pd.DataFrame({"ts_utc": ts, "import_kwh": 0.0, "export_kwh": 0.0})
    out = economics.arbitrage_ceiling_eur_yr(m, toy_tariff, capacity_kwh=10.0)

    # Toy bands: buy at 0.20, displace 0.30, round trip 0.90.
    assert out["best_spread_eur_kwh"] == pytest.approx(0.30 - 0.20 / 0.90)
    assert out["ceiling_eur_yr"] > 0.0


def test_arbitrage_ceiling_is_zero_when_every_band_costs_the_same():
    flat = Tariff(
        bands=pd.DataFrame({"band": ["A", "B"],
                            "levering_eur_kwh": [0.25, 0.25],
                            "terugleverkosten_eur_kwh": [0.05, 0.05],
                            "vergoeding_eur_kwh": [0.06, 0.06]}),
        hour_map=np.zeros((2, 24), dtype=int),
    )
    ts = pd.date_range("2024-06-01", periods=96 * 5, freq="15min", tz="UTC")
    m = pd.DataFrame({"ts_utc": ts, "import_kwh": 0.0, "export_kwh": 0.0})
    out = economics.arbitrage_ceiling_eur_yr(m, flat, capacity_kwh=10.0)
    assert out["ceiling_eur_yr"] == pytest.approx(0.0)


def test_arbitrage_ceiling_never_moves_more_than_the_inverter_can():
    """A huge battery is still limited by what the inverter can push in a day."""
    ts = pd.date_range("2024-06-01", periods=96 * 5, freq="15min", tz="UTC")
    m = pd.DataFrame({"ts_utc": ts, "import_kwh": 0.0, "export_kwh": 0.0})
    small = economics.arbitrage_ceiling_eur_yr(
        m, _two_band_tariff(), capacity_kwh=10.0)
    huge = economics.arbitrage_ceiling_eur_yr(
        m, _two_band_tariff(), capacity_kwh=1000.0)
    assert huge["ceiling_eur_yr"] < 30.0 * small["ceiling_eur_yr"]


def _two_band_tariff():
    bands = pd.DataFrame({
        "band": ["Peak", "Off"],
        "levering_eur_kwh": [0.30, 0.20],
        "terugleverkosten_eur_kwh": [0.07, 0.05],
        "vergoeding_eur_kwh": [0.08, 0.06],
    })
    hour_map = np.zeros((2, 24), dtype=int)
    hour_map[:, 0:12] = 1
    return Tariff(bands=bands, hour_map=hour_map)


def test_a_year_with_interior_holes_is_not_a_full_year():
    """Spanning a year is not covering it.

    2024 spans fine but is missing seven midwinter days from the January
    outage — the season that dominates a battery answer. Counted as whole it
    understates that year and moves the published optimum by half a step.
    """
    ts = pd.date_range("2024-01-01", "2024-12-31 23:45", freq="15min", tz="UTC")
    keep = ~((ts >= "2024-01-08") & (ts < "2024-01-15"))
    holed = pd.Series(ts[keep])
    whole = pd.Series(ts)

    assert economics.full_years(whole) == {2024}
    assert economics.full_years(holed) == set()


def test_vat_is_applied_where_a_household_actually_pays_it():
    assert economics.incl_vat(1000.0, 0.21) == pytest.approx(1210.0)
    assert economics.incl_vat(1000.0, 0.0) == pytest.approx(1000.0)


def test_the_quote_table_carries_both_vat_treatments():
    annual = pd.DataFrame({"capacity_kwh": [0.0, 5.0, 10.0, 15.0],
                           "saving_eur": [0.0, 285.0, 379.0, 407.0]})
    q = economics.quote_table(annual, fixed_cost_eur=941.82,
                              years=(10.0, 15.0))
    assert (q["payback_yr_incl_vat"] > q["payback_yr"]).all()
    assert q["total_incl_vat"].to_numpy() == pytest.approx(
        (q["total_ex_vat"] * (1 + economics.VAT_RATE)).to_numpy())
    # Both horizons are addressable; pinning one crashed the report build.
    assert {"optimum_kwh_10yr", "optimum_kwh_15yr"} <= set(q.columns)


def test_the_fixed_cost_terms_sum_to_the_published_total():
    """Every term is named, so a reader can see which are assumptions."""
    assert economics.FIXED_COST_TERMS["eur"].sum() == pytest.approx(
        economics.FIXED_COST_EUR)
    assumed = economics.FIXED_COST_TERMS["basis"].str.startswith("ASSUMED")
    assert assumed.sum() == 3, "the assumed terms must stay labelled"


def test_the_euro_optimum_does_not_move_with_the_sweeps_truncation():
    """The result phase 3 could not achieve, so it must not regress silently.

    Phase 3's geometric elbow drifted with where the capacity sweep stopped —
    8.5 kWh at a 25 kWh sweep, 9.0 at 30, 11.0 at 60, never settling — and
    that hidden dependence on the axis extent is why its headline had to be
    published as a lower bound. A euro curve against a euro price has a real
    exchange rate, so the argmax is a property of the data rather than of the
    plotting range.

    Built on a concave saving curve, which is the shape the real data has.
    """
    caps = np.arange(0.0, 30.01, 0.5)
    annual = pd.DataFrame({
        "capacity_kwh": caps,
        "saving_eur": 430.0 * (1.0 - np.exp(-caps / 6.0)),
    })
    at = [economics.recommend_capacity_eur(
              annual[annual.capacity_kwh <= top], 122.07, 941.82, 10.0)
          for top in (12.0, 15.0, 20.0, 30.0)]
    assert len(set(at)) == 1, f"optimum drifted with the sweep top: {at}"
