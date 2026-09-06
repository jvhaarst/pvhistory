"""Discounted cash flow for a battery purchase that has already been chosen.

Phase 4 answered "what size". This answers "is buying this thing better than
leaving the money invested elsewhere", which is a different question and has
a different failure mode: the arithmetic is easy and the assumptions decide
everything, so the tests pin the arithmetic and the report publishes the
sensitivity.
"""

import re

import numpy as np
import pandas as pd
import pytest

from pvnight import finance


@pytest.fixture
def flat_curve():
    """Saving independent of capacity, so degradation cannot bite.

    Isolates the discounting from the capacity lookup: any test using this
    curve is testing money, not the battery.
    """
    return pd.DataFrame({"capacity_kwh": np.arange(0.0, 30.01, 0.5),
                         "saving_eur": 100.0})


def test_npv_of_a_flat_stream_matches_the_annuity_formula(flat_curve):
    """The closed form is the check: sum of 100/(1.1^n) for n=1..10."""
    cf = finance.cashflows(flat_curve, capacity_kwh=10.0, cost_eur=0.0,
                           years=10, degradation=0.0, inflation=0.0,
                           discount=0.10)
    want = sum(100.0 / 1.10 ** n for n in range(1, 11))
    assert finance.npv(cf) == pytest.approx(want)


def test_zero_discount_and_zero_inflation_is_a_plain_sum(flat_curve):
    cf = finance.cashflows(flat_curve, capacity_kwh=10.0, cost_eur=250.0,
                           years=5, degradation=0.0, inflation=0.0,
                           discount=0.0)
    assert finance.npv(cf) == pytest.approx(5 * 100.0 - 250.0)


def test_inflation_and_discount_compose_as_a_real_rate(flat_curve):
    """Nominal cash flows inflating at 3% discounted at 10% behave exactly
    like flat cash flows discounted at 1.10/1.03 - 1.

    Deflating the savings AND discounting at 10% would double-count
    inflation; this pins that they compose rather than stack.
    """
    a = finance.npv(finance.cashflows(
        flat_curve, 10.0, 0.0, 12, degradation=0.0, inflation=0.03,
        discount=0.10))
    b = finance.npv(finance.cashflows(
        flat_curve, 10.0, 0.0, 12, degradation=0.0, inflation=0.0,
        discount=1.10 / 1.03 - 1.0))
    assert a == pytest.approx(b)


def test_degradation_shrinks_the_capacity_and_is_looked_up_not_scaled():
    """The rule that is easiest to get wrong.

    The saving curve is concave, so a capacity past its elbow loses very
    little to early degradation. Scaling the euro saving by 0.985**n instead
    of looking up the degraded capacity would understate the early years —
    here by more than a third over fifteen years.
    """
    curve = pd.DataFrame({"capacity_kwh": [0.0, 5.0, 10.0, 20.0],
                          "saving_eur": [0.0, 300.0, 340.0, 350.0]})
    cf = finance.cashflows(curve, capacity_kwh=10.0, cost_eur=0.0, years=15,
                           degradation=0.015, inflation=0.0, discount=0.0)

    assert cf["capacity_kwh"].iloc[0] == pytest.approx(10.0 * 0.985)
    assert cf["capacity_kwh"].iloc[-1] == pytest.approx(10.0 * 0.985 ** 15)

    looked_up = float(cf["saving_eur"].sum())
    scaled = sum(340.0 * 0.985 ** n for n in range(1, 16))
    assert looked_up > scaled * 1.05, (
        "scaling the saving instead of looking up the degraded capacity "
        "should understate a concave curve past its elbow")


def test_implied_return_is_the_rate_that_zeroes_the_npv(flat_curve):
    r = finance.implied_return(flat_curve, capacity_kwh=10.0,
                               cost_eur=600.0, years=10, degradation=0.0,
                               inflation=0.0)
    cf = finance.cashflows(flat_curve, 10.0, 600.0, 10, degradation=0.0,
                           inflation=0.0, discount=r)
    assert finance.npv(cf) == pytest.approx(0.0, abs=1e-6)


def test_implied_return_is_nan_when_the_stream_never_repays(flat_curve):
    """A cost no stream of savings can recover has no internal rate."""
    r = finance.implied_return(flat_curve, capacity_kwh=10.0,
                               cost_eur=1e9, years=10, degradation=0.0,
                               inflation=0.0)
    assert np.isnan(r)


def test_horizon_sweep_brackets_the_break_even_year(flat_curve):
    """The deliverable: where the investment crosses the alternative."""
    sweep = finance.horizon_sweep(flat_curve, capacity_kwh=10.0,
                                  cost_eur=600.0, years=range(2, 16),
                                  degradation=0.0, inflation=0.0,
                                  discount=0.10)
    assert list(sweep["years"]) == list(range(2, 16))
    assert sweep["npv_eur"].is_monotonic_increasing
    crossings = sweep[sweep["npv_eur"] >= 0]
    assert not crossings.empty
    assert finance.break_even_year(sweep) == int(crossings["years"].iloc[0])


def test_break_even_year_is_none_when_it_never_crosses(flat_curve):
    sweep = finance.horizon_sweep(flat_curve, 10.0, 1e9, range(2, 8),
                                  degradation=0.0, inflation=0.0,
                                  discount=0.10)
    assert finance.break_even_year(sweep) is None


def test_the_assumptions_table_carries_every_rate_the_model_uses():
    """No rate may reach the report except through this table.

    Three assumptions decide this answer entirely, so they are declared in
    one place with their provenance rather than scattered as defaults.
    """
    a = finance.ASSUMPTIONS.set_index("name")
    for k in ("degradation", "inflation", "discount"):
        assert k in a.index
        assert a.loc[k, "source"], f"{k} has no stated source"
    assert a.loc["degradation", "value"] == pytest.approx(0.015)
    assert a.loc["inflation", "value"] == pytest.approx(0.03)
    assert a.loc["discount", "value"] == pytest.approx(0.10)


def test_sensitivity_varies_one_rate_at_a_time(flat_curve):
    s = finance.sensitivity(flat_curve, capacity_kwh=10.0, cost_eur=600.0,
                            years=15)
    assert set(s["rate"]) == {"degradation", "inflation", "discount"}
    base = s[s["is_base"]]
    assert len(base) == 3, "each rate must show its own baseline row"
    # The three baselines describe the same scenario, so they must agree.
    assert base["npv_eur"].nunique() == 1


def test_the_page_states_the_verdict_and_its_own_fragility(flat_curve):
    """The verdict and the break-even year must move with the data.

    This page's whole risk is a confident sentence over assumptions that
    could go either way, so the prose is derived and the assumptions are
    published beside it.
    """
    from pvnight import investment_report

    sweep = finance.horizon_sweep(flat_curve, 10.0, 600.0, range(5, 16),
                                  degradation=0.0, inflation=0.0,
                                  discount=0.10)
    cf = finance.cashflows(flat_curve, 10.0, 600.0, 15, degradation=0.0,
                           inflation=0.0, discount=0.10)
    sens = finance.sensitivity(flat_curve, 10.0, 600.0, 15)
    html = investment_report.build_html(
        finance.ASSUMPTIONS, cf, sweep, sens, 15, 0.10, "Test cell", 10.0,
        600.0, 100.0, purchase=finance.purchase_table())

    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    be = finance.break_even_year(sweep)
    assert f"first pulls ahead at {be} years" in text
    assert "Test cell" in text
    # Every assumption is on the page with its provenance, and the inputs
    # appear before the verdict that rests on them.
    for label in finance.ASSUMPTIONS["label"]:
        assert label in text
    assert "owner's assumption" in text
    assert text.index("What goes in") < text.index("The verdict")


def test_the_page_cannot_be_built_without_its_inputs():
    """A verdict with no visible assumptions is the failure this page risks."""
    from pvnight import investment_report

    with pytest.raises(TypeError):
        investment_report.build_html(
            finance.ASSUMPTIONS, pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), 15, 0.10, "Test cell", 10.0, 600.0, 100.0)


def test_the_verdict_flips_when_the_investment_loses(flat_curve):
    """A page that can only say yes is not reporting anything."""
    from pvnight import investment_report

    sweep = finance.horizon_sweep(flat_curve, 10.0, 5000.0, range(5, 16),
                                  degradation=0.0, inflation=0.0,
                                  discount=0.10)
    cf = finance.cashflows(flat_curve, 10.0, 5000.0, 15, degradation=0.0,
                           inflation=0.0, discount=0.10)
    sens = finance.sensitivity(flat_curve, 10.0, 5000.0, 15)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", investment_report.build_html(
        finance.ASSUMPTIONS, cf, sweep, sens, 15, 0.10, "Test cell", 10.0,
        5000.0, 100.0, purchase=finance.purchase_table())))
    assert "loses to" in text
    assert "never pulls ahead" in text
