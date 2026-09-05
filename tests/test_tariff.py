from pathlib import Path

import pandas as pd
import pytest

from pvnight.tariff import TARIFF_FILE, load_tariff

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tariff():
    return load_tariff(REPO / TARIFF_FILE)


def test_the_three_bands_carry_the_prices_from_the_file(tariff):
    b = tariff.bands.set_index("band")
    assert b.loc["Normaal", "levering_eur_kwh"] == pytest.approx(0.30566)
    assert b.loc["Normaal", "terugleverkosten_eur_kwh"] == pytest.approx(0.07049)
    assert b.loc["Normaal", "vergoeding_eur_kwh"] == pytest.approx(0.08050)
    assert b.loc["Dal", "levering_eur_kwh"] == pytest.approx(0.27939)
    assert b.loc["SuperDal", "levering_eur_kwh"] == pytest.approx(0.18225)


def test_net_export_is_one_cent_in_every_band(tariff):
    """The finding the whole phase rests on: terugleverkosten claw back all
    but a cent of the vergoeding, so there is no net metering left."""
    assert tariff.net_export_eur_kwh == pytest.approx(0.01, abs=2e-5)


def test_the_bands_tile_every_hour_of_both_seasons(tariff):
    assert tariff.hour_map.shape == (2, 24)
    assert (tariff.hour_map >= 0).all(), "an hour was left unassigned"


def test_the_same_clock_hour_changes_band_across_the_season_boundary(tariff):
    """31 March 14:00 local is Dal; 1 April 14:00 local is SuperDal.

    Both are 12:00 UTC in 2026 — the season, not the clock, decides.
    """
    ts = pd.DatetimeIndex(["2026-03-31 12:00", "2026-04-01 12:00"], tz="UTC")
    names = tariff.bands["band"].to_numpy()[tariff.index_for(ts)]
    assert list(names) == ["Dal", "SuperDal"]


def test_a_winter_afternoon_is_not_superdal(tariff):
    ts = pd.DatetimeIndex(["2026-01-15 11:00"], tz="UTC")   # 12:00 local
    assert tariff.bands["band"].to_numpy()[tariff.index_for(ts)][0] == "Dal"


def test_evening_peak_is_normaal_in_both_seasons(tariff):
    ts = pd.DatetimeIndex(["2026-01-17 17:30", "2026-07-17 16:30"], tz="UTC")
    names = tariff.bands["band"].to_numpy()[tariff.index_for(ts)]
    assert list(names) == ["Normaal", "Normaal"]


def test_every_interval_of_a_dst_spring_forward_day_gets_a_band(tariff):
    """2026-03-29: local jumps 02:00 -> 03:00, so local hour 2 never occurs.

    Assigning bands by naive local arithmetic rather than tz_convert is the
    kind of thing that produces a silently wrong hour here.
    """
    ts = pd.date_range("2026-03-29", periods=96, freq="15min", tz="UTC")
    idx = tariff.index_for(ts)
    assert len(idx) == 96
    assert (idx >= 0).all()
    local_hours = set(pd.DatetimeIndex(ts).tz_convert("Europe/Amsterdam").hour)
    assert 2 not in local_hours, "fixture assumption: 02:00 local is skipped"


def test_every_interval_of_a_dst_fall_back_day_gets_a_band(tariff):
    """2026-10-25: local hour 2 occurs twice. Both are Dal (00:00-07:00)."""
    ts = pd.date_range("2026-10-25", periods=96, freq="15min", tz="UTC")
    names = tariff.bands["band"].to_numpy()[tariff.index_for(ts)]
    local = pd.DatetimeIndex(ts).tz_convert("Europe/Amsterdam")
    twos = [n for n, h in zip(names, local.hour) if h == 2]
    assert len(twos) == 8, "fixture assumption: 02:00 local repeats"
    assert set(twos) == {"Dal"}


def test_index_for_accepts_a_series_as_well_as_an_index(tariff):
    """Callers hold `meter_df["ts_utc"]`, which is a Series, not an Index."""
    ts = pd.date_range("2026-07-01", periods=8, freq="15min", tz="UTC")
    assert list(tariff.index_for(pd.Series(ts))) == list(tariff.index_for(ts))


def test_a_malformed_file_raises_rather_than_returning_half_a_tariff(tmp_path):
    bad = tmp_path / "tariff.txt"
    bad.write_text("Zomer (1 april t/m 30 september)\n"
                   "    Normaal zomer (07:00 - 10:00)\n"
                   "        Leveringskosten         € 0,30566 per kWh\n"
                   "        Terugleverkosten        € 0,07049 per kWh\n"
                   "        Terugleververgoeding    € 0,08050 per kWh\n")
    with pytest.raises(ValueError, match="tile"):
        load_tariff(bad)
