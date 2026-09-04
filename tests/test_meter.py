import pandas as pd
import pytest

from pvnight.meter import METER_SUBDIR, find_gaps, load_meter


@pytest.fixture(scope="session")
def meter(repo_root):
    return load_meter(repo_root / METER_SUBDIR)


def test_every_column_is_parsed_from_strings(meter):
    """The workbook stores the timestamp AND every energy value as text, with
    a decimal comma. Nothing may survive as an object dtype."""
    assert isinstance(meter["ts_utc"].dtype, pd.DatetimeTZDtype)
    assert str(meter["ts_utc"].dtype.tz) == "UTC"
    assert meter["import_kwh"].dtype == "float64"
    assert meter["export_kwh"].dtype == "float64"


def test_decimal_comma_becomes_a_real_number(meter):
    """'0,04' must be 0.04, not 4.0 and not a string."""
    assert meter["import_kwh"].max() < 20.0     # a 15-min interval, in kWh
    assert meter["export_kwh"].max() < 20.0
    assert (meter["import_kwh"] >= 0).all()
    assert (meter["export_kwh"] >= 0).all()


def test_span_and_row_count_match_the_measured_file_set(meter):
    """Spec §2.3: 233,358 rows, 2019-12-31 23:15 UTC to 2026-09-03 22:00."""
    assert len(meter) == 233358
    assert meter["ts_utc"].iloc[0] == pd.Timestamp("2019-12-31T23:15:00Z")
    assert meter["ts_utc"].iloc[-1] == pd.Timestamp("2026-09-03T22:00:00Z")


def test_timestamps_are_unique_and_ordered(meter):
    assert meter["ts_utc"].is_monotonic_increasing
    assert not meter["ts_utc"].duplicated().any()


def test_both_dst_offsets_land_at_the_right_utc_instant(meter):
    """The workbook writes +0100 in winter and +0200 in summer. A naive parse
    would put the summer rows an hour late."""
    winter = meter[meter["ts_utc"] == pd.Timestamp("2023-01-15T12:00:00Z")]
    summer = meter[meter["ts_utc"] == pd.Timestamp("2023-07-15T12:00:00Z")]
    assert len(winter) == 1
    assert len(summer) == 1


def test_tariff_pairs_are_summed_not_dropped(meter):
    """Each pair is null when the other tariff is active. Summing after a
    zero-fill is correct; dropping nulls would halve the totals."""
    assert meter["import_kwh"].sum() > 25_000     # ~4.5 MWh/yr over 6.7 years
    assert meter["export_kwh"].sum() > 20_000
    assert meter["import_kwh"].isna().sum() == 0
    assert meter["export_kwh"].isna().sum() == 0


def test_find_gaps_reports_all_seven(meter):
    """Spec §2.3 lists seven, four of them in January 2024. They must be
    reported, never silently interpolated."""
    g = find_gaps(meter)
    assert len(g) == 7
    assert int(g["missing_intervals"].sum()) == 686
    jan24 = g[g["gap_start_utc"].dt.strftime("%Y-%m") == "2024-01"]
    assert len(jan24) == 4     # the fifth 2024 gap is in July
    biggest = g.loc[g["missing_intervals"].idxmax()]
    assert biggest["gap_start_utc"] == pd.Timestamp("2024-01-08T23:00:00Z")


def test_find_gaps_on_a_clean_frame_returns_nothing():
    clean = pd.DataFrame({
        "ts_utc": pd.date_range("2023-01-01", periods=10, freq="15min", tz="UTC"),
        "import_kwh": 0.1, "export_kwh": 0.0,
    })
    assert len(find_gaps(clean)) == 0


def test_load_meter_rejects_a_directory_with_no_workbooks(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_meter(tmp_path)
