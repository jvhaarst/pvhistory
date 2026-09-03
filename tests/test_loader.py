import datetime as dt

import pandas as pd
import pytest

from pvnight.loader import generating_flag, load


def test_utc_index_is_monotonic_and_unique(loaded):
    """The single assertion that catches the entire DST bug class."""
    ts = loaded["ts_utc"]
    assert ts.is_monotonic_increasing
    assert not ts.duplicated().any()


def test_spring_forward_day_has_23_hours_of_samples(loaded):
    """2023-03-26 is a genuine 23-hour local day: 276 five-minute slots."""
    day = loaded[loaded["solar_date"] == dt.date(2023, 3, 26)]
    assert len(day) == 276


def test_fall_back_day_has_288_slots_and_no_duplicate_instants(loaded):
    """Fall-back days hold 288 rows, not the 300 a true 25-hour day would
    need. One hour is unrecoverable (spec fact 5); what matters is that the
    rows we do have map to distinct UTC instants."""
    day = loaded[loaded["solar_date"] == dt.date(2023, 10, 29)]
    assert len(day) == 288
    assert not day["ts_utc"].duplicated().any()


def test_expected_columns_and_dtypes(loaded):
    """Tz-aware and UTC is the contract. The datetime resolution is
    pandas-version-dependent (2.x gives ns, 3.x gives us) and deliberately
    not asserted."""
    dtype = loaded["ts_utc"].dtype
    assert isinstance(dtype, pd.DatetimeTZDtype)
    assert str(dtype.tz) == "UTC"
    assert loaded["generating"].dtype == bool
    for col in ["power_gen_w", "power_avg_w", "energy_gen_wh",
                "power_cons_w", "energy_cons_wh"]:
        assert loaded[col].dtype == "float64"


def test_generation_columns_have_no_nulls(loaded):
    """Generation nulls are zero: the cumulative counter resets daily, so 0
    is the true value for a pre-dawn null (spec fact 2)."""
    for col in ["power_gen_w", "power_avg_w", "energy_gen_wh"]:
        assert not loaded[col].isna().any()


def test_consumption_nulls_are_preserved(loaded):
    """Deliberately NOT filled. A null here means no measurement was
    recorded, not that no power was drawn; zero-filling would understate
    night-time consumption, which is the whole point of the dataset."""
    assert loaded["power_cons_w"].isna().any()
    assert loaded["energy_cons_wh"].isna().any()


def test_energy_rise_detects_first_light_when_power_still_reads_zero(loaded):
    """Regression: on 2025-05-13 the counter reaches 1 Wh at 03:50 UTC while
    both power columns still read 0.0. The energy limb must catch it."""
    import datetime as dt

    day = loaded[loaded["solar_date"] == dt.date(2025, 5, 13)]
    first = day.loc[day["generating"], "ts_utc"].min()
    assert first == pd.Timestamp("2025-05-13T03:50:00", tz="UTC")


def _frame(rows):
    """rows: list of (solar_date, power_gen_w, power_avg_w, energy_gen_wh)."""
    return pd.DataFrame(
        rows, columns=["solar_date", "power_gen_w", "power_avg_w", "energy_gen_wh"]
    )


def test_generating_flag_is_true_on_positive_power():
    df = _frame([(dt.date(2023, 6, 1), 0.0, 0.0, 0.0),
                 (dt.date(2023, 6, 1), 12.0, 0.0, 0.0)])
    assert generating_flag(df).tolist() == [False, True]


def test_generating_flag_is_true_on_positive_average_power():
    """Some years report Average Power when Instantaneous Power is absent."""
    df = _frame([(dt.date(2023, 6, 1), 0.0, 0.0, 0.0),
                 (dt.date(2023, 6, 1), 0.0, 9.0, 0.0)])
    assert generating_flag(df).tolist() == [False, True]


def test_generating_flag_detects_rising_cumulative_energy():
    """2020-2021 write nulls for power at night but still accumulate energy."""
    df = _frame([(dt.date(2023, 6, 1), 0.0, 0.0, 100.0),
                 (dt.date(2023, 6, 1), 0.0, 0.0, 140.0),
                 (dt.date(2023, 6, 1), 0.0, 0.0, 140.0)])
    assert generating_flag(df).tolist() == [False, True, False]


def test_generating_flag_does_not_leak_across_the_day_boundary():
    """Cumulative energy resets each day. Grouping by solar_date means the
    first sample of a new day is never compared against the previous day."""
    df = _frame([(dt.date(2023, 6, 1), 0.0, 0.0, 5000.0),
                 (dt.date(2023, 6, 2), 0.0, 0.0, 0.0),
                 (dt.date(2023, 6, 2), 0.0, 0.0, 30.0)])
    assert generating_flag(df).tolist() == [False, False, True]


def test_load_rejects_a_directory_with_no_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)
