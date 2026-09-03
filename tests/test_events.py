import datetime as dt

import pandas as pd

from pvnight.events import first_last_light


def _frame(rows):
    """rows: (ts_utc iso, solar_date, generating, energy_gen_wh)."""
    return pd.DataFrame(
        [
            {
                "ts_utc": pd.Timestamp(ts, tz="UTC"),
                "solar_date": d,
                "generating": g,
                "energy_gen_wh": e,
            }
            for ts, d, g, e in rows
        ]
    )


def test_picks_first_and_last_generating_sample():
    df = _frame([
        ("2023-06-01T03:00", dt.date(2023, 6, 1), False, 0.0),
        ("2023-06-01T04:00", dt.date(2023, 6, 1), True, 10.0),
        ("2023-06-01T12:00", dt.date(2023, 6, 1), True, 900.0),
        ("2023-06-01T20:00", dt.date(2023, 6, 1), False, 900.0),
    ])
    out = first_last_light(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["first_light_utc"] == pd.Timestamp("2023-06-01T04:00", tz="UTC")
    assert row["last_light_utc"] == pd.Timestamp("2023-06-01T12:00", tz="UTC")
    assert row["n_generating"] == 2
    assert row["daily_yield_wh"] == 900.0


def test_days_with_no_generation_are_absent():
    df = _frame([
        ("2023-01-01T10:00", dt.date(2023, 1, 1), False, 0.0),
        ("2023-01-02T10:00", dt.date(2023, 1, 2), True, 50.0),
    ])
    out = first_last_light(df)
    assert out["solar_date"].tolist() == [dt.date(2023, 1, 2)]


def test_real_data_matches_the_established_count(loaded):
    """Spec fact 4: 2,047 of 2,052 days recorded generation."""
    out = first_last_light(loaded)
    assert len(out) == 2047


def test_first_light_never_after_last_light(loaded):
    out = first_last_light(loaded)
    assert (out["first_light_utc"] <= out["last_light_utc"]).all()
