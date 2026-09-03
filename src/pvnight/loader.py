"""Load PVOutput history exports into a tidy, UTC-indexed frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_GLOB, SITE_TZ


def generating_flag(df: pd.DataFrame) -> pd.Series:
    """True where the installation was producing.

    Three independent signals are OR-ed because the null-versus-zero
    convention changes across years (spec fact 2): explicit power in either
    power column, or an increase in the within-day cumulative energy counter.
    Grouping the diff by ``solar_date`` matters — the counter resets at
    midnight, so a global diff would compare across the day boundary.
    """
    power_on = (df["power_gen_w"] > 0) | (df["power_avg_w"] > 0)
    energy_rose = df.groupby("solar_date")["energy_gen_wh"].diff().fillna(0.0) > 0
    return (power_on | energy_rose).rename("generating")


def load(data_dir: Path) -> pd.DataFrame:
    """Read every yearly export in ``data_dir`` into one frame.

    Timestamps are localized once and immediately converted to UTC; every
    downstream computation works in UTC. ``solar_date`` keeps the local
    calendar date, which is the correct grouping key for a solar day.
    """
    paths = sorted(Path(data_dir).glob(DATA_GLOB))
    if not paths:
        raise FileNotFoundError(f"no files matching {DATA_GLOB!r} in {data_dir}")

    # These are plain Parquet despite the .xz suffix — do not use lzma.
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    naive = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"], format="%Y%m%d %H:%M"
    )
    local = naive.dt.tz_localize(
        SITE_TZ,
        nonexistent="shift_forward",  # spring-forward gap
        ambiguous=False,              # arbitrary; made inconsequential, spec 5.1
    )

    df = pd.DataFrame(
        {
            "ts_utc": local.dt.tz_convert("UTC"),
            "solar_date": pd.to_datetime(raw["Date"], format="%Y%m%d").dt.date,
            "power_gen_w": raw["Instantaneous Power"].astype("float64").fillna(0.0),
            "power_avg_w": raw["Average Power"].astype("float64").fillna(0.0),
            "energy_gen_wh": raw["Energy Generation"].astype("float64"),
            "power_cons_w": raw["Power Consumption"].astype("float64"),
            "energy_cons_wh": raw["Energy Consumption"].astype("float64"),
        }
    ).sort_values("ts_utc", kind="stable").reset_index(drop=True)

    df["generating"] = generating_flag(df)
    return df
