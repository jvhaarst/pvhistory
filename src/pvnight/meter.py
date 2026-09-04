"""Read the smart-meter exports.

The meter measures what crosses the grid connection, which is exactly what a
battery interacts with. Unlike PVOutput's consumption channel, it sees the
whole house — see the spec's §2 for the December 2022 fault that makes this
module necessary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METER_SUBDIR = "data/meterdata"
METER_GLOB = "data_*.xlsx"
DT_HOURS = 0.25

_TS_FORMAT = "%d-%m-%Y %H:%M:%S %z"
_IMPORT_COLS = ("levering_normaal", "levering_laag")
_EXPORT_COLS = ("teruglevering_normaal", "teruglevering_laag")


def _decimal_comma(series: pd.Series) -> pd.Series:
    """'0,04' -> 0.04. Everything arrives as text, including the numbers."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )


def load_meter(meter_dir: Path) -> pd.DataFrame:
    """Every yearly workbook, as one UTC-indexed frame.

    Each tariff column is null when the *other* tariff is active, so the pair
    is zero-filled and summed. Dropping nulls instead would halve the totals.
    Timestamps label the interval END, and carry +0100 or +0200 explicitly,
    so they localise without any DST guesswork.
    """
    paths = sorted(Path(meter_dir).glob(METER_GLOB))
    if not paths:
        raise FileNotFoundError(f"no files matching {METER_GLOB!r} in {meter_dir}")

    frames = []
    for p in paths:
        raw = pd.read_excel(p)
        imp = sum(_decimal_comma(raw[c]).fillna(0.0) for c in _IMPORT_COLS)
        exp = sum(_decimal_comma(raw[c]).fillna(0.0) for c in _EXPORT_COLS)
        frames.append(pd.DataFrame({
            "ts_utc": pd.to_datetime(raw["datum_tijd"], format=_TS_FORMAT, utc=True),
            "import_kwh": imp.astype("float64"),
            "export_kwh": exp.astype("float64"),
        }))

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("ts_utc", kind="stable")
        .reset_index(drop=True)
    )


def find_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Every run of missing intervals, reported rather than interpolated.

    Four of the seven gaps fall in January 2024 and together remove most of
    8-19 January — midwinter, when night consumption peaks. A night touching
    one of these must be excluded, not counted as a quiet night.
    """
    ts = frame["ts_utc"]
    step = pd.Timedelta(hours=DT_HOURS)
    delta = ts.diff()
    idx = np.where(delta > step)[0]
    return pd.DataFrame({
        "gap_start_utc": ts.iloc[idx - 1].to_numpy(),
        "gap_end_utc": ts.iloc[idx].to_numpy(),
        "missing_intervals": (delta.iloc[idx] / step - 1).astype(int).to_numpy(),
    })
