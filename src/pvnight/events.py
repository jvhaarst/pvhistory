"""Reduce the sample-level frame to one first/last-light pair per solar day.

Timestamps only. Attaching solar geometry is the envelope module's job.
"""

from __future__ import annotations

import pandas as pd


def first_last_light(df: pd.DataFrame) -> pd.DataFrame:
    """One row per solar day on which the installation produced anything."""
    gen = df[df["generating"]]
    out = gen.groupby("solar_date").agg(
        first_light_utc=("ts_utc", "min"),
        last_light_utc=("ts_utc", "max"),
        n_generating=("ts_utc", "size"),
        daily_yield_wh=("energy_gen_wh", "max"),
    )
    return out.reset_index()
