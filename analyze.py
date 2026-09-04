"""Run the full pipeline: load history, fit the envelope, write the outputs."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from pvnight import envelope as env
from pvnight import report
from pvnight.events import first_last_light
from pvnight.loader import load

START_DATE = dt.date(2020, 1, 1)
END_DATE = dt.date(2026, 12, 31)


def run(data_dir: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load(Path(data_dir))
    events = env.attach_solar(first_last_light(samples))
    model = env.fit(events)
    windows = env.build_windows(
        model, START_DATE, END_DATE, set(events["solar_date"])
    )

    env.thresholds_table(model).to_csv(out_dir / "solar_thresholds.csv", index=False)
    windows.to_csv(out_dir / "solar_windows.csv", index=False)
    (out_dir / "report.html").write_text(
        report.build_html(model, events, windows, samples)
    )

    nights = windows["night_duration_h"].dropna()
    return {
        "n_days_observed": len(events),
        "n_windows": len(windows),
        "median_start_offset_min": float(windows["start_offset_min"].median()),
        "shortest_night_h": float(nights.min()),
        "longest_night_h": float(nights.max()),
    }


if __name__ == "__main__":
    summary = run(Path(__file__).parent, Path(__file__).parent / "out")
    for key, value in summary.items():
        print(f"{key}: {value}")
