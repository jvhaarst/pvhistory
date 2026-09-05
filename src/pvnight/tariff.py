"""The electricity tariff, parsed from the supplier's own file.

Every euro figure in this project traces back to this module. The file gives
three prices per band, not one: what a kWh costs to import (`levering`), what
it costs to export (`terugleverkosten`), and what exporting pays
(`vergoeding`). The last two nearly cancel — net feed-in is EUR 0.01/kWh in
every band — which is why storing a kWh is worth 18-30x exporting it, and the
whole reason a battery can pay for itself here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SITE_TZ

TARIFF_FILE = "data/Greenchoice - Variabele kosten - 2027"

_SEASON_RE = re.compile(r"^(Zomer|Winter)\s*\(")
_BAND_RE = re.compile(r"^\s+(Normaal|Dal|SuperDal)\s+(zomer|winter)\s*\((.+)\)\s*$")
_PRICE_RE = re.compile(
    r"^\s+(Leveringskosten|Terugleverkosten|Terugleververgoeding)"
    r"\s+€\s*([0-9]+,[0-9]+)\s+per kWh\s*$")
_RANGE_RE = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")

_PRICE_COL = {
    "Leveringskosten": "levering_eur_kwh",
    "Terugleverkosten": "terugleverkosten_eur_kwh",
    "Terugleververgoeding": "vergoeding_eur_kwh",
}


@dataclass(frozen=True)
class Tariff:
    """Band prices, and the map from (season, local hour) to a band."""

    bands: pd.DataFrame
    hour_map: np.ndarray

    @property
    def net_export_eur_kwh(self) -> np.ndarray:
        """What a kWh exported actually earns, per band.

        Derived here rather than left to callers, because the two components
        nearly cancelling is the finding the whole phase rests on.
        """
        return (self.bands["vergoeding_eur_kwh"].to_numpy()
                - self.bands["terugleverkosten_eur_kwh"].to_numpy())

    def index_for(self, ts_utc) -> np.ndarray:
        """Band index for each UTC timestamp.

        The one place this project converts to local time. Band boundaries are
        wall-clock Amsterdam, so 07:00-10:00 moves against UTC twice a year;
        `tz_convert` handles both transition days, and no arithmetic on naive
        local timestamps happens anywhere. An hour misassigned between Normaal
        and SuperDal is a 12.3 cent/kWh error landing systematically in summer
        afternoons, the largest block of export in the record.
        """
        local = pd.DatetimeIndex(ts_utc).tz_convert(SITE_TZ)
        summer = ((local.month >= 4) & (local.month <= 9)).astype(int)
        return self.hour_map[summer, local.hour]


def _parse_ranges(text: str) -> list[tuple[int, int]]:
    """`"07:00 - 10:00 en 17:00 - 22:00"` -> `[(7, 10), (17, 22)]`.

    An end of `00:00` means midnight at the end of the day, so it becomes 24.
    """
    out = []
    for h1, m1, h2, m2 in _RANGE_RE.findall(text):
        if m1 != "00" or m2 != "00":
            raise ValueError(f"tariff: non-hourly band boundary in {text!r}")
        start, end = int(h1), int(h2)
        out.append((start, 24 if end == 0 else end))
    if not out:
        raise ValueError(f"tariff: no time ranges in {text!r}")
    return out


def load_tariff(path: Path) -> Tariff:
    """Parse the supplier's file into a `Tariff`.

    Strict by choice: a file that does not tile both seasons raises rather
    than yielding a tariff with silent holes. A hole would price some energy
    at zero, and no test of the euro figures downstream would look wrong.
    """
    rows: list[dict] = []
    season = None
    cur: dict | None = None

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _SEASON_RE.match(line)
        if m:
            season = "summer" if m.group(1) == "Zomer" else "winter"
            cur = None
            continue
        m = _BAND_RE.match(line)
        if m:
            cur = {"season": season, "band": m.group(1),
                   "hours": _parse_ranges(m.group(3))}
            rows.append(cur)
            continue
        m = _PRICE_RE.match(line)
        if m:
            if cur is None:
                raise ValueError("tariff: a price line preceded any band")
            cur[_PRICE_COL[m.group(1)]] = float(m.group(2).replace(",", "."))

    if not rows:
        raise ValueError(f"tariff: no bands parsed from {path}")

    parsed = pd.DataFrame(rows)
    missing = [c for c in _PRICE_COL.values() if c not in parsed.columns]
    if missing:
        raise ValueError(f"tariff: bands are missing prices {missing}")
    if parsed[list(_PRICE_COL.values())].isna().any().any():
        raise ValueError("tariff: at least one band is missing a price")

    bands = (parsed.groupby("band", as_index=False)[list(_PRICE_COL.values())]
             .first()
             .sort_values("levering_eur_kwh", ascending=False)
             .reset_index(drop=True))
    band_row = {b: i for i, b in enumerate(bands["band"])}

    hour_map = np.full((2, 24), -1, dtype=int)
    for r in rows:
        s = 1 if r["season"] == "summer" else 0
        for start, end in r["hours"]:
            for h in range(start, end):
                if hour_map[s, h] != -1:
                    raise ValueError(
                        f"tariff: hour {h} of {r['season']} is in two bands")
                hour_map[s, h] = band_row[r["band"]]

    if (hour_map < 0).any():
        holes = [(s, h) for s in (0, 1) for h in range(24) if hour_map[s, h] < 0]
        raise ValueError(f"tariff: bands do not tile the day, holes at {holes}")

    return Tariff(bands=bands, hour_map=hour_map)
