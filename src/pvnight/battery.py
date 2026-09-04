"""Chronological battery simulation, vectorised across candidate capacities.

The simulation is inherently sequential in time — each step's state depends on
the last — but every candidate capacity can advance together as a numpy vector,
so the whole sweep is one pass over the data rather than one pass per capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT_HOURS = 5.0 / 60.0


@dataclass(frozen=True)
class BatterySpec:
    """Candidate capacities and the inverter they sit behind.

    Capacities are nameplate — the number a product is sold under. Usable
    energy at the terminal is `capacity * usable_fraction`, and the round trip
    is split symmetrically so charging and discharging each cost sqrt(eta).
    """

    capacities_kwh: np.ndarray
    power_kw: float
    round_trip: float = 0.90
    usable_fraction: float = 0.90

    @property
    def usable_wh(self) -> np.ndarray:
        return np.asarray(self.capacities_kwh, dtype=float) * 1000.0 * self.usable_fraction

    @property
    def eta(self) -> float:
        """One-way efficiency: sqrt of the round trip, applied on each side."""
        return float(np.sqrt(self.round_trip))


@dataclass
class SimResult:
    capacities_kwh: np.ndarray
    grid_import_wh: np.ndarray
    export_wh: np.ndarray
    charge_wh: np.ndarray
    discharge_wh: np.ndarray
    night_grid_import_wh: np.ndarray
    nonev_grid_import_wh: np.ndarray
    monthly_discharge_wh: np.ndarray
    final_stored_wh: np.ndarray


def simulate(
    net_wh: np.ndarray,
    night_mask: np.ndarray,
    nonev_mask: np.ndarray,
    month_idx: np.ndarray,
    spec: BatterySpec,
) -> SimResult:
    """Run the whole timeline once for every capacity in `spec`.

    `net_wh` is generation minus load per interval. Positive charges the
    battery and exports the rest; negative discharges and imports the rest.
    `night_mask` and `nonev_mask` select which intervals contribute to the
    night-specific accumulators — the timeline itself is never broken up, so
    the battery's state always reflects every night that actually happened.
    """
    caps = spec.usable_wh
    n = len(caps)
    eta = spec.eta
    limit_wh = spec.power_kw * 1000.0 * DT_HOURS

    stored = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    grid = np.zeros(n)
    export = np.zeros(n)
    night_grid = np.zeros(n)
    nonev_grid = np.zeros(n)
    monthly = np.zeros((12, n))

    for t in range(len(net_wh)):
        e = float(net_wh[t])
        if e > 0.0:
            accept = np.minimum(np.minimum(e, limit_wh), (caps - stored) / eta)
            stored += accept * eta
            charge += accept
            export += e - accept
        elif e < 0.0:
            need = -e
            deliver = np.minimum(np.minimum(need, limit_wh), stored * eta)
            stored -= deliver / eta
            discharge += deliver
            shortfall = need - deliver
            grid += shortfall
            if night_mask[t]:
                night_grid += shortfall
            if nonev_mask[t]:
                nonev_grid += shortfall
            monthly[month_idx[t]] += deliver

    return SimResult(
        capacities_kwh=np.asarray(spec.capacities_kwh, dtype=float),
        grid_import_wh=grid,
        export_wh=export,
        charge_wh=charge,
        discharge_wh=discharge,
        night_grid_import_wh=night_grid,
        nonev_grid_import_wh=nonev_grid,
        monthly_discharge_wh=monthly,
        final_stored_wh=stored,
    )
