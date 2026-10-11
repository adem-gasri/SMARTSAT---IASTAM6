"""
simulation/satellite_sim.py

Simplified simulation of on-board satellite resources: battery, memory,
and ground-station contact windows. Used to reason about latency (a
detection made outside a contact window must wait for the next one) and
about the bandwidth available during a window.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SatelliteState:
    battery_pct: float = 100.0
    memory_mb: float = 500.0
    time_s: float = 0.0

    # orbit parameters (defaults: low-earth-orbit-like cadence)
    orbit_period_s: float = 90 * 60        # 90 minutes
    contact_duration_s: float = 5 * 60     # 5 minutes per orbit
    downlink_rate_kbps: float = 512.0      # bandwidth during contact

    # energy model (rough, illustrative)
    energy_cost_per_inference_pct: float = 0.05
    energy_cost_per_mb_stored_pct: float = 0.002

    def in_contact_window(self) -> bool:
        phase = self.time_s % self.orbit_period_s
        return phase < self.contact_duration_s

    def available_bandwidth_kb(self, dt_s: float) -> float:
        """Kilobytes transmittable in dt_s seconds, if in a contact window."""
        if not self.in_contact_window():
            return 0.0
        return (self.downlink_rate_kbps / 8.0) * dt_s

    def seconds_until_next_contact(self) -> float:
        """Waiting time before the next contact window opens (0 if in one)."""
        if self.in_contact_window():
            return 0.0
        phase = self.time_s % self.orbit_period_s
        return self.orbit_period_s - phase

    def consume_for_inference(self):
        self.battery_pct = max(0.0, self.battery_pct - self.energy_cost_per_inference_pct)

    def store(self, size_mb: float) -> bool:
        """Returns True if there was enough memory to store the item."""
        if self.memory_mb - size_mb < 0:
            return False
        self.memory_mb -= size_mb
        self.battery_pct = max(0.0, self.battery_pct - self.energy_cost_per_mb_stored_pct * size_mb)
        return True

    def tick(self, dt_s: float = 1.0):
        self.time_s += dt_s


def simulate_contact_schedule(total_duration_s: float,
                               orbit_period_s: float = 90 * 60,
                               contact_duration_s: float = 5 * 60) -> List[Tuple[float, float]]:
    """Returns a list of (start, end) contact windows within the horizon."""
    windows = []
    t = 0.0
    while t < total_duration_s:
        windows.append((t, min(t + contact_duration_s, total_duration_s)))
        t += orbit_period_s
    return windows


if __name__ == "__main__":
    sat = SatelliteState()
    windows = simulate_contact_schedule(total_duration_s=6 * 3600)  # 6-hour simulation
    print(f"{len(windows)} contact windows in a 6-hour horizon")
    for w in windows:
        print(f"  window: {w[0]/60:.1f} min -> {w[1]/60:.1f} min")

    print("In contact at t=0?", sat.in_contact_window())
    sat.time_s = 40 * 60
    print("In contact at t=40min?", sat.in_contact_window())
    print(f"Wait until next contact: {sat.seconds_until_next_contact()/60:.1f} min")