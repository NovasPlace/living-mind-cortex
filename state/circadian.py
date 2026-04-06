"""
Circadian Clock — Living Mind
Autonomy/Integration category. Fires every pulse.

Maps real-world clock time to four biological phases:
dawn (05-09) → day (09-18) → evening (18-22) → night (22-05)

Each phase modulates:
  - Melatonin injection into hormone bus
  - Adenosine (sleep pressure) build/clear
  - Pulse interval scaling (slower at night)
  - Brain fire rate (less frequent thinking at night)
  - Memory consolidation intensity (peaks at night)
"""

import time
from datetime import datetime


# Phase boundaries (24h clock)
PHASES = {
    "dawn":    (5,  9),   # 05:00 - 09:00
    "day":     (9,  18),  # 09:00 - 18:00
    "evening": (18, 22),  # 18:00 - 22:00
    "night":   (22, 5),   # 22:00 - 05:00 next day
}

# How each phase affects hormone bus
PHASE_HORMONE_DELTAS = {
    "dawn":    {"melatonin": -0.06, "norepinephrine": +0.04, "cortisol": +0.03},
    "day":     {"melatonin": -0.03, "norepinephrine": +0.01},
    "evening": {"melatonin": +0.05, "norepinephrine": -0.02, "serotonin": -0.01},
    "night":   {"melatonin": +0.08, "norepinephrine": -0.04, "cortisol": -0.02},
}

# Pulse interval multipliers per phase (base is 10s)
PHASE_PULSE_SCALE = {
    "dawn":    0.9,   # slightly faster — waking up
    "day":     1.0,   # normal
    "evening": 1.1,   # winding down
    "night":   1.4,   # slower — resting
}

# Brain fire rate modifiers (fraction of normal frequency)
PHASE_BRAIN_RATE = {
    "dawn":    0.8,   # thinking starting to ramp
    "day":     1.0,   # full cognition
    "evening": 0.7,   # slowing
    "night":   0.3,   # minimal — mostly dreaming, not deciding
}

# Adenosine (sleep pressure) change per pulse per phase
ADENOSINE_RATE = {
    "dawn":    +0.004,   # clearing fast at wake
    "day":     +0.002,   # builds during wakefulness
    "evening": +0.003,   # building more
    "night":   -0.010,   # clears during sleep
}


class CircadianClock:
    def __init__(self):
        self.phase:           str   = "day"
        self.hour_of_day:     int   = datetime.now().hour
        self.adenosine:       float = 0.0   # 0 = fully rested, 1 = critically tired
        self.phase_start:     float = time.time()
        self.transitions:     int   = 0
        self._prev_phase:     str   = "day"

    # ------------------------------------------------------------------
    # PULSE — called every pulse_event
    # ------------------------------------------------------------------
    async def pulse(self, pulse: int, telemetry_broker) -> dict:
        now      = datetime.now()
        hour     = now.hour
        self.hour_of_day = hour

        # Determine current phase
        new_phase = self._compute_phase(hour)

        # Log phase transitions
        if new_phase != self.phase:
            self._prev_phase = self.phase
            self.phase       = new_phase
            self.phase_start = time.time()
            self.transitions += 1
            ts = now.strftime("%H:%M:%S")
            print(
                f"[{ts}] [CIRCADIAN] 🌙 Phase transition: "
                f"{self._prev_phase} → {self.phase}"
            )

        # Apply hormone deltas for current phase
        deltas = PHASE_HORMONE_DELTAS.get(self.phase, {})
        for hormone, delta in deltas.items():
            telemetry_broker.inject(hormone, delta * 0.1, source=f"circadian:{self.phase}")

        # Adenosine dynamics
        rate              = ADENOSINE_RATE.get(self.phase, 0.001)
        self.adenosine    = max(0.0, min(1.0, self.adenosine + rate))

        # High adenosine → inject melatonin spike + cortisol reduction
        if self.adenosine > 0.6:
            telemetry_broker.inject("melatonin", 0.05, source="adenosine_pressure")
            telemetry_broker.inject("cortisol",  -0.03, source="adenosine_pressure")

        return self.snapshot()

    # ------------------------------------------------------------------
    # PULSE SCALE — runtime uses this to adjust pulse_event speed
    # ------------------------------------------------------------------
    def pulse_scale(self) -> float:
        return PHASE_PULSE_SCALE.get(self.phase, 1.0)

    # ------------------------------------------------------------------
    # BRAIN RATE — controls how often brain fires (fraction)
    # ------------------------------------------------------------------
    def brain_rate(self) -> float:
        return PHASE_BRAIN_RATE.get(self.phase, 1.0)

    # ------------------------------------------------------------------
    # Should the runtime consolidate memory now?
    # Night phase = peak consolidation intensity
    # ------------------------------------------------------------------
    def consolidation_intensity(self) -> float:
        return {
            "dawn":    0.6,
            "day":     0.4,
            "evening": 0.7,
            "night":   1.0,   # deep consolidation during sleep
        }.get(self.phase, 0.5)

    # ------------------------------------------------------------------
    # SNAPSHOT — for vitals and API
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "phase":                  self.phase,
            "hour_of_day":            self.hour_of_day,
            "adenosine":              round(self.adenosine, 3),
            "pulse_scale":            self.pulse_scale(),
            "brain_rate":             self.brain_rate(),
            "consolidation_intensity": self.consolidation_intensity(),
            "transitions":            self.transitions,
        }

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------
    def _compute_phase(self, hour: int) -> str:
        if 5 <= hour < 9:
            return "dawn"
        elif 9 <= hour < 18:
            return "day"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"


# Module-level singleton
circadian = CircadianClock()
