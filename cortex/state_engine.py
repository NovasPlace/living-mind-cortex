"""
cortex/state_engine.py — Bounded Adaptive State Vector Engine

Replaces the unconstrained hormone bus with a formalized, control-theoretic
dynamical system. Three typed state classes enforce different dynamics:

  drives     → compete via softmax normalization (push behavior)
  loads      → accumulate + decay slowly (constrain system)
  regulators → modulate transfer functions (slow drift toward baseline)

All variables ∈ [0.0, 1.0], rate-limited at ±0.1/step, clipped after every
update cycle. This system cannot explode. It can only evolve within bounds.

State update cycle per step:
  1. Apply external event input (weighted deltas)
  2. Internal coupling  (drive competition, load suppression, overstimulation damping)
  3. Decay              (drives fast, loads slow, regulators drift)
  4. Rate limiting      (cap per-step delta to max_delta)
  5. Final clip         (hard [0, 1] bounds)

Thermorphic bridge:
  temperature = base_temp + 0.7 * stress_load
  retention   = base_ret  + 0.5 * reward_drive
  noise       = 0.1       + 0.3 * novelty_exploration

April 2026 — Architecture hardening pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _rate_limit(old: float, new: float, max_delta: float = 0.10) -> float:
    """Prevent shocks: cap per-step change to ±max_delta."""
    delta = new - old
    delta = max(-max_delta, min(max_delta, delta))
    return old + delta


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ── State Types ───────────────────────────────────────────────────────────────

@dataclass
class StateVector:
    """
    Typed, bounded cognitive state space.

    drives      – compete (softmax normalised); push agent behavior
    loads       – accumulate under pressure; suppress drives
    regulators  – modulate transfer functions; slow homeostatic drift
    """
    drives: Dict[str, float] = field(default_factory=lambda: {
        "reward_drive":       0.30,  # reward, motivation (was: dopamine)
        "novelty_exploration": 0.30,  # curiosity / alertness (was: norepinephrine)
    })
    loads: Dict[str, float] = field(default_factory=lambda: {
        "stress_load":    0.20,  # stress accumulation (was: cortisol)
        "sleep_pressure": 0.10,  # fatigue / sleep homeostasis (was: melatonin)
    })
    regulators: Dict[str, float] = field(default_factory=lambda: {
        "focus_stability": 0.55,  # attention coherence (was: acetylcholine)
        "social_trust":    0.50,  # bonding / cooperation signal (was: oxytocin)
    })


# ── Event Input Weights ───────────────────────────────────────────────────────
# Maps named event signals → (state_group, state_key, weight)
# Covers the common stimulus vocabulary from the old hormone injection calls.

EVENT_WEIGHTS: Dict[str, list[Tuple[str, str, float]]] = {
    "success":     [("drives", "reward_drive",       +0.30)],
    "novelty":     [("drives", "novelty_exploration", +0.20)],
    "failure":     [("loads",  "stress_load",         +0.40),
                    ("drives", "reward_drive",        -0.05)],
    "threat":      [("loads",  "stress_load",         +0.20),
                    ("drives", "novelty_exploration", -0.05)],
    "rest":        [("loads",  "sleep_pressure",      -0.10),
                    ("loads",  "stress_load",         -0.05)],
    "memory_hit":  [("drives", "reward_drive",       +0.03)],
    "memory_miss": [("loads",  "stress_load",         +0.02)],
    "feedback_pos":[("drives", "reward_drive",       +0.08),
                    ("loads",  "stress_load",         -0.04)],
    "feedback_neg":[("loads",  "stress_load",         +0.08),
                    ("drives", "reward_drive",        -0.04)],
    "session_start":[("drives","reward_drive",        +0.05),
                     ("drives","novelty_exploration",  +0.03)],
    "session_end":  [("loads", "stress_load",         -0.03)],
    "learning":     [("regulators","focus_stability", +0.04),
                     ("drives",    "reward_drive",    +0.03)],
}

# Decay multipliers per step
DECAY = {
    # drives decay faster (reset after stimulus fades)
    "reward_drive":       0.85,
    "novelty_exploration": 0.88,
    # loads decay slower (stress lingers)
    "stress_load":        0.92,
    "sleep_pressure":     0.97,
    # regulators drift very slowly toward baseline
    "focus_stability":    None,  # lerp handled separately
    "social_trust":       None,
}

REGULATOR_BASELINES = {
    "focus_stability": 0.55,
    "social_trust":    0.50,
}

REGULATOR_DRIFT = 0.05  # how fast regulators pull back to baseline


# ── Engine ────────────────────────────────────────────────────────────────────

class StateEngine:
    """
    Bounded, stable dynamical system for Living Mind cognitive state.

    Usage:
        engine = StateEngine()
        state  = engine.step({"success": 1.0, "novelty": 0.5})
        delta  = engine.get_delta()
        thermo = engine.to_thermo()
    """

    def __init__(self, max_delta: float = 0.10):
        self.state      = StateVector()
        self.prev_state = copy.deepcopy(self.state)
        self.max_delta  = max_delta
        self._step_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, event: Dict[str, float] | None = None) -> StateVector:
        """
        Advance one timestep.
        event: dict of named stimuli with magnitudes 0-1 (e.g. {"success": 1.0}).
        """
        self.prev_state = copy.deepcopy(self.state)

        self._apply_input(event or {})
        self._apply_internal_dynamics()
        self._apply_decay()
        self._apply_rate_limits()
        self._clip_all()

        self._step_count += 1
        return self.state

    def inject(self, group: str, key: str, delta: float, source: str = "unknown") -> None:
        """
        Direct delta injection. Rate-limited + clipped immediately.
        Equivalent to old telemetry_broker.inject() but typed.
        """
        target = getattr(self.state, group, None)
        if target is None or key not in target:
            return
        raw      = _rate_limit(target[key], target[key] + delta, self.max_delta)
        target[key] = _clamp(raw)

    def inject_event(self, event_name: str, magnitude: float = 1.0,
                     source: str = "unknown") -> None:
        """
        Fire a named event (from EVENT_WEIGHTS vocab).
        Scales all deltas by magnitude.
        """
        for group, key, weight in EVENT_WEIGHTS.get(event_name, []):
            self.inject(group, key, weight * magnitude, source=source)

    def get_delta(self) -> Dict[str, Dict[str, float]]:
        """Compute state diff since last step."""
        return self._compute_delta(self.prev_state, self.state)

    def to_thermo(self) -> Dict[str, float]:
        """
        Map state onto Thermorphic physics parameters.
        Provides explicit temperature, retention, and noise values
        so the coupling is declared, not implicit.
        """
        stress  = self.state.loads["stress_load"]
        reward  = self.state.drives["reward_drive"]
        novelty = self.state.drives["novelty_exploration"]
        return {
            "temperature": round(_clamp(0.20 + 0.70 * stress),  3),
            "retention":   round(_clamp(0.30 + 0.50 * reward),  3),
            "noise":       round(_clamp(0.10 + 0.30 * novelty), 3),
        }

    def cognitive_stance(self) -> str:
        """Named cognitive operating mode (mirrors old telemetry_broker.cognitive_stance)."""
        d = self.state.drives
        l = self.state.loads
        r = self.state.regulators

        stress  = l["stress_load"]
        reward  = d["reward_drive"]
        novelty = d["novelty_exploration"]
        focus   = r["focus_stability"]
        sleep   = l["sleep_pressure"]

        if stress > 0.6 and reward < 0.4:
            return "frozen"
        if reward > 0.70 and stress < 0.3:
            return "flow"
        if novelty > 0.65:
            return "vigilant"
        if sleep > 0.5:
            return "winding-down"
        if focus > 0.65 and novelty > 0.5:
            return "focused-analytical"
        return "balanced"

    def snapshot(self) -> Dict[str, Any]:
        """Full state snapshot dict for telemetry, API, and DB writes."""
        d = self.state.drives
        l = self.state.loads
        r = self.state.regulators
        return {
            "reward_drive":       round(d["reward_drive"],        3),
            "novelty_exploration": round(d["novelty_exploration"], 3),
            "stress_load":         round(l["stress_load"],        3),
            "sleep_pressure":      round(l["sleep_pressure"],     3),
            "focus_stability":     round(r["focus_stability"],    3),
            "social_trust":        round(r["social_trust"],       3),
            "cognitive_stance":    self.cognitive_stance(),
            "thermo":              self.to_thermo(),
            "step":                self._step_count,
        }

    def stability_score(self, window: list[StateVector]) -> float:
        """
        Compute variance-based stability score over a window of past states.
        Lower is more stable. Used by evolution fitness function.
        """
        if len(window) < 2:
            return 0.0

        all_vals: Dict[str, list[float]] = {}
        for sv in window:
            for group in ["drives", "loads", "regulators"]:
                for k, v in getattr(sv, group).items():
                    all_vals.setdefault(k, []).append(v)

        total_variance = sum(
            self._variance(vals) for vals in all_vals.values()
        )
        return round(total_variance / max(len(all_vals), 1), 5)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _apply_input(self, event: Dict[str, float]) -> None:
        for name, magnitude in event.items():
            for group, key, weight in EVENT_WEIGHTS.get(name, []):
                target = getattr(self.state, group)
                if key in target:
                    target[key] += weight * magnitude

    def _apply_internal_dynamics(self) -> None:
        d = self.state.drives
        l = self.state.loads
        r = self.state.regulators

        # 1. Drive competition — softmax normalization (prevents dual-max runaway)
        total = sum(d.values()) + 1e-6
        for k in d:
            d[k] /= total

        # 2. Load suppression — stress weighs on reward, sleep weighs on focus
        d["reward_drive"]    *= (1.0 - l["stress_load"])
        r["focus_stability"] *= (1.0 - l["sleep_pressure"])

        # 3. Overstimulation damping — exponential bleed when reward saturates
        if d["reward_drive"] > 0.9:
            d["reward_drive"] *= 0.95

    def _apply_decay(self) -> None:
        d = self.state.drives
        l = self.state.loads
        r = self.state.regulators

        for k, rate in DECAY.items():
            if rate is None:
                continue  # handled below (regulators)
            if k in d:
                d[k] *= rate
            elif k in l:
                l[k] *= rate

        # Regulators drift toward their set-point
        for k, baseline in REGULATOR_BASELINES.items():
            r[k] = _lerp(r[k], baseline, REGULATOR_DRIFT)

    def _apply_rate_limits(self) -> None:
        for group in ["drives", "loads", "regulators"]:
            current = getattr(self.state, group)
            prev    = getattr(self.prev_state, group)
            for k in current:
                current[k] = _rate_limit(prev.get(k, current[k]),
                                          current[k], self.max_delta)

    def _clip_all(self) -> None:
        for group in ["drives", "loads", "regulators"]:
            g = getattr(self.state, group)
            for k in g:
                g[k] = _clamp(g[k])

    def _compute_delta(self, before: StateVector,
                       after: StateVector) -> Dict[str, Dict[str, float]]:
        delta: Dict[str, Dict[str, float]] = {}
        for group in ["drives", "loads", "regulators"]:
            b = getattr(before, group)
            a = getattr(after,  group)
            delta[group] = {k: round(a[k] - b.get(k, 0.0), 5) for k in a}
        return delta

    @staticmethod
    def _variance(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        return sum((v - mean) ** 2 for v in vals) / len(vals)


# ── Module-level singleton (drop-in replacement for telemetry_broker) ─────────
state_engine = StateEngine()
