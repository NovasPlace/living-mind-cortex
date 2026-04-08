"""
state/telemetry_broker.py — Cognitive State Bus

v3.0: Delegates to cortex.state_engine.StateEngine (bounded control-theoretic
      dynamical system). Replaces the unconstrained hormone float soup.

Backward-compat shim:
  - All old telemetry_broker.inject("dopamine", ...) calls still work.
    They are translated to the equivalent state_engine event via LEGACY_MAP.
  - telemetry_broker.state still exposes .dopamine, .cortisol, etc. as
    read-only computed properties (delegates to state_engine internally).
  - cognitive_stance(), snapshot(), mood_bias() all forward to state_engine.

Architecture:
  SecurityPerimeter inflammation   -> inject_event("threat")
  Memory growth                    -> inject_event("memory_hit")
  Agent success/fail               -> inject_event("success") / ("failure")
  Session start/end                -> inject_event("session_start/end")
"""

import time
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from cortex.state_engine import StateEngine, state_engine as _global_engine

# -- Legacy hormone -> state_engine event translation map --------------------
# Maps old hormone injection names to (event_name, sign_multiplier) pairs.
# Preserves call-site behavior without requiring immediate refactors.
LEGACY_MAP = {
    "dopamine":        ("success",      1.0),
    "norepinephrine":  ("novelty",      1.0),
    "cortisol":        ("failure",      1.0),
    "adrenaline":      ("threat",       1.0),
    "melatonin":       ("rest",         0.5),
    "serotonin":       ("feedback_pos", 0.5),
    "oxytocin":        ("feedback_pos", 0.4),
    "acetylcholine":   ("learning",     0.8),
    "endorphin":       ("success",      0.6),
}

# Legacy baseline values for API consumers that read BASELINES directly
BASELINES = {
    "dopamine":       0.65,
    "serotonin":      0.60,
    "cortisol":       0.20,
    "adrenaline":     0.05,
    "melatonin":      0.10,
    "oxytocin":       0.50,
    "norepinephrine": 0.35,
    "acetylcholine":  0.55,
    "endorphin":      0.30,
}

DECAY_RATES = {
    "dopamine":       0.08,
    "serotonin":      0.04,
    "cortisol":       0.06,
    "adrenaline":     0.20,
    "melatonin":      0.05,
    "oxytocin":       0.05,
    "norepinephrine": 0.10,
    "acetylcholine":  0.12,
    "endorphin":      0.07,
}


# -- Backward-compat state shim -----------------------------------------------

class HormoneState:
    """
    Read-only shim that delegates to the underlying StateEngine.
    Legacy code that reads telemetry_broker.state.dopamine still works.
    All values derived from bounded state_engine output.
    """
    def __init__(self, engine: StateEngine):
        self._e = engine

    @property
    def dopamine(self) -> float:
        return self._e.state.drives["reward_drive"]

    @property
    def norepinephrine(self) -> float:
        return self._e.state.drives["novelty_exploration"]

    @property
    def cortisol(self) -> float:
        return self._e.state.loads["stress_load"]

    @property
    def melatonin(self) -> float:
        return self._e.state.loads["sleep_pressure"]

    @property
    def acetylcholine(self) -> float:
        return self._e.state.regulators["focus_stability"]

    @property
    def oxytocin(self) -> float:
        return self._e.state.regulators["social_trust"]

    @property
    def serotonin(self) -> float:
        return self._e.state.regulators["focus_stability"]

    @property
    def adrenaline(self) -> float:
        d = self._e.state
        return round(d.drives["novelty_exploration"] * d.loads["stress_load"], 3)

    @property
    def endorphin(self) -> float:
        d = self._e.state
        return round(d.drives["reward_drive"] * (1 - d.loads["stress_load"]), 3)

    @property
    def valence(self) -> str:
        stance = self._e.cognitive_stance()
        if stance in ("flow", "focused-analytical"):
            return "positive"
        if stance in ("frozen", "winding-down"):
            return "negative"
        return "neutral"

    @property
    def arousal(self) -> float:
        d = self._e.state
        return round(
            d.drives["novelty_exploration"] * 0.5 +
            d.drives["reward_drive"] * 0.3 +
            (1 - d.loads["sleep_pressure"]) * 0.2,
            3
        )

    @property
    def dominant_emotion(self) -> str:
        d = self._e.state
        stress  = d.loads["stress_load"]
        reward  = d.drives["reward_drive"]
        novelty = d.drives["novelty_exploration"]
        if stress > 0.6 and reward < 0.4:
            return "fear"
        if novelty > 0.65:
            return "surprise"
        if stress > 0.5:
            return "anger"
        if reward > 0.70 and stress < 0.3:
            return "joy"
        if d.regulators["focus_stability"] < 0.35:
            return "sadness"
        return "neutral"


# -- Broker -------------------------------------------------------------------

class TelemetryBroker:
    """
    Cognitive state bus. All organs read / write through here.
    Internally delegates to StateEngine for bounded, stable dynamics.
    """

    def __init__(self, engine: Optional[StateEngine] = None):
        self._engine: StateEngine = engine or _global_engine
        self.state = HormoneState(self._engine)
        self._last_memory_count: int = 0
        self._event_log: list = []

    # -- Injection API (backward-compat) --------------------------------------

    def inject(self, hormone: str, delta: float, source: str = "unknown") -> None:
        """
        Legacy injection API. Translates hormone name -> state_engine event.
        Rate-limited and clipped by the engine automatically.
        """
        mapping = LEGACY_MAP.get(hormone)
        if mapping is None:
            return

        event_name, multiplier = mapping
        if delta < 0 and event_name in ("failure", "threat"):
            event_name = "feedback_pos"
            delta = abs(delta)

        self._engine.inject_event(event_name, magnitude=abs(delta) * multiplier * 10,
                                  source=source)

        ts = datetime.now().strftime("%H:%M:%S")
        self._event_log.append({
            "ts": ts, "hormone": hormone,
            "delta": delta, "source": source,
        })
        if len(self._event_log) > 50:
            self._event_log.pop(0)

    def inject_event(self, event_name: str, magnitude: float = 1.0,
                     source: str = "unknown") -> None:
        """Direct typed event injection -- preferred over legacy hormone inject."""
        self._engine.inject_event(event_name, magnitude=magnitude, source=source)

    def inject_emotion(self, emotion: str, source: str = "brain") -> None:
        """Translate emotion label to state events."""
        EMOTION_EVENTS = {
            "joy":         [("success", 0.8), ("feedback_pos", 0.4)],
            "fear":        [("threat", 0.9), ("failure", 0.3)],
            "anger":       [("threat", 0.5), ("failure", 0.4)],
            "surprise":    [("novelty", 0.8)],
            "sadness":     [("failure", 0.5), ("feedback_neg", 0.3)],
            "disgust":     [("feedback_neg", 0.4)],
            "curiosity":   [("novelty", 0.6), ("learning", 0.3)],
            "frustration": [("failure", 0.4), ("feedback_neg", 0.3)],
            "neutral":     [],
        }
        for event_name, mag in EMOTION_EVENTS.get(emotion, []):
            self._engine.inject_event(event_name, magnitude=mag, source=f"{source}:{emotion}")

    # -- Pulse ----------------------------------------------------------------

    async def pulse(self, pulse: int, mem_stats: dict, inflammation: float) -> None:
        """Called every pulse by runtime. Steps the state engine."""
        event: dict = {}

        current_count = mem_stats.get("total", 0)
        if current_count > self._last_memory_count:
            event["memory_hit"] = min(1.0, (current_count - self._last_memory_count) * 0.05)
        self._last_memory_count = current_count

        if inflammation > 0.3:
            event["threat"] = min(1.0, inflammation * 0.5)

        self._engine.step(event)

        s = self._engine.snapshot()
        if s["stress_load"] > 0.6 or s["cognitive_stance"] == "frozen":
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{ts}] [STATE] warning  stress={s['stress_load']:.2f}  "
                f"reward={s['reward_drive']:.2f}  stance={s['cognitive_stance']}"
            )

    # -- Read API -------------------------------------------------------------

    def cognitive_stance(self) -> str:
        return self._engine.cognitive_stance()

    def snapshot(self) -> dict:
        snap = self._engine.snapshot()
        snap.update({
            "dopamine":         self.state.dopamine,
            "cortisol":         self.state.cortisol,
            "norepinephrine":   self.state.norepinephrine,
            "valence":          self.state.valence,
            "arousal":          self.state.arousal,
            "dominant_emotion": self.state.dominant_emotion,
            "cognitive_stance": self.cognitive_stance(),
        })
        return snap

    def mood_bias(self) -> str:
        stance = self.cognitive_stance()
        if stance == "flow":         return "in a creative flow state -- tackle complex problems"
        if stance == "frozen":       return "stressed and stuck -- simplify and break tasks down"
        if stance == "vigilant":     return "alert and precise -- prioritize correctness over speed"
        if stance == "winding-down": return "winding down -- favour summarization and consolidation"
        v = self.state.valence
        a = self.state.arousal
        if v == "positive" and a > 0.6:  return "energized and motivated"
        if v == "positive":               return "calm and content"
        if v == "negative" and self.state.cortisol > 0.5: return "stressed and vigilant"
        return "balanced and steady"


# -- Module-level singleton ---------------------------------------------------
telemetry_broker = TelemetryBroker()
