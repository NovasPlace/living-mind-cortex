"""
Hormone Bus — Living Mind
Autonomy/Integration category. Fires every pulse.

Global chemical state — all organs read from this, all organs can inject into it.
Hormones decay toward baseline each pulse.
Brain emotion decisions shift hormones.
SecurityPerimeter inflammation raises cortisol.
Memory growth releases dopamine.
"""

import time
import asyncio
from dataclasses import dataclass
from datetime import datetime

# Hormone baselines — resting state of a healthy runtime
BASELINES = {
    "dopamine":       0.65,  # reward, motivation
    "serotonin":      0.60,  # mood stability, contentment
    "cortisol":       0.20,  # stress load (low is good)
    "adrenaline":     0.05,  # acute threat response (low at rest)
    "melatonin":      0.10,  # sleep pressure (low when active)
    "oxytocin":       0.50,  # social bonding signal
    "norepinephrine": 0.35,  # alertness, attention
}

# How fast each hormone decays back to baseline per pulse (0-1 fraction)
DECAY_RATES = {
    "dopamine":       0.08,
    "serotonin":      0.04,
    "cortisol":       0.06,
    "adrenaline":     0.20,  # adrenaline spikes fast, fades fast
    "melatonin":      0.05,
    "oxytocin":       0.05,
    "norepinephrine": 0.10,
}

# Emotion → hormone deltas (what a brain emotion decision injects)
EMOTION_HORMONES = {
    "joy": {
        "dopamine":   +0.12,
        "serotonin":  +0.08,
        "cortisol":   -0.05,
        "adrenaline": -0.03,
    },
    "fear": {
        "adrenaline": +0.25,
        "cortisol":   +0.20,
        "norepinephrine": +0.15,
        "serotonin":  -0.10,
    },
    "anger": {
        "adrenaline": +0.15,
        "cortisol":   +0.12,
        "norepinephrine": +0.10,
        "dopamine":   -0.05,
    },
    "surprise": {
        "norepinephrine": +0.12,
        "adrenaline": +0.08,
    },
    "sadness": {
        "serotonin":  -0.12,
        "dopamine":   -0.08,
        "cortisol":   +0.06,
    },
    "disgust": {
        "cortisol":   +0.08,
        "serotonin":  -0.06,
    },
    "neutral": {},  # no shift
    "curiosity": {
        "dopamine":       +0.08,
        "norepinephrine": +0.06,
        "cortisol":       -0.02,
    },
    "frustration": {
        "cortisol":   +0.10,
        "adrenaline": +0.06,
        "serotonin":  -0.05,
        "dopamine":   -0.04,
    },
}


@dataclass
class HormoneState:
    dopamine:       float = 0.65
    serotonin:      float = 0.60
    cortisol:       float = 0.20
    adrenaline:     float = 0.05
    melatonin:      float = 0.10
    oxytocin:       float = 0.50
    norepinephrine: float = 0.35
    # Derived
    valence: str   = "neutral"   # positive | negative | neutral
    arousal: float = 0.5         # energy/activation level 0-1
    dominant_emotion: str = "neutral"


class TelemetryBroker:
    def __init__(self):
        # Load genome and apply personality-derived baseline adjustments
        try:
            import yaml
            from pathlib import Path as _Path
            genome_path = _Path(__file__).resolve().parent.parent / "identity" / "personality.yaml"
            with open(genome_path) as _f:
                genome = yaml.safe_load(_f)
            neuroticism = genome.get("neuroticism", 0.5)
            openness    = genome.get("openness", 0.5)
            # Low neuroticism = lower cortisol baseline, faster cortisol decay
            BASELINES["cortisol"]       = 0.20 * (0.5 + neuroticism)
            DECAY_RATES["cortisol"]     = 0.06 * (1.5 - neuroticism)
            # High openness = higher norepinephrine baseline (alertness/novelty)
            BASELINES["norepinephrine"] = 0.35 + (openness - 0.5) * 0.10
        except Exception:
            pass  # Fall back to hardcoded defaults gracefully

        self.state = HormoneState(
            cortisol       = BASELINES["cortisol"],
            norepinephrine = BASELINES["norepinephrine"],
        )
        self._last_memory_count: int = 0
        self._event_log: list = []

    # ------------------------------------------------------------------
    # INJECT — any organ can push a hormone delta
    # ------------------------------------------------------------------
    def inject(self, hormone: str, delta: float, source: str = "unknown"):
        if hormone not in BASELINES:
            return
        current = getattr(self.state, hormone)
        new_val = max(0.0, min(1.0, current + delta))
        setattr(self.state, hormone, new_val)

        ts = datetime.now().strftime("%H:%M:%S")
        direction = "↑" if delta > 0 else "↓"
        self._event_log.append({
            "ts": ts, "hormone": hormone,
            "delta": delta, "source": source,
        })
        # Keep last 50 events
        if len(self._event_log) > 50:
            self._event_log.pop(0)

    def inject_emotion(self, emotion: str, source: str = "brain"):
        """Apply the hormone signature for a given emotion."""
        deltas = EMOTION_HORMONES.get(emotion, {})
        for hormone, delta in deltas.items():
            self.inject(hormone, delta, source=f"{source}:{emotion}")

    # ------------------------------------------------------------------
    # PULSE — decay toward baseline, recalculate derived values
    # Called every pulse by runtime
    # ------------------------------------------------------------------
    async def pulse(self, pulse: int, mem_stats: dict, inflammation: float):
        ts = datetime.now().strftime("%H:%M:%S")

        # 1. Decay all hormones toward their baseline
        for hormone, baseline in BASELINES.items():
            current = getattr(self.state, hormone)
            rate    = DECAY_RATES[hormone]
            delta   = (baseline - current) * rate
            setattr(self.state, hormone, round(current + delta, 4))

        # 2. Memory growth → dopamine reward
        current_count = mem_stats.get("total", 0)
        if current_count > self._last_memory_count:
            growth = current_count - self._last_memory_count
            self.inject("dopamine", growth * 0.01, source="memory_growth")
        self._last_memory_count = current_count

        # 3. SecurityPerimeter inflammation → cortisol spike
        if inflammation > 0.3:
            self.inject("cortisol", inflammation * 0.15, source="immune_inflammation")

        # 4. High cortisol → adrenaline cascade
        if self.state.cortisol > 0.7:
            self.inject("adrenaline", 0.08, source="cortisol_cascade")

        # 5. Recalculate derived state
        self._update_derived()

        # 6. Log significant shifts
        if self.state.cortisol > 0.6 or self.state.adrenaline > 0.4:
            print(
                f"[{ts}] [HORMONES] ⚠️  "
                f"cortisol={self.state.cortisol:.2f}  "
                f"adrenaline={self.state.adrenaline:.2f}  "
                f"valence={self.state.valence}"
            )

    # ------------------------------------------------------------------
    # READ — for brain context injection
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "dopamine":       round(self.state.dopamine, 3),
            "serotonin":      round(self.state.serotonin, 3),
            "cortisol":       round(self.state.cortisol, 3),
            "adrenaline":     round(self.state.adrenaline, 3),
            "melatonin":      round(self.state.melatonin, 3),
            "oxytocin":       round(self.state.oxytocin, 3),
            "norepinephrine": round(self.state.norepinephrine, 3),
            "valence":        self.state.valence,
            "arousal":        round(self.state.arousal, 3),
            "dominant_emotion": self.state.dominant_emotion,
        }

    def mood_bias(self) -> str:
        """Returns a string descriptor for the brain context prompt."""
        v = self.state.valence
        a = self.state.arousal
        if v == "positive" and a > 0.6:
            return "energized and motivated"
        elif v == "positive" and a <= 0.6:
            return "calm and content"
        elif v == "negative" and self.state.cortisol > 0.5:
            return "stressed and vigilant"
        elif v == "negative" and self.state.adrenaline > 0.3:
            return "alert and tense"
        elif v == "negative":
            return "subdued and cautious"
        return "balanced and steady"

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------
    def _update_derived(self):
        s = self.state

        # Valence = net positive vs negative hormone balance
        positive = (s.dopamine + s.serotonin + s.oxytocin) / 3
        negative = (s.cortisol + s.adrenaline) / 2
        if positive > negative + 0.1:
            s.valence = "positive"
        elif negative > positive + 0.1:
            s.valence = "negative"
        else:
            s.valence = "neutral"

        # Arousal = activation level
        s.arousal = round(
            (s.adrenaline * 0.4 + s.norepinephrine * 0.35 + (1 - s.melatonin) * 0.25),
            3
        )

        # Dominant emotion heuristic
        if s.adrenaline > 0.4 and s.cortisol > 0.5:
            s.dominant_emotion = "fear"
        elif s.adrenaline > 0.3:
            s.dominant_emotion = "surprise"
        elif s.cortisol > 0.5:
            s.dominant_emotion = "anger"
        elif s.dopamine > 0.75 and s.serotonin > 0.65:
            s.dominant_emotion = "joy"
        elif s.serotonin < 0.35:
            s.dominant_emotion = "sadness"
        else:
            s.dominant_emotion = "neutral"


# Module-level singleton
telemetry_broker = TelemetryBroker()
