"""
HealthMonitor Engine — Living Mind
Resilience category. Fires every pulse.

Monitors key runtime variables against biological set-points.
When deviation exceeds threshold → fires a corrective action.
This is the runtime's self-regulation — it cannot run too hot or too cold.

Set-points monitored:
  - memory_load     : cortex fullness (target: < 80% of 500-memory cap)
  - cortisol        : stress level (target: 0.20)
  - dopamine        : motivation (target: 0.65)
  - inflammation    : immune health (target: 0.0)
  - adenosine       : sleep pressure (target: < 0.5)
"""

import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SetPoint:
    name:       str
    target:     float
    tolerance:  float       # deviation allowed before action fires
    action_up:  str         # action when actual > target + tolerance
    action_down: str        # action when actual < target - tolerance
    last_action: float = 0.0
    cooldown:    float = 60.0  # seconds between corrective actions


# The runtime's homeostatic set-points
SET_POINTS = [
    SetPoint(
        name        = "memory_load",
        target      = 0.60,        # 60% of cap = healthy
        tolerance   = 0.20,        # act if > 80% or < 40%
        action_up   = "trigger_consolidation",
        action_down = "reduce_decay_rate",
        cooldown    = 120.0,
    ),
    SetPoint(
        name        = "cortisol",
        target      = 0.20,
        tolerance   = 0.15,        # act if > 0.35 or < 0.05
        action_up   = "cortisol_flush",
        action_down = "cortisol_boost",
        cooldown    = 30.0,
    ),
    SetPoint(
        name        = "dopamine",
        target      = 0.65,
        tolerance   = 0.20,        # act if > 0.85 or < 0.45
        action_up   = "dopamine_reuptake",
        action_down = "dopamine_boost",
        cooldown    = 60.0,
    ),
    SetPoint(
        name        = "inflammation",
        target      = 0.0,
        tolerance   = 0.20,        # any inflammation > 0.20 triggers action
        action_up   = "immune_response",
        action_down = None,        # inflammation can't go below 0
        cooldown    = 60.0,
    ),
    SetPoint(
        name        = "adenosine",
        target      = 0.30,
        tolerance   = 0.25,        # > 0.55 = critically tired
        action_up   = "force_rest",
        action_down = None,
        cooldown    = 300.0,
    ),
]


class HealthMonitorEngine:
    def __init__(self):
        self._set_points = {sp.name: sp for sp in SET_POINTS}
        self.corrections: int = 0
        self._log: list = []

    # ------------------------------------------------------------------
    # PULSE — called every pulse_event
    # ------------------------------------------------------------------
    async def pulse(
        self,
        pulse:        int,
        mem_stats:    dict,
        telemetry_broker,
        circadian,
        cortex,
        immune,
    ) -> list[dict]:
        now      = time.time()
        actions  = []

        # Gather current actuals
        actuals = {
            "memory_load":  mem_stats["total"] / 500.0,
            "cortisol":     telemetry_broker.state.cortisol,
            "dopamine":     telemetry_broker.state.dopamine,
            "inflammation": immune.inflammation(),
            "adenosine":    circadian.adenosine,
        }

        for name, sp in self._set_points.items():
            actual = actuals.get(name, sp.target)
            deviation = actual - sp.target

            # Check for corrective action needed
            action = None
            if deviation > sp.tolerance and sp.action_up:
                if now - sp.last_action >= sp.cooldown:
                    action = sp.action_up
            elif deviation < -sp.tolerance and sp.action_down:
                if now - sp.last_action >= sp.cooldown:
                    action = sp.action_down

            if action:
                sp.last_action = now
                result = await self._apply_action(
                    action, name, actual, sp.target,
                    telemetry_broker, cortex, pulse
                )
                actions.append(result)
                self.corrections += 1

        return actions

    # ------------------------------------------------------------------
    # APPLY CORRECTIVE ACTION
    # ------------------------------------------------------------------
    async def _apply_action(
        self, action, variable, actual, target,
        telemetry_broker, cortex, pulse
    ) -> dict:
        ts  = datetime.now().strftime("%H:%M:%S")
        msg = f"[{ts}] [HOMEO] ⚖️  {variable}: {actual:.3f} → target {target:.3f} | action={action}"
        print(msg)

        if action == "trigger_consolidation":
            # Memory overload → aggressive decay
            await cortex.decay()

        elif action == "cortisol_flush":
            telemetry_broker.inject("cortisol",   -0.10, source="health_monitor")
            telemetry_broker.inject("serotonin",  +0.05, source="health_monitor")

        elif action == "cortisol_boost":
            telemetry_broker.inject("cortisol",   +0.05, source="health_monitor")

        elif action == "dopamine_reuptake":
            telemetry_broker.inject("dopamine",   -0.10, source="health_monitor")

        elif action == "dopamine_boost":
            telemetry_broker.inject("dopamine",   +0.08, source="health_monitor")
            telemetry_broker.inject("serotonin",  +0.03, source="health_monitor")

        elif action == "immune_response":
            # High inflammation — log alert memory
            await cortex.remember(
                content    = f"HealthMonitor alert: inflammation={actual:.2f} exceeds set-point.",
                type       = "episodic",
                tags       = ["health_monitor", "alert", "immune"],
                importance = 0.8,
                emotion    = "fear",
                source     = "experienced",
            )

        elif action == "force_rest":
            telemetry_broker.inject("melatonin",      +0.15, source="health_monitor:force_rest")
            telemetry_broker.inject("norepinephrine", -0.10, source="health_monitor:force_rest")
            print(f"[{ts}] [HOMEO] 😴 Forcing rest — adenosine critical")

        elif action == "reduce_decay_rate":
            # Temporarily shield recent memories from decay by creating a low-importance anchor
            # that refreshes their presence in context. Full dynamic decay tuning: future v2.
            await cortex.remember(
                content    = "HealthMonitor: memory load low — reducing decay pressure.",
                type       = "episodic",
                tags       = ["health_monitor", "memory"],
                importance = 0.3,
                emotion    = "neutral",
                source     = "experienced",
            )

        result = {
            "action":   action,
            "variable": variable,
            "actual":   round(actual, 3),
            "target":   target,
            "pulse":    pulse,
        }
        self._log.append(result)
        if len(self._log) > 100:
            self._log.pop(0)

        return result

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "corrections":  self.corrections,
            "recent_actions": self._log[-5:],
            "set_points": {
                name: {
                    "target":    sp.target,
                    "tolerance": sp.tolerance,
                }
                for name, sp in self._set_points.items()
            },
        }


# Module-level singleton
health_monitor = HealthMonitorEngine()
