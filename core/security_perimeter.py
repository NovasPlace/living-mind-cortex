"""
SecurityPerimeter System — Living Mind
Defense category. Fires every pulse (Phase 7).

Responsibilities:
- Track health score per pipeline/organ (0–100)
- Decay health on failure, recover on success
- Auto-quarantine after 3 consecutive failures
- Rate limit: max 5 fires per minute per organ
- Auto-heal: release quarantine after recovery window
"""

import time
from dataclasses import dataclass, field
from collections import defaultdict, deque


QUARANTINE_THRESHOLD   = 3      # consecutive failures before quarantine
RECOVERY_WINDOW        = 300    # seconds before a quarantined organ can attempt return
RATE_LIMIT_WINDOW      = 60     # seconds
RATE_LIMIT_MAX_FIRES   = 5      # max fires per organ per window
HEALTH_DECAY_ON_FAIL   = 20     # health points lost per failure
HEALTH_RECOVER_ON_OK   = 10     # health points gained per success
HEALTH_PASSIVE_DECAY   = 0.5    # points lost per hour of no activity


@dataclass
class OrganHealth:
    name: str
    category: str           = "Unknown"
    health: float           = 100.0     # 0–100
    status: str             = "healthy" # healthy | degraded | quarantined
    consecutive_failures: int = 0
    total_failures: int     = 0
    total_successes: int    = 0
    quarantined_at: float   = 0.0
    last_fire: float        = 0.0
    fire_times: deque       = field(default_factory=lambda: deque(maxlen=100))


class SecurityPerimeterSystem:
    def __init__(self):
        self._organs: dict[str, OrganHealth] = {}
        self._inflammation: float = 0.0  # system-wide stress (0–1)

    # ------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------
    def register(self, name: str, category: str = "Unknown"):
        if name not in self._organs:
            self._organs[name] = OrganHealth(
                name=name, category=category, last_fire=time.time()
            )

    # ------------------------------------------------------------------
    # REPORT OUTCOME — called after every organ execution
    # ------------------------------------------------------------------
    def report(self, name: str, success: bool, category: str = "Unknown") -> dict:
        if name not in self._organs:
            self.register(name, category)

        organ = self._organs[name]
        now = time.time()
        organ.last_fire = now
        organ.fire_times.append(now)

        if success:
            organ.consecutive_failures = 0
            organ.total_successes += 1
            organ.health = min(100.0, organ.health + HEALTH_RECOVER_ON_OK)

            # Attempt release from quarantine
            if organ.status == "quarantined":
                if now - organ.quarantined_at >= RECOVERY_WINDOW:
                    organ.status = "healthy"
                    organ.health = 50.0  # restart at half health
                    self._log(f"[IMMUNE] ✅ {name} released from quarantine — healing")
        else:
            organ.consecutive_failures += 1
            organ.total_failures += 1
            organ.health = max(0.0, organ.health - HEALTH_DECAY_ON_FAIL)
            self._update_status(organ, now)

        self._recalculate_inflammation()

        return {
            "name":   name,
            "health": round(organ.health, 1),
            "status": organ.status,
        }

    # ------------------------------------------------------------------
    # RATE CHECK — call before firing an organ
    # ------------------------------------------------------------------
    def is_rate_limited(self, name: str) -> bool:
        if name not in self._organs:
            return False
        organ = self._organs[name]
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW
        recent = sum(1 for t in organ.fire_times if t >= window_start)
        return recent >= RATE_LIMIT_MAX_FIRES

    def is_quarantined(self, name: str) -> bool:
        if name not in self._organs:
            return False
        return self._organs[name].status == "quarantined"

    # ------------------------------------------------------------------
    # PATROL — called every pulse (Phase 7)
    # ------------------------------------------------------------------
    async def patrol(self, pulse: int) -> dict:
        now = time.time()
        report = {
            "pulse":          pulse,
            "total_organs":   len(self._organs),
            "healthy":        0,
            "degraded":       0,
            "quarantined":    0,
            "inflammation":   round(self._inflammation, 3),
            "events":         [],
        }

        for org in self._organs.values():
            # Passive health decay for long-idle organs
            idle = now - org.last_fire
            if idle > 3600 and org.status == "healthy":
                decay = HEALTH_PASSIVE_DECAY * (idle / 3600)
                org.health = max(0.0, org.health - decay)
                if org.health < 60:
                    org.status = "degraded"

            # Auto-release quarantined organs that missed their recovery probe window
            if org.status == "quarantined":
                elapsed = now - org.quarantined_at
                if elapsed >= RECOVERY_WINDOW * 2:
                    org.status = "healthy"
                    org.health = 30.0
                    org.consecutive_failures = 0
                    self._log(f"[IMMUNE] \u23f1 {org.name} auto-released after 2x timeout (no probe received)")

            # Count by status
            if org.status == "quarantined":
                report["quarantined"] += 1
            elif org.status == "degraded":
                report["degraded"] += 1
            else:
                report["healthy"] += 1

        self._recalculate_inflammation()
        return report

    # ------------------------------------------------------------------
    # CENSUS — for API / UI
    # ------------------------------------------------------------------
    def census(self) -> list[dict]:
        return [
            {
                "name":                 o.name,
                "category":             o.category,
                "health":               round(o.health, 1),
                "status":               o.status,
                "consecutive_failures": o.consecutive_failures,
                "total_failures":       o.total_failures,
                "total_successes":      o.total_successes,
            }
            for o in self._organs.values()
        ]

    def inflammation(self) -> float:
        return round(self._inflammation, 3)

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------
    def _update_status(self, organ: OrganHealth, now: float):
        if organ.consecutive_failures >= QUARANTINE_THRESHOLD:
            if organ.status != "quarantined":
                organ.status = "quarantined"
                organ.quarantined_at = now
                self._log(
                    f"[IMMUNE] 🚨 {organ.name} QUARANTINED "
                    f"({organ.consecutive_failures} consecutive failures)"
                )
        elif organ.health < 60:
            organ.status = "degraded"
        else:
            organ.status = "healthy"

    def _recalculate_inflammation(self):
        if not self._organs:
            self._inflammation = 0.0
            return
        total = len(self._organs)
        sick = sum(
            1 for o in self._organs.values()
            if o.status in ("quarantined", "degraded")
        )
        self._inflammation = sick / total

    def _log(self, msg: str):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


# Module-level singleton
immune = SecurityPerimeterSystem()
