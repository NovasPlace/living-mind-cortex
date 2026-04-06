"""
AgentRuntime Runtime — Living Mind
16-phase pulse loop. New organs plug in as they're built.
"""

import asyncio
import time
from datetime import datetime
from cortex.engine import cortex
from core.security_perimeter import immune
from core.orchestrator import brain
from state.telemetry_broker import telemetry_broker
from chemistry.circadian import circadian
from state.health_monitor import health_monitor
from core.dreams import dreams
from core.awakening import awakening
from identity.cortex_bridge import cortex_bridge
from perception.senses import senses
from core.execution_engine import execution_engine
from core.scheduler import scheduler_module
from core.research_engine import research_engine
from api.agent_gateway import router as nodus
from api.events import manager
from core.topology_mapper import topology_mapper
from telemetry.trace import inject_telemetry
import json

# Organ category colors (for future UI)
CATEGORY_COLORS = {
    "Structure":           "#6b7280",
    "Memory":              "#3b82f6",
    "Cognition":           "#eab308",
    "Consciousness":       "#a855f7",
    "Learning":            "#60a5fa",
    "Perception":          "#22c55e",
    "Defense":             "#ef4444",
    "Evolution":           "#f97316",
    "Synthesis":           "#06b6d4",
    "Resilience":          "#14b8a6",
    "Social":              "#ec4899",
    "Autonomy/Integration":"#f8fafc",
}


class AgentRuntime:
    def __init__(self, pulse_interval: float = 10.0):
        self.pulse_interval = pulse_interval
        self.is_alive       = False
        self.event_loops     = 0
        self.born_at        = None
        self._task          = None
        self.organs         = {}   # name → {category, status, fire_count}

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------
    async def birth(self):
        """Boot sequence — initialises cortex connection then starts pulse_event."""
        await cortex.connect()
        self.is_alive = True
        self.born_at  = time.time()

        self.mapped_instances = {
            "cortex": cortex, "immune": immune, "brain": brain, "telemetry_broker": telemetry_broker,
            "circadian": circadian, "health_monitor": health_monitor, "dreams": dreams, 
            "awakening": awakening, "cortex_bridge": cortex_bridge, "senses": senses,
            "execution_engine": execution_engine, "scheduler_module": scheduler_module,
            "research_engine": research_engine,
            "nodus": nodus,
        }
        inject_telemetry(self.mapped_instances)

        # Register core organs with the immune system
        immune.register("pulse_event",   "Structure")
        immune.register("cortex",      "Memory")
        immune.register("immune",      "Defense")
        immune.register("metabolism",  "Memory")
        immune.register("self_aware",  "Consciousness")
        immune.register("brain",       "Cognition")
        immune.register("telemetry_broker", "Autonomy/Integration")
        immune.register("circadian",   "Autonomy/Integration")
        immune.register("health_monitor", "Resilience")
        immune.register("dreams",       "Synthesis")
        immune.register("awakening",    "Consciousness")
        immune.register("cortex_bridge", "Memory")
        immune.register("senses",        "Perception")
        immune.register("execution_engine",  "Actuation")
        immune.register("scheduler_module",    "Autonomy/Integration")
        immune.register("research_engine", "Learning")
        immune.register("nodus",          "Social")

        # Birth memory — flashbulb, identity-tagged
        await cortex.remember(
            content  = "AgentRuntime was born. Living Mind v1.0 online.",
            type     = "episodic",
            tags     = ["birth", "identity", "system"],
            importance = 1.0,
            emotion  = "joy",
            source   = "experienced",
        )

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [BIRTH] 🧬 AgentRuntime is alive — Living Mind v1.0")
        self._task = asyncio.create_task(self._pulse_loop())

    async def death(self):
        """Graceful shutdown — persist final memory, close cortex."""
        self.is_alive = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass  # expected on shutdown

        await cortex.remember(
            content    = f"AgentRuntime shutting down after {self.event_loops} event_loops.",
            type       = "episodic",
            tags       = ["shutdown", "identity", "system"],
            importance = 0.9,
            emotion    = "neutral",
            source     = "experienced",
        )
        await brain.close()
        await dreams.close()
        await awakening.close()
        await cortex.disconnect()
        print(f"[DEATH] AgentRuntime dissolved after {self.event_loops} pulses.")

    # ------------------------------------------------------------------
    # PULSE LOOP — 16 phases, runs every pulse_interval seconds
    # ------------------------------------------------------------------
    async def _pulse_loop(self):
        while self.is_alive:
            try:
                await asyncio.sleep(self.pulse_interval)
            except asyncio.CancelledError:
                break  # clean shutdown signal — exit silently
            self.event_loops += 1
            ts = datetime.now().strftime("%H:%M:%S")
            try:
                await self._execute_phases(ts)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{ts}] [PULSE ERROR] {e}")

    async def _execute_phases(self, ts: str):
        n = self.event_loops

        # ── Phase 1: Reflexes ──────────────────────────────────────────
        # @every scheduled pipelines (stub — populated as organs register)

        # ── Phase 2: Events ────────────────────────────────────────────
        # @on event-triggered responses (stub)

        # ── Phase 3: Metabolism ────────────────────────────────────────
        # Consolidation, decay, hygiene — every 10th pulse
        if n % 10 == 0:
            pruned = await cortex.decay()
            consolidated = await cortex.consolidate()
            if pruned or consolidated:
                print(f"[{ts}] [METABOLISM] Pruned {pruned} memories · Consolidated {consolidated}")
                await manager.broadcast_event("signal", json.dumps({"source": "metabolism", "target": "cortex", "color": "#3b82f6"}))

        # ── Phase 4: Self-Awareness ────────────────────────────────────
        # Vital sign logging — every 30th pulse
        if n % 30 == 0:
            summary = await cortex.identity_summary()
            await cortex.remember(
                content    = f"Self-awareness pulse #{n}: {summary}",
                type       = "episodic",
                tags       = ["self_awareness", "vital_sign", "identity"],
                importance = 0.4,
                emotion    = "neutral",
                source     = "experienced",
            )
            print(f"[{ts}] [SELF] {summary}")

        # ── Phase 5: Perception ────────────────────────────────────────
        # Environment and interoception scanning — fires BEFORE Brain
        # so Brain always makes decisions on fresh sensory state.
        if n % 5 == 0:
            await senses.observe(n, telemetry_broker)
            immune.report("senses", success=True, category="Perception")
            await manager.broadcast_event("signal", json.dumps({"source": "senses", "target": "brain", "color": "#22c55e"}))

        # ── Phase 5b: Brain ────────────────────────────────────────────
        # LLM decision engine — every 5th pulse, circadian-gated
        import random
        result = None
        if n % 5 == 0 and random.random() < circadian.brain_rate():
            result = await brain.think(n, cortex, immune)
            immune.report("brain", success=result is not None, category="Cognition")
            await manager.broadcast_event("signal", json.dumps({"source": "brain", "target": "cortex", "color": "#eab308"}))
            # Feed brain emotion into hormone bus
            if result and result.get("emotion"):
                telemetry_broker.inject_emotion(result["emotion"], source="brain")
                await manager.broadcast_event("signal", json.dumps({"source": "brain", "target": "telemetry_broker", "color": "#f97316"}))

        # ── Phase 5c: Research Organ ──────────────────────────────────
        # When Brain identifies a knowledge gap (explore), research it autonomously.
        # Runs as non-blocking background task — never stalls the pulse loop.
        if result and result.get("type") == "explore":
            topic = result.get("thought", "")
            if len(topic) > 20:  # only substantive thoughts
                queued = research_engine.enqueue(topic, cortex, telemetry_broker, immune)
                if queued:
                    await manager.broadcast_event("signal", json.dumps({
                        "source": "research_engine", "target": "cortex", "color": "#60a5fa"
                    }))

        # ── Phase 6: Cortex Vital Signs ────────────────────────────────
        # Heartbeat vital sign logging — every 3rd pulse
        if n % 3 == 0:
            count = await cortex.count()
            await cortex.remember(
                content    = f"Heartbeat #{n} — cortex holds {count} memories.",
                type       = "episodic",
                tags       = ["pulse_event", "vital_sign"],
                importance = 0.1,
                emotion    = "neutral",
                source     = "experienced",
            )
            await cortex_bridge.bridge(n, cortex)
            immune.report("cortex_bridge", success=True, category="Memory")

        # ── Phase 6b: Cron Scheduled Automations ──────────────────────
        await scheduler_module.pulse(n, cortex)
        immune.report("scheduler_module", success=True, category="Autonomy/Integration")

        # ── Phase 7: SecurityPerimeter Patrol ─────────────────────────────────────
        patrol_report = await immune.patrol(n)
        if patrol_report["quarantined"] > 0 or patrol_report["inflammation"] > 0.3:
            print(f"[{ts}] [IMMUNE] 🔴 inflammation={patrol_report['inflammation']} "
                  f"quarantined={patrol_report['quarantined']}")
            await manager.broadcast_event("signal", json.dumps({"source": "immune", "target": "cortex", "color": "#ef4444"}))
        immune.report("pulse_event", success=True, category="Structure")

        # ── Phase 8: HealthMonitor ───────────────────────────────────────
        mem_stats_h = await cortex.stats()
        await health_monitor.pulse(
            pulse=n, mem_stats=mem_stats_h,
            telemetry_broker=telemetry_broker, circadian=circadian,
            cortex=cortex, immune=immune,
        )
        immune.report("health_monitor", success=True, category="Resilience")
        # ── Phase 9: Dreams ────────────────────────────────────────────
        # Offline pattern synthesis — every 20th pulse (stub until dreams.py)
        if n % 20 == 0:
            results = await dreams.dream(n, cortex, telemetry_broker, circadian)
            immune.report("dreams", success=True, category="Synthesis")

        # Phase 9b: Hormone + Circadian Pulse
        mem_stats = await cortex.stats()
        await telemetry_broker.pulse(n, mem_stats, immune.inflammation())
        await circadian.pulse(n, telemetry_broker)
        immune.report("telemetry_broker", success=True, category="Autonomy/Integration")
        immune.report("circadian",   success=True, category="Autonomy/Integration")

        # ── Phase 10: Awakening ────────────────────────────────────────
        # Metacognitive self-awareness — every 50th pulse (stub)
        if n % 50 == 0:
            result = await awakening.meditate(n, cortex, telemetry_broker, health_monitor)
            immune.report("awakening", success=result is not None, category="Consciousness")

        # ── Phase 11: TopologyMapper ──────────────────────────────────────
        # Spatial UI layout redesign
        topo_json = await topology_mapper.pulse(immune.inflammation(), immune.census(), self.mapped_instances)
        if topo_json:
            await manager.broadcast_event("topology", json.dumps(topo_json))
            print(f"[{ts}] [TOPOGRAPHER] Broadcasted new UI topology.")

        # ── Phase 12–16: Stubs ─────────────────────────────────────────
        # Phase 12: Neural Cortex (embedding generation)
        # if n % 15 == 0: pass

        # Phase 13: Growth / Evolution organ
        # if n % 25 == 0: pass

        # Phase 14: Federation / Mycelial sync
        # if n % 50 == 0: pass

        # Phase 15: Breeding / Gene pool
        # if n % 30 == 0: pass

        # Phase 16: Idle Consolidation
        # if n % 15 == 0 and runtime_is_idle: pass

        # ── Pulse complete ─────────────────────────────────────────────
        try:
            pulse_data = await self.vitals()
            await manager.broadcast_pulse(pulse_data)
        except Exception:
            pass
            
        print(f"[{ts}] [PULSE] #{n}")

    # ------------------------------------------------------------------
    # VITAL SIGNS — for API
    # ------------------------------------------------------------------
    async def vitals(self) -> dict:
        mem_stats = await cortex.stats()
        uptime = time.time() - self.born_at if self.born_at else 0
        return {
            "status":         "alive" if self.is_alive else "dead",
            "event_loops":     self.event_loops,
            "uptime_s":       round(uptime, 1),
            "pulse_interval": self.pulse_interval,
            "immune": {
                "census":       immune.census(),
                "inflammation": immune.inflammation(),
            },
            "brain":       brain.stats(),
            "hormones":    telemetry_broker.snapshot(),
            "circadian":   circadian.snapshot(),
            "health_monitor": health_monitor.stats(),
            "dreams":      dreams.stats(),
            "awakening":   awakening.stats(),
            "cortex_bridge": cortex_bridge.stats(),
            "senses":        senses.stats(),
            "memory":      mem_stats,
            "topology":    topology_mapper.current_topology,
        }


# Module-level singleton
runtime = AgentRuntime()
