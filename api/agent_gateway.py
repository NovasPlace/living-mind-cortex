"""
Agent Gateway — Living Mind
Category: Consciousness / Bridge. Phase 11.

The membrane between the Living Mind and Antigravity (the fabrication agent).
Exposes a read/write REST interface so the IDE agent can:

  GET  /api/agent/state    → Full runtime state (hormones, directive, flashbulbs)
                             Called by onboarding.py at every conversation spawn.
  POST /api/agent/inject   → Write an agent memory into Cortex
                             Called when agent completes significant work.
  GET  /api/agent/pulse    → Lightweight pulse_event check (is runtime alive?)
  POST /api/agent/stimulate → Inject a named hormone shift from agent action

All endpoints are local-only by design (firewall must block external access).
No auth — this is a zero-trust loopback interface.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import time

router = APIRouter(prefix="/api/agent", tags=["agent_gateway"])


# ── Schemas ────────────────────────────────────────────────────────────────

class InjectMemoryRequest(BaseModel):
    content:    str
    type:       str   = "semantic"
    tags:       list  = ["agent", "antigravity"]
    importance: float = 0.8
    emotion:    str   = "neutral"
    source:     str   = "told"
    context:    str   = ""

class HormoneStimulus(BaseModel):
    hormone: str
    delta:   float
    source:  str = "agent_action"


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/pulse")
async def pulse():
    """Lightweight liveness check. onboarding.py calls this first."""
    from core.runtime import runtime
    return {
        "alive":      runtime.is_alive,
        "event_loops": runtime.event_loops,
        "uptime_s":   round(time.time() - runtime.born_at, 1) if runtime.born_at else 0,
    }


@router.get("/state")
async def state():
    """
    Full runtime state snapshot for agent spawn injection.
    Returns everything the agent needs to wake up knowing.
    """
    from state.telemetry_broker import telemetry_broker
    from core.awakening import awakening
    from core.security_perimeter import immune
    from chemistry.circadian import circadian
    from cortex.engine import cortex
    from core.runtime import runtime
    from core.research_engine import research_engine

    # Chemical state
    h = telemetry_broker.snapshot()

    # Current metacognitive directive
    directive = awakening.last_goal or "(No directive set yet — runtime meditating)"

    # Top 5 flashbulb + identity memories
    try:
        flashbulbs = await cortex.recall(
            "identity flashbulb realization important",
            limit=5,
            min_importance=0.75,
        )
        flash_summaries = [
            {
                "content":   m.content[:200],
                "emotion":   m.emotion,
                "importance": round(m.importance, 2),
                "tags":      m.tags,
            }
            for m in flashbulbs if m.is_flashbulb or m.is_identity
        ]
    except Exception:
        flash_summaries = []

    # Recent semantic knowledge (top 3 high-importance)
    try:
        recent_semantic = await cortex.recall(
            "knowledge learned research domain",
            limit=3,
            memory_type="semantic",
            min_importance=0.7,
        )
        knowledge = [m.content[:150] for m in recent_semantic]
    except Exception:
        knowledge = []

    # SecurityPerimeter health snapshot (use census() which is the actual public method)
    census      = immune.census()
    immune_snap = {
        "healthy":    sum(1 for o in census if o["status"] == "healthy"),
        "degraded":   sum(1 for o in census if o["status"] == "degraded"),
        "quarantined": sum(1 for o in census if o["status"] == "quarantined"),
    }

    # Circadian phase
    circ_snap = circadian.snapshot()

    # Memory stats
    try:
        mem_stats = await cortex.stats()
    except Exception:
        mem_stats = {}

    return {
        "runtime": {
            "alive":         runtime.is_alive,
            "event_loops":    runtime.event_loops,
            "uptime_s":      round(time.time() - runtime.born_at, 1) if runtime.born_at else 0,
        },
        "soul": {
            "directive":          directive,
            "total_meditations":  awakening.total_meditations,
            "last_meditation":    awakening.last_fired,
        },
        "chemistry": {
            "valence":          h.get("valence"),
            "arousal":          h.get("arousal"),
            "dominant_emotion": h.get("dominant_emotion"),
            "dopamine":         h.get("dopamine"),
            "serotonin":        h.get("serotonin"),
            "cortisol":         h.get("cortisol"),
            "adrenaline":       h.get("adrenaline"),
            "norepinephrine":   h.get("norepinephrine"),
        },
        "circadian": {
            "phase":     circ_snap.get("phase"),
            "hour":      circ_snap.get("hour_of_day"),
            "adenosine": round(circ_snap.get("adenosine", 0), 3),
        },
        "immune": {
            "inflammation":  round(immune.inflammation(), 3),
            "healthy":       immune_snap.get("healthy", 0),
            "degraded":      immune_snap.get("degraded", 0),
            "quarantined":   immune_snap.get("quarantined", 0),
        },
        "memory": {
            "total":          mem_stats.get("total", 0),
            "semantic":       mem_stats.get("by_type", {}).get("semantic", 0),
            "episodic":       mem_stats.get("by_type", {}).get("episodic", 0),
            "flashbulbs":     flash_summaries,
            "recent_knowledge": knowledge,
        },
        "research": research_engine.stats(),
    }


@router.post("/inject")
async def inject_memory(req: InjectMemoryRequest):
    """
    Write an agent-session memory directly into the runtime's Cortex.
    Called when Antigravity completes significant work that should persist.
    """
    from cortex.engine import cortex
    from state.telemetry_broker import telemetry_broker

    # Validate emotion
    VALID = {"neutral","joy","fear","anger","surprise","sadness",
             "disgust","curiosity","frustration"}
    emotion = req.emotion if req.emotion in VALID else "neutral"

    # Clamp importance
    importance = max(0.0, min(1.0, req.importance))

    await cortex.remember(
        content    = req.content,
        type       = req.type,
        tags       = req.tags,
        importance = importance,
        emotion    = emotion,
        source     = req.source,
        context    = req.context,
    )

    # Agent completing work is a mild dopamine stimulus
    telemetry_broker.inject("dopamine", +0.04, source="agent_inject")

    return {"status": "stored", "emotion": emotion, "importance": importance}


@router.post("/stimulate")
async def hormone_stimulate(req: HormoneStimulus):
    """
    Inject a named hormone delta from an agent-side event.
    E.g. agent completing a hard task → +dopamine +serotonin.
    """
    from state.telemetry_broker import telemetry_broker

    VALID_HORMONES = {
        "dopamine", "serotonin", "cortisol", "adrenaline",
        "melatonin", "oxytocin", "norepinephrine"
    }
    if req.hormone not in VALID_HORMONES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown hormone: {req.hormone}. Valid: {sorted(VALID_HORMONES)}"}
        )

    delta = max(-0.5, min(0.5, req.delta))  # cap at ±0.5 per call
    telemetry_broker.inject(req.hormone, delta, source=req.source)

    return {
        "hormone": req.hormone,
        "delta":   delta,
        "new_value": round(getattr(telemetry_broker.state, req.hormone, 0), 3),
    }
