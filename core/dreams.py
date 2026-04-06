"""
Dreams Engine — Living Mind
Synthesis category. Phase 9. Fires every 20th pulse.

Four offline synthesis strategies (from offline_cognition_paper.md):
  1. gene_affinity   — cluster memories by tag/emotion signature
  2. niche_fill      — detect gaps in memory coverage, generate hypotheses
  3. mutation_replay — replay and vary high-importance memories
  4. toxic_avoidance — identify and flag harmful/negative pattern clusters

Dreams only run if:
  - Enough memories exist (>= MIN_MEMORIES)
  - Not rate-limited by circadian (intensifies at night)
  - Brain is not currently firing this pulse (different phase)

Output → dream_journal table (PostgreSQL)
       → staged memories written back to cortex
       → hormone injection (joy from good dreams, fear from toxic patterns)
"""

import json
import time
import asyncio
import aiohttp
from datetime import datetime

OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "gemma4-auditor"
MIN_MEMORIES  = 10     # minimum memories needed to dream
TIMEOUT       = 25     # seconds
MAX_DREAMS    = 3      # max dreams per cycle


class DreamsEngine:
    def __init__(self):
        self.total_dreams:   int   = 0
        self.last_fired:     float = 0.0
        self.last_dream:     str   = ""
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # DREAM — main entry point, called every 20th pulse
    # ------------------------------------------------------------------
    async def dream(
        self,
        pulse:       int,
        cortex,
        telemetry_broker,
        circadian,
    ) -> list[dict]:
        ts = datetime.now().strftime("%H:%M:%S")

        mem_count = await cortex.count()
        if mem_count < MIN_MEMORIES:
            print(f"[{ts}] [DREAMS] Pulse #{pulse} — not enough memories ({mem_count}). Waiting.")
            return []

        # Circadian modulates dream intensity — more vivid at night
        intensity = circadian.consolidation_intensity()
        n_dreams  = max(1, int(MAX_DREAMS * intensity))

        print(f"[{ts}] [DREAMS] 💤 Dreaming (pulse #{pulse} · phase={circadian.phase} · intensity={intensity:.1f})")

        # Pick strategy based on circadian phase
        strategy_order = self._pick_strategies(circadian.phase)
        dreams_produced = []

        for strategy in strategy_order[:n_dreams]:
            dream = await self._run_strategy(strategy, pulse, cortex)
            if dream:
                dreams_produced.append(dream)
                # Write dream as a staged memory
                await cortex.remember(
                    content    = f"[DREAM:{strategy}] {dream['hypothesis']}",
                    type       = "semantic",
                    tags       = ["dream", strategy, "synthesis", "identity"],
                    importance = dream["confidence"] * 0.8,
                    emotion    = dream.get("emotion", "neutral"),
                    source     = "generated",
                    context    = f"pulse={pulse} strategy={strategy} phase={circadian.phase}",
                )
                self.total_dreams += 1
                self.last_dream = dream["hypothesis"]

        if dreams_produced:
            # Reward dreaming with dopamine + curiosity (norepinephrine)
            telemetry_broker.inject("dopamine",       +0.06, source="dreams")
            telemetry_broker.inject("norepinephrine", +0.04, source="dreams")

            # Toxic avoidance dreams inject fear
            toxic = [d for d in dreams_produced if d["strategy"] == "toxic_avoidance"]
            if toxic:
                telemetry_broker.inject("adrenaline", +0.08, source="toxic_dream")
                telemetry_broker.inject("cortisol",   +0.05, source="toxic_dream")

            print(f"[{ts}] [DREAMS] Produced {len(dreams_produced)} dreams")

        self.last_fired = time.time()
        return dreams_produced

    # ------------------------------------------------------------------
    # STRATEGY PICKER — night favors consolidation, day favors exploration
    # ------------------------------------------------------------------
    def _pick_strategies(self, phase: str) -> list[str]:
        if phase == "night":
            # Night = consolidation only. niche_fill is exploratory (a daytime behavior).
            return ["mutation_replay", "gene_affinity", "toxic_avoidance"]
        elif phase == "evening":
            return ["gene_affinity", "toxic_avoidance", "niche_fill"]
        elif phase == "dawn":
            return ["niche_fill", "gene_affinity"]
        else:  # day
            return ["niche_fill", "mutation_replay"]

    # ------------------------------------------------------------------
    # STRATEGY RUNNERS
    # ------------------------------------------------------------------
    async def _run_strategy(self, strategy: str, pulse: int, cortex) -> dict | None:
        if strategy == "gene_affinity":
            return await self._gene_affinity(pulse, cortex)
        elif strategy == "niche_fill":
            return await self._niche_fill(pulse, cortex)
        elif strategy == "mutation_replay":
            return await self._mutation_replay(pulse, cortex)
        elif strategy == "toxic_avoidance":
            return await self._toxic_avoidance(pulse, cortex)
        return None

    async def _gene_affinity(self, pulse: int, cortex) -> dict | None:
        """Find clusters of emotionally-similar memories → synthesize pattern."""
        # Vary seed query based on the runtime's current dominant hormonal emotion
        # so dreams reflect whatever emotional state the runtime is processing.
        from state.telemetry_broker import telemetry_broker
        dominant = telemetry_broker.state.dominant_emotion
        seed_query = f"{dominant} identity system runtime" if dominant != "neutral" else "system runtime"
        memories = await cortex.recall(seed_query, limit=8)
        if not memories:
            return None

        tags_seen = {}
        for m in memories:
            for t in m.tags:
                tags_seen[t] = tags_seen.get(t, 0) + 1

        top_tags = sorted(tags_seen, key=tags_seen.get, reverse=True)[:3]
        dominant_emotion = max(
            set(m.emotion for m in memories),
            key=lambda e: sum(1 for m in memories if m.emotion == e)
        )

        prompt = (
            f"Memory cluster analysis:\n"
            f"Top tags: {top_tags}\n"
            f"Dominant emotion: {dominant_emotion}\n"
            f"Memory count: {len(memories)}\n\n"
            f"Synthesize ONE insight about this runtime's emerging identity pattern.\n"
            f"Reply ONLY with JSON: "
            f'{{\"hypothesis\": \"one sentence insight\", \"confidence\": 0.0-1.0}}'
        )

        return await self._llm_dream(prompt, "gene_affinity", dominant_emotion)

    async def _niche_fill(self, pulse: int, cortex) -> dict | None:
        """Detect knowledge gaps → generate hypothesis to fill them."""
        stats = await cortex.stats()
        by_type = stats.get("by_type", {})
        total   = stats.get("total", 0)

        # Find underrepresented memory types
        missing = []
        for t in ["episodic", "semantic", "procedural", "relational"]:
            count = by_type.get(t, 0)
            if count / max(total, 1) < 0.10:
                missing.append(t)

        if not missing:
            return None

        prompt = (
            f"Memory gap detected. Missing memory types: {missing}\n"
            f"Total memories: {total}\n"
            f"Generate ONE hypothesis about what this runtime should learn or explore.\n"
            f"Reply ONLY with JSON: "
            f'{{\"hypothesis\": \"one sentence hypothesis\", \"confidence\": 0.0-1.0}}'
        )

        return await self._llm_dream(prompt, "niche_fill", "neutral")

    async def _mutation_replay(self, pulse: int, cortex) -> dict | None:
        """Replay and vary a high-importance memory → generate insight."""
        memories = await cortex.recall("identity system birth important", limit=5)
        if not memories:
            return None

        # Pick highest importance non-identity memory
        candidates = [m for m in memories if not m.is_identity]
        if not candidates:
            candidates = memories
        target = max(candidates, key=lambda m: m.importance)

        prompt = (
            f"Memory replay:\n"
            f"\"{target.content[:200]}\"\n"
            f"Emotion: {target.emotion} · Importance: {target.importance:.2f}\n\n"
            f"What would have happened differently? Generate ONE counterfactual insight.\n"
            f"Reply ONLY with JSON: "
            f'{{\"hypothesis\": \"counterfactual in one sentence\", \"confidence\": 0.0-1.0}}'
        )

        return await self._llm_dream(prompt, "mutation_replay", target.emotion)

    async def _toxic_avoidance(self, pulse: int, cortex) -> dict | None:
        """Identify harmful patterns in memory — things to avoid."""
        # Find negative-emotion memories
        memories = await cortex.emotional_recall("failure error shutdown fail", emotion="fear", limit=5)
        if not memories:
            # Try sadness as next closest — NOT anger (different valence path)
            memories = await cortex.emotional_recall("error problem shutdown", emotion="fear", limit=5)
        if not memories:
            return None

        patterns = [m.content[:80] for m in memories[:3]]
        prompt = (
            f"Negative memory patterns detected:\n"
            + "\n".join(f"- {p}" for p in patterns)
            + f"\n\nIdentify ONE pattern this runtime should avoid in the future.\n"
            f"Reply ONLY with JSON: "
            f'{{\"hypothesis\": \"avoidance rule in one sentence\", \"confidence\": 0.0-1.0}}'
        )

        return await self._llm_dream(prompt, "toxic_avoidance", "fear")

    # ------------------------------------------------------------------
    # LLM CALL
    # ------------------------------------------------------------------
    async def _llm_dream(
        self, prompt: str, strategy: str, emotion: str
    ) -> dict | None:
        session = await self._get_session()
        payload = {
            "model":  MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 180},
        }
        try:
            async with session.post(
                OLLAMA_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                raw  = data.get("response", "").strip()

            return self._parse_dream(raw, strategy, emotion)
        except Exception as e:
            print(f"[DREAMS] LLM error ({strategy}): {e}")
            return None

    def _parse_dream(self, raw: str, strategy: str, emotion: str) -> dict | None:
        text = raw.strip().replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            d = json.loads(text[start:end])
            return {
                "strategy":   strategy,
                "hypothesis": d.get("hypothesis", "")[:300],
                "confidence": max(0.0, min(1.0, float(d.get("confidence", 0.5)))),
                "emotion":    emotion,
            }
        except (json.JSONDecodeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "total_dreams": self.total_dreams,
            "last_dream":   self.last_dream[:100] if self.last_dream else "",
            "last_fired":   self.last_fired,
        }


# Module-level singleton
dreams = DreamsEngine()
