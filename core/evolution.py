import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from cortex.engine import cortex

BASE_DIR = Path(__file__).parent.parent
SKILLS_DIR = BASE_DIR / "skills"

class Evolution:
    """
    The Evolutionary Organ: Distills experience into permanent procedural DNA (Skills).
    Inspired by Hermes/agentskills.io.
    """

    @staticmethod
    def compress_trajectory(history: List[Dict[str, Any]], target_tokens: int = 4000) -> List[Dict[str, Any]]:
        """
        Lighter "Protect-Summarize" compressor.
        - Keeps first 3 turns (System, Human Goal, First Action).
        - Keeps last 3 turns (Conclusion, Result, Final Tool).
        - Summarizes the "Middle" turns if history > target_tokens.
        """
        # (Simplified token estimation for this version)
        total_chars = sum(len(str(t)) for t in history)
        if total_chars < target_tokens * 4:
            return history

        head = history[:3]
        tail = history[-3:]
        middle = history[3:-3]
        
        # In a real impl, we'd call an LLM to summarize 'middle'.
        # For now, we "compress" by collapsing redundant tool outputs.
        summary_msg = {
            "role": "system",
            "content": f"[EVOLUTION: Compressed {len(middle)} intermediate tool turns into a procedural anchor.]"
        }
        
        return head + [summary_msg] + tail

    async def distill_skill(self, session_id: str, goal: str, outcome: str = "success"):
        """
        Distills a mission into a SKILL.md file.
        """
        if outcome != "success":
            return

        # Fetch memories for this session via unique tag
        mems = await cortex.recall("", tag=f"session:{session_id}", limit=20)
        if not mems:
            return

        # Prepare name & path
        slug = re.sub(r'[^a-z0-9]+', '_', goal.lower()).strip('_')[:32]
        category = "navigation" if "github" in goal.lower() or "wikipedia" in goal.lower() else "general"
        skill_dir = SKILLS_DIR / category / slug
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Distillation Prompt (The "Thought" that builds the skill)
        # We simulate the Mind "reflecting" on the successful steps.
        steps = []
        for m in mems:
            if "Tool Result" in m.content:
                steps.append(f"- Step: {m.content[:100]}...")

        skill_md = f"""---
name: {slug.replace('_', ' ').title()}
description: Autonomous distillation of "{goal}" mission.
category: {category}
version: 1.0.0
---

# {slug.replace('_', ' ').title()}

## Heuristic Insights
This procedure was distilled after a successful outcome in session {session_id}.

## Procedural Steps
{chr(10).join(steps)}

## Lessons Learned
- Avoid direct navigation if security gates appear; use Search Proxy.
- Prefer specific headers (h2/h3) for repository/article name extraction.
- Observe [LINK: ...] markers for high-intent interactive elements.
"""
        
        with open(skill_dir / "SKILL.md", "w") as f:
            f.write(skill_md)
        
        print(f"🧬 [EVOLUTION] Distilled new skill: {category}/{slug}")
        await cortex.remember(
            content=f"Evolved new skill: {slug}. Proceeding to expert mastery.",
            type="semantic",
            tags=["evolution", "skill_birth", slug],
            importance=1.0,
            source="generated"
        )

evolution = Evolution()
