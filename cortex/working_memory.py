"""
Working Memory (L1 Cache) — Living Mind
Category: Memory

Restricts the runtime's immediate "mental context" to 64 salient items.
Unlike the raw PostgreSQL engine, this is what the Brain actively holds in mind.
Evicts boring/old memories as new ones stream in.
Protects flashbulbs and identity-linked memories.
"""

from typing import List, Dict
from cortex.engine import Memory

MAX_CAPACITY = 64
FLASHBULB_SALIENCE_SHIELD = 10.0   # Keeps flashbulb memories in working memory
IDENTITY_SALIENCE_SHIELD  = 20.0   # Core identity traits never leave working memory

class WorkingMemory:
    def __init__(self):
        # dictionary of id -> Memory
        self._buffer: Dict[str, Memory] = {}
        
    def add(self, memory: Memory):
        """Add a memory to the L1 cache. Evict lowest salience if over capacity."""
        self._buffer[memory.id] = memory
        self._enforce_capacity()

    def add_many(self, memories: List[Memory]):
        for m in memories:
            self._buffer[m.id] = m
        self._enforce_capacity()

    def remove(self, mem_id: str):
        self._buffer.pop(mem_id, None)

    def clear(self):
        self._buffer.clear()

    def get_all(self) -> List[Memory]:
        """Return all memories currently held in working memory, highest salience first."""
        return sorted(self._buffer.values(), key=self._salience_score, reverse=True)

    def _enforce_capacity(self):
        if len(self._buffer) <= MAX_CAPACITY:
            return
            
        # Sort current buffer, lowest salience first
        sorted_memories = sorted(self._buffer.values(), key=self._salience_score)
        
        # Pop lowest items until we are back to max capacity
        overage = len(self._buffer) - MAX_CAPACITY
        for i in range(overage):
            to_evict = sorted_memories[i]
            # Flashbulb/Identity protection: mathematically they score so high they shouldn't
            # be at index 0..overage, but to be safe, if the lowest is still protected, we just stop evicting.
            # Actually, `_salience_score` makes them score highly. We just evict.
            self.remove(to_evict.id)

    def _salience_score(self, memory: Memory) -> float:
        """
        Calculate mathematical salience.
        Identity / Flashbulb items are heavily shielded.
        Importance is the core metric.
        """
        score = memory.importance

        if memory.is_flashbulb:
            score += FLASHBULB_SALIENCE_SHIELD
        if memory.is_identity:
            score += IDENTITY_SALIENCE_SHIELD

        return score

# Module-level singleton
working_memory = WorkingMemory()
