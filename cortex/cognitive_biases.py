"""
Cognitive Biases Engine — Living Mind
Category: Learning

Mathematically skews the retrieval of memories. 
When the Brain or Dreams engine recalls memories, this layer intercepts them
and applies heuristics.
- Availability Heuristic: Memories matching the current hormonal emotion get boosted.
- Confirmation Bias: Memories aligning with the Awakening Directive get boosted.
"""

from typing import List
import time
from cortex.engine import Memory

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "i", "my", "it", "of", "to", "and", "in"}

class CognitiveBiases:
    def __init__(self):
        self.stats_applied = 0
        
    def apply_biases(self, memories: List[Memory], hormone_state, directive: str) -> List[Memory]:
        """
        Takes raw memories from Postgres, applies bias scaling, and re-sorts them.
        This modifies the representation BEFORE it hits Working Memory or the Brain.
        """
        if not memories:
            return []
            
        dominant_emotion = hormone_state.dominant_emotion if hormone_state else "neutral"
        
        # We don't want to permanently alter importance in the DB based on transient mood.
        # So we clone and modify the instances for this specific retrieval context.
        biased_memories = []
        for raw in memories:
            # We copy the bare minimum needed for working memory
            m = Memory(
                id=raw.id,
                content=raw.content,
                type=raw.type,
                tags=list(raw.tags),
                importance=raw.importance,
                created_at=raw.created_at,
                last_accessed=raw.last_accessed,
                access_count=raw.access_count,
                emotion=raw.emotion,
                confidence=raw.confidence,
                context=raw.context,
                source=raw.source,
                linked_ids=list(raw.linked_ids),
                metadata=dict(raw.metadata),
                is_flashbulb=raw.is_flashbulb,
                is_identity=raw.is_identity
            )
            
            # 1. Availability Heuristic
            if dominant_emotion != "neutral" and m.emotion == dominant_emotion:
                m.importance *= 1.30  # 30% boost for matching the mood
                
            # 2. Confirmation Bias
            if directive and len(directive) > 5:
                directive_words = set(directive.lower().split()) - STOPWORDS
                mem_words = set(m.content.lower().split()) - STOPWORDS
                overlap = len(directive_words.intersection(mem_words))
                if overlap >= 3:  # raised from >2 to avoid common runtime vocabulary
                    m.importance *= 1.25  # 25% boost for agreeing with overarching goals

            # 3. Recency Bias (Cortex Paper Phase 4)
            # Boost anything created in the last 15 minutes
            age = time.time() - m.created_at
            if age < 900:  # 15 minutes
                # Linear boost: younger = higher boost, up to 40%
                recency_boost = 1.0 + (0.4 * (1.0 - (age / 900.0)))
                m.importance *= recency_boost

            # Cap at 1.0 legally
            m.importance = min(1.0, m.importance)
            biased_memories.append(m)
            
        self.stats_applied += 1
        
        # Return re-sorted by the new, biased importance
        return sorted(biased_memories, key=lambda x: x.importance, reverse=True)

# Module-level singleton
biases = CognitiveBiases()
