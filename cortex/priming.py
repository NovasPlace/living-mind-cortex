"""
Priming Engine — Living Mind
Category: Learning

Implements Spreading Activation (3-hop cascade).
When a memory is fetched, the Engine automatically grants a temporary "primed"
status to linked memories. This keeps the network of thought cohesive.
It updates PostgreSQL to refresh `last_accessed` and bump `access_count`,
which mathematically shields those memories from Ebbinghaus decay.
"""

import time
import asyncio

class PrimingEngine:
    def __init__(self):
        self.total_primes = 0

    async def cascade(self, memory, cortex, depth: int = 3):
        """
        Recursively prime linked memories up to the given depth.
        Each depth level uses ONE connection for a batch UPDATE — not N.
        """
        if depth <= 0 or not memory.linked_ids:
            return

        now = time.time()
        ids = list(memory.linked_ids)

        async with cortex._pool.acquire() as conn:
            await conn.execute("""
                UPDATE memories
                SET last_accessed = $2,
                    access_count = access_count + 1
                WHERE id = ANY($1::uuid[])
            """, ids, now)
            self.total_primes += len(ids)

            # Fetch next-hop linked_ids in the same connection
            if depth > 1:
                rows = await conn.fetch(
                    "SELECT id, linked_ids FROM memories WHERE id = ANY($1::uuid[])", ids
                )

        if depth > 1:
            from types import SimpleNamespace
            for row in rows:
                if row["linked_ids"]:
                    m = SimpleNamespace(linked_ids=list(row["linked_ids"]))
                    await self.cascade(m, cortex, depth - 1)

# Module-level singleton
priming = PrimingEngine()

