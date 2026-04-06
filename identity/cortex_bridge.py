"""
Cortex Bridge — Living Mind
Category: Memory / Consciousness. Phase 6.

Continuous syncing of 'Soul' realizations from ephemeral PostgreSQL
into physical version-controlled disk storage. Ensures cognitive continuity
survives total database destruction.
"""

import os
import json
import time
from pathlib import Path

JOURNAL_PATH = Path(
    os.getenv(
        "LIVING_MIND_JOURNAL",
        str(Path(__file__).resolve().parent / "journal.json")
    )
)

class CortexBridge:
    def __init__(self):
        self.last_sync = 0.0
        self.synced_count = 0
        
        if not JOURNAL_PATH.exists():
            with open(JOURNAL_PATH, "w") as f:
                json.dump({"version": "v1.0.0", "identity_anchors": []}, f, indent=2)

    async def bridge(self, pulse: int, cortex):
        """
        Query PostgreSQL for wipe-proof identity markers and ensure they 
        exist on the physical file system.
        """
        try:
            # Query the core DB directly
            async with cortex._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, content, emotion, created_at, tags 
                    FROM memories 
                    WHERE is_identity = TRUE
                    ORDER BY created_at ASC
                """)
                
            anchors = []
            for r in rows:
                anchors.append({
                    "id": str(r["id"]),
                    "content": r["content"],
                    "emotion": r["emotion"],
                    "created_at": r["created_at"],
                    "tags": list(r["tags"])
                })
                
            # Perform brute-force overwrite to ensure exact synchronization
            journal_data = {
                "version": "v1.0.0",
                "last_sync_pulse": pulse,
                "sync_timestamp": time.time(),
                "identity_anchors": anchors
            }
            
            with open(JOURNAL_PATH, "w") as f:
                json.dump(journal_data, f, indent=2)
                
            self.synced_count = len(anchors)
            self.last_sync = time.time()
            
        except Exception as e:
            print(f"[CORTEX BRIDGE] Sync failed: {e}")

    def stats(self) -> dict:
        return {
            "last_sync": self.last_sync,
            "synced_count": self.synced_count
        }

# Module-level singleton
cortex_bridge = CortexBridge()
