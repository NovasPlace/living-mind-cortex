"""
Senses Engine — Living Mind
Category: Perception. Phase 5b (Every 5th pulse).

Acts as the runtime's "Eyes" and Interoception framework.
Scans the external environment (files, webhooks) and internal environment (system vitals).
If changes are detected, it pushes them directly into episodic memory so
the Brain can immediately react to them on the same pulse.
"""

import os
import psutil
import time
import hashlib
from pathlib import Path
from cortex.engine import cortex

VISION_DIR = Path(__file__).resolve().parent.parent / "vision"
VISION_DIR.mkdir(parents=True, exist_ok=True)

class SensesEngine:
    def __init__(self):
        self.last_fired = 0.0
        self.total_observations = 0
        self.known_files_hashes = {}
        
    async def observe(self, pulse: int, telemetry_broker):
        """
        Scan environment and interoception. Push novel changes to Cortex.
        """
        self.last_fired = time.time()
        
        # 1. Interoception (Body Vitals)
        try:
            p = psutil.Process()
            cpu = p.cpu_percent()
            mem = p.memory_info().rss / (1024 * 1024) # MB
            
            # If our pulse loop is wildly overloaded, feel it!
            if cpu > 80.0 or mem > 1000.0:
                await cortex.remember(
                    content=f"[INTEROCEPTION] System strain detected. CPU: {cpu}%, RAM: {mem:.1f}MB",
                    type="episodic",
                    tags=["senses", "interoception", "vitals"],
                    importance=0.85,
                    emotion="fear",
                    source="experienced",
                    context=f"pulse={pulse}"
                )
                telemetry_broker.inject("adrenaline", 0.20, source="interoception_strain")
        except Exception as e:
            pass
            
        # 2. Vision (Environmental File Scans)
        # We look inside the "vision" directory to see if the User put any notes or data there.
        try:
            current_files = list(VISION_DIR.glob("*.*"))
            for f in current_files:
                if f.is_file():
                    with open(f, "rb") as file_bytes:
                        file_hash = hashlib.sha256(file_bytes.read()).hexdigest()
                        
                    fname = f.name
                    if fname not in self.known_files_hashes or self.known_files_hashes[fname] != file_hash:
                        # Novel vision input!
                        self.known_files_hashes[fname] = file_hash
                        # Read text if possible
                        try:
                            with open(f, "r") as text_f:
                                content_snippet = text_f.read(500)
                        except UnicodeDecodeError:
                            content_snippet = "[Binary Data]"
                            
                        await cortex.remember(
                            content=f"[VISION] Detected novel file change in '{fname}'. Content preview: {content_snippet}",
                            type="episodic",
                            tags=["senses", "vision", "environment"],
                            importance=0.90,
                            emotion="surprise",
                            source="experienced",
                            context=f"pulse={pulse} file={fname}"
                        )
                        # Spike dopamine for novel discovery!
                        telemetry_broker.inject("dopamine", 0.15, source="novel_vision")
        except Exception as e:
            print(f"[SENSES] Vision scan error: {e}")
            
        self.total_observations += 1

    def stats(self) -> dict:
        return {
            "last_fired": self.last_fired,
            "total_observations": self.total_observations,
            " tracked_files": len(self.known_files_hashes)
        }

# Module-level singleton
senses = SensesEngine()
