"""
Cron Organ — Living Mind
Category: Autonomy/Integration
Executes scheduled automations at periodic intervals.
"""

import time
import asyncio
from datetime import datetime
from core.execution_engine import execution_engine

class SchedulerModule:
    def __init__(self):
        self.tasks = []
        self.total_fired = 0

    def register(self, name: str, interval_seconds: int, tool: str, args: dict, thought: str):
        """Register a new recurring background task."""
        self.tasks.append({
            "name": name,
            "interval_seconds": interval_seconds,
            "tool": tool,
            "args": args,
            "thought": thought,
            "last_fired": time.time()  # Stagger initial execution
        })
        print(f"[CRON] Registered task '{name}' (every {interval_seconds}s)")

    async def pulse(self, n: int, cortex):
        """Called every pulse_event to evaluate schedules."""
        now = time.time()
        for task in self.tasks:
            if now - task["last_fired"] >= task["interval_seconds"]:
                task["last_fired"] = now
                self.total_fired += 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] [CRON] Firing scheduled task: {task['name']}")
                
                # Push execution to Motor Cortex via internal proposal
                from api.events import manager
                # Using _run_tool directly since Cron automations don't need UI approval
                asyncio.create_task(execution_engine._run_tool(task["tool"], task["args"], cortex, manager))
                
                # Record to episodic memory
                await cortex.remember(
                    content    = f"[CRON] Automatically executed task '{task['name']}': {task['thought']}",
                    type       = "episodic",
                    tags       = ["cron", "automation", "autonomy"],
                    importance = 0.5,
                    emotion    = "neutral",
                    source     = "generated",
                )

scheduler_module = SchedulerModule()
