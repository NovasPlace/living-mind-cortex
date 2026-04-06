import asyncio
import time
import json
import logging
from core.runtime import runtime
from core.execution_engine import execution_engine
from cortex.engine import cortex
from core.security_perimeter import immune
from api.events import ConnectionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AUTODIDACT")

# Mute standard uvicorn/access logs to keep the console clean
logging.getLogger("uvicorn").setLevel(logging.WARNING)

class MockManager(ConnectionManager):
    """A mock manager so the autodidact can capture broadcast output without breaking the real UI."""
    def __init__(self):
        super().__init__()
        self.last_broadcast = ""

    async def broadcast_event(self, event_type: str, data: str):
        self.last_broadcast = data
        logger.info(f"[{event_type.upper()}] {data}")

class Autodidact:
    def __init__(self):
        from core.orchestrator import brain
        self.brain = brain
        self.manager = MockManager()
        self.nav_history = []  # To detect refresh loops
        
        self.curriculum = [
            {
                "goal": "Navigate to news.ycombinator.com and extract the exact title of the very first story.",
                "verification": "The output must contain the exact story title from the DOM."
            },
            {
                "goal": "Navigate to github.com/trending. Extract the name of the top trending repository.",
                "verification": "The output must contain the top trending repo name."
            },
            {
                "goal": "Navigate to en.wikipedia.org/wiki/Main_Page. Find the 'Search' box, type 'Turing Test', and press Enter. Extract the first sentence defining the Turing Test.",
                "verification": "The output must contain the definition of the Turing Test."
            },
            {
                "goal": "Navigate to dev.to. Extract the title of the first featured article.",
                "verification": "The output must contain the featured article title."
            },
            {
                "goal": "If direct navigation is blocked, use duckduckgo.com to search for 'reddit technology top post'. Navigate to the Reddit result and extract the title of the top post.",
                "verification": "The output must contain the top reddit post title."
            }
        ]

    async def train(self):
        logger.info("🟢 INITIATING AUTODIDACT TRAINING PROTOCOL 🟢")
        try:
            await cortex.connect()
            
            # Start the learning loop
            for i, task in enumerate(self.curriculum):
                logger.info(f"\n======================================")
                logger.info(f"🎓 LEVEL {i + 1} EXPERIMENT")
                logger.info(f"TARGET: {task['goal']}")
                logger.info(f"======================================")
                
                # Task Setup
                session_id = f"autodidact_{i+1}_{int(time.time())}"
                attempts = 0
                max_attempts = 10
                success = False
                
                while attempts < max_attempts and not success:
                    attempts += 1
                    logger.info(f"\n--- Attempt {attempts}/{max_attempts} ---")
                    
                    # Contextualize the prompt with the current goal
                    system_stimulus = (
                        f"AUTODIDACT TRAINING PROTOCOL. Your goal is: {task['goal']}. "
                        f"You have a persistent browser. Use 'act' to browse_web (goto, click_text, type_text, press_key, scroll).\n"
                        f"Look at the [MOTOR SENSORY] outputs in your memory. If you have the answer, summarize it. If stuck, PIVOT to a new strategy (scroll, click, search)!"
                    )
                    
                    decision = await self.brain.think(runtime.event_loops, cortex, immune, user_stimulus=system_stimulus)
                    
                    if not decision:
                        logger.warning("Brain failed to formulate a decision.")
                        continue
                    
                    action_type = decision.get("type")
                    thought = decision.get("thought", "")
                    
                    logger.info(f"🧠 THOUGHT: {thought}")
                    
                    # LOOP GUARD: Detect recursive goto refresh loops
                    if action_type == "act":
                        args = decision.get("arguments", {})
                        if args.get("action") == "goto":
                            url = args.get("url", "")
                            self.nav_history.append(url)
                            # If we've hit the same URL 3 times in a row, inject frustration
                            if len(self.nav_history) >= 3 and all(u == url for u in self.nav_history[-3:]):
                                logger.warning(f"⚠️ LOOP DETECTED: {url}. Injecting frustration.")
                                await cortex.remember(
                                    content=f"I am STUCK in a refresh loop on {url}. Navigating there again is NOT WORKING. I must try clicking a specific link, scrolling, or searching instead. Do not repeat the same goto!",
                                    type="episodic",
                                    tags=["autodidact", "frustration", "loop_detected", f"session:{session_id}"],
                                    importance=0.9,
                                    emotion="anger"
                                )
                                # Clear history so we don't spam
                                self.nav_history = []

                        tool = decision.get("tool_call")
                        
                        logger.info(f"🦾 ACTUATING: {tool} with {args}")
                        
                        # BYPASS APPROVAL: Directly invoke the tool runner for autonomous training
                        output, display = await execution_engine._run_tool(tool, args, cortex, self.manager)
                        
                        logger.info(f"👁️ RESULT:\n{display[:300]}...\n")
                        
                        # Evaluate success using the LLM itself as a judge
                        judge_prompt = (
                            f"You are the Teacher. The goal was: {task['goal']}.\n"
                            f"The AgentRuntime just executed a tool and got this output:\n{display[:1500]}\n\n"
                            f"Did the AgentRuntime successfully extract the required information to fulfill the goal? Yes or No?"
                        )
                        judgement = await self.brain._call_llm(judge_prompt)
                        
                        if "yes" in judgement.lower() and "no" not in judgement.lower()[:10]:
                            logger.info("✅ SUCCESS! Task complete.")
                            # EVOLUTION: Distill the successful trajectory into a permanent Skill
                            from core.evolution import evolution
                            await evolution.distill_skill(session_id, task['goal'])

                            # Store the successful strategy in Semantic Memory
                            await cortex.remember(
                                f"I successfully learned to: {task['goal']}. The winning strategy was: {thought}",
                                type="semantic",
                                tags=["autodidact", "success", "web_navigation", f"session:{session_id}"],
                                importance=1.0,
                                emotion="joy"
                            )
                            # Close the browser to reset for the next task
                            await execution_engine._run_tool("browse_web", {"action": "close"}, cortex, self.manager)
                            success = True
                        else:
                            logger.info("❌ FAILED OR IN PROGRESS. Evaluating next steps...")
                            # Store the failure mode
                            await cortex.remember(
                                f"While trying to {task['goal']}, I tried: {args}. It did not yield the final answer. I need to keep trying.",
                                type="episodic",
                                tags=["autodidact", "failure", "web_navigation", f"session:{session_id}"],
                                importance=0.6,
                                emotion="anger"
                            )
                            
                    else:
                        logger.info("AgentRuntime is pondering without acting...")
                        
                    # Pause to let the DOM settle and throttle LLM API calls
                    await asyncio.sleep(4)

                if not success:
                    logger.error(f"☠️ ORGANISM FAILED LEVEL {i + 1} AFTER {max_attempts} ATTEMPTS. Shutting down browser.")
                    await execution_engine._run_tool("browse_web", {"action": "close"}, cortex, self.manager)
                
                # COOLDOWN between levels for stealth
                logger.info("⏳ COOLDOWN: Sleeping 15s to avoid IP rate-limiting...")
                await asyncio.sleep(15)
                    
            logger.info("🏁 AUTODIDACT CURRICULUM COMPLETE.")

        finally:
            logger.info("🔻 Shutting down autodidact resources...")
            await execution_engine._run_tool("browse_web", {"action": "close"}, cortex, self.manager)
            await self.brain.close()
            await cortex.disconnect()

if __name__ == "__main__":
    tutor = Autodidact()
    asyncio.run(tutor.train())
