"""
Living Mind — API entry point
FastAPI + lifespan that boots the runtime.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from core.runtime import runtime
from api.events import manager
from state.telemetry_broker import telemetry_broker
from core.security_perimeter import immune
from cortex.engine import cortex
from api.agent_gateway import router as agent_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await runtime.birth()
    yield
    await runtime.death()

app = FastAPI(
    title="Living Mind",
    description="A real living digital runtime.",
    version="1.0.0",
    lifespan=lifespan,
)

# Agent Gateway — bridge between Living Mind and Antigravity
app.include_router(agent_router)

from fastapi.staticfiles import StaticFiles
import os

# Mount the UI directory at the root
ui_path = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(ui_path):
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")

# ------------------------------------------------------------------------------
# REST API (MALKHUT)
# ------------------------------------------------------------------------------

@app.get("/status")
async def status():
    return await runtime.vitals()

@app.get("/memory/stats")
async def memory_stats():
    return await cortex.stats()

@app.get("/memory/identity")
async def identity():
    return {"summary": await cortex.identity_summary()}

@app.get("/memory/autobio")
async def get_autobio():
    from cortex.autobio import autobio
    story = await autobio.life_story()
    return {"narrative": story}

@app.get("/memory/recall")
async def recall(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, le=50),
    memory_type: str = Query(None),
):
    memories = await cortex.recall(q, limit=limit, memory_type=memory_type)
    return [
        {
            "id":          m.id,
            "content":     m.content,
            "type":        m.type,
            "emotion":     m.emotion,
            "importance":  round(m.importance, 3),
            "tags":        m.tags,
            "is_identity": m.is_identity,
            "is_flashbulb": m.is_flashbulb,
        }
        for m in memories
    ]

@app.get("/hormones")
async def hormones():
    return telemetry_broker.snapshot()

@app.get("/circadian")
async def circ():
    from chemistry.circadian import circadian
    return circadian.snapshot()

@app.get("/awakening")
async def awakening_stats():
    from core.awakening import awakening
    return awakening.stats()

@app.get("/dreams")
async def dream_stats():
    from core.dreams import dreams
    return dreams.stats()

# ------------------------------------------------------------------------------
# WEBSOCKETS (THE 22 PATHS)
# ------------------------------------------------------------------------------

@app.websocket("/ws/pulse")
async def websocket_pulse(websocket: WebSocket):
    """Output stream for the UI tree animations"""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # keepalive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def _parse_bash_intent(text: str) -> str:
    """
    Deterministically extract a bash command from natural language.
    Strips intent verbs so 'run git status' → 'git status',
    'can you execute pip install requests' → 'pip install requests'.
    Falls back to empty string if no recognizable command is found.
    """
    import re
    t = text.strip()
    # Strip surrounding backticks or quotes
    t = t.strip("`\"'")
    # Strip markdown bash fences
    t = re.sub(r'^```\w*\s*', '', t, flags=re.IGNORECASE).strip('`').strip()
    # Strip leading intent phrases (order matters — longest first)
    INTENT_PREFIXES = [
        r"^can you (please )?",
        r"^please (run|execute|do) ",
        r"^(run|execute|do|try) the (command|bash command|following) ",
        r"^(run|execute|do|try) this[:\s]+",
        r"^(run|execute|bash|do|try|use|call|invoke) ",
        r"^(the command is|command)[:\s]+",
        r"^bash[:\s]+",
    ]
    for pattern in INTENT_PREFIXES:
        t = re.sub(pattern, '', t, flags=re.IGNORECASE).strip()
    # Must look like a real command: starts with a known binary or path
    KNOWN_COMMANDS = [
        'ls', 'cat', 'git', 'pip', 'pip3', 'python', 'python3',
        'npm', 'npx', 'node', 'sudo', 'apt', 'systemctl', 'curl', 'wget',
        'grep', 'find', 'chmod', 'chown', 'cp', 'mv', 'mkdir', 'rm',
        'echo', 'ps', 'kill', 'df', 'du', 'top', 'htop', 'which',
        'env', 'export', 'source', 'cd', 'pwd', 'whoami', 'uname',
        'docker', 'docker-compose', 'make', 'cargo', 'go', 'rustc',
        './', '/', '~/',
    ]
    first_word = t.split()[0] if t.split() else ""
    if any(first_word == cmd or t.startswith(cmd) for cmd in KNOWN_COMMANDS):
        return t
    return ""   # not a recognizable command — fall through to LLM


# Module-level conversation context — persists across WS reconnects
_ctx = {"last_file": None, "last_dir": None}

@app.websocket("/ws/stimulus")
async def websocket_stimulus(websocket: WebSocket):
    """Input stream for UI interventions flowing up the tree"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            node = data.get("node")
            
            if node == "malkhut" and "text" in data:
                text_input = data['text']
                await cortex.remember(
                    content=f"[USER STIMULUS] {text_input}",
                    type="episodic", tags=["senses", "input"],
                    importance=0.9, emotion="surprise", source="experienced"
                )
                
                from core.runtime import runtime
                from core.execution_engine import execution_engine
                brain_inst = runtime.mapped_instances["brain"]
                # Directly invoke the Brain's ReAct agentic loop
                decision = await brain_inst.think(runtime.event_loops, cortex, immune, user_stimulus=text_input)
                
                if decision:
                    # Strip internal simulation tags for the UI
                    import re as _re
                    thought = _re.sub(r'\[Simulation:.*?\]', '', decision.get("thought", ""), flags=_re.DOTALL).strip()
                    if thought:
                        await manager.broadcast_event("chat_reply", thought)
                    
                    # Note: If decision["type"] == "act", brain.think() already automatically 
                    # invokes execution_engine.propose_action(), which broadcasts the proposal to the UI.
                else:
                    await manager.broadcast_event("chat_reply", "(Neural lag — I could not form a cohesive thought.)")
            elif node == "hesed":
                # Manual Dopamine/expansion spike
                telemetry_broker.inject("dopamine", 0.3, "manual stimulus")
                await manager.broadcast_event("hesed", "Expansion stimulated")
                
            elif node == "gevurah":
                # Manual SecurityPerimeter/pruning spike
                immune.report("manual_stimulus", success=False, category="Defense")
                await manager.broadcast_event("gevurah", "Pruning stimulated")
                
            elif node == "approve":
                from core.execution_engine import execution_engine
                msg = await execution_engine.execute_approved(cortex, manager)
                await manager.broadcast_event("chat_reply", msg)
                
            elif node == "reject":
                from core.execution_engine import execution_engine
                msg = await execution_engine.reject()
                await manager.broadcast_event("chat_reply", msg)
                
            # Additional Sephirot mappings can be captured here
            
            await websocket.send_json({"status": "received"})
    except WebSocketDisconnect:
        pass
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8008, reload=False)
