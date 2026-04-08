import asyncio
import numpy as np
import uuid
import time
import json
from cortex.engine import cortex
from cortex.router import BiomechanicRouter
from cortex.thermorphic import encode_atom

async def test_biomechanic_router():
    print("====================================")
    print("Testing Biomechanic LoRA Router")
    print("====================================")
    
    # Connect to the cortex DB
    await cortex.connect()
    
    # 0. Clean up any previous test runs that crashed
    async with cortex._pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE created_at = 1.0")
        
    router = BiomechanicRouter(cortex)
    
    # 1. Inject domain-specific memories purely for testing
    print("[Test] Forging synthetic long-term memory...")
    
    # We create distinct, orthogonal-ish random vectors so resonance works.
    code_hvec = np.random.randn(256).astype(np.float32)
    code_hvec = code_hvec / np.linalg.norm(code_hvec)
    
    logic_hvec = np.random.randn(256).astype(np.float32)
    logic_hvec = logic_hvec / np.linalg.norm(logic_hvec)
    
    # We will bypass remember() pipeline to directly inject bytea embeddings for a deterministic check
    queries = []
    
    code_id_1 = str(uuid.uuid4())
    code_id_2 = str(uuid.uuid4())
    logic_id_1 = str(uuid.uuid4())
    logic_id_2 = str(uuid.uuid4())
    
    async with cortex._pool.acquire() as conn:
        # Note: In Python asyncpg, $1 syntax requires passing args normally, but since we are looping, we'll do it safely.
        await conn.execute("""
            INSERT INTO memories (id, content, type, tags, importance, created_at, last_accessed, access_count, emotion, confidence, context, source, linked_ids, metadata, embedding)
            VALUES ($1, 'def fibonacci(n): return n if n<=1 else ...', 'episodic', '{}', 0.4, 1.0, 1.0, 0, 'neutral', 1.0, '', 'experienced', '{}', '{"cognitive_domain": "code_expert"}', $2)
        """, code_id_1, code_hvec.tobytes())
        
        await conn.execute("""
            INSERT INTO memories (id, content, type, tags, importance, created_at, last_accessed, access_count, emotion, confidence, context, source, linked_ids, metadata, embedding)
            VALUES ($1, 'class BaseAgent(object): def init(self)...', 'episodic', '{}', 0.45, 1.0, 1.0, 0, 'neutral', 1.0, '', 'experienced', '{}', '{"cognitive_domain": "code_expert"}', $2)
        """, code_id_2, code_hvec.tobytes())
        
        await conn.execute("""
            INSERT INTO memories (id, content, type, tags, importance, created_at, last_accessed, access_count, emotion, confidence, context, source, linked_ids, metadata, embedding)
            VALUES ($1, 'The process requires sequential analysis', 'episodic', '{}', 0.49, 1.0, 1.0, 0, 'neutral', 1.0, '', 'experienced', '{}', '{"cognitive_domain": "logic_expert"}', $2)
        """, logic_id_1, logic_hvec.tobytes())
        
        await conn.execute("""
            INSERT INTO memories (id, content, type, tags, importance, created_at, last_accessed, access_count, emotion, confidence, context, source, linked_ids, metadata, embedding)
            VALUES ($1, 'Therefore, conclusion A strictly implies conclusion B', 'episodic', '{}', 0.45, 1.0, 1.0, 0, 'neutral', 1.0, '', 'experienced', '{}', '{"cognitive_domain": "logic_expert"}', $2)
        """, logic_id_2, logic_hvec.tobytes())

    import logging
    logging.basicConfig(level=logging.INFO)

    print("\\n[Test] Sending prompt resonating with code...")
    # Add slight noise to demonstrate resonance matching
    prompt_code = code_hvec + (np.random.randn(256).astype(np.float32) * 0.1)
    prompt_code /= np.linalg.norm(prompt_code)
    
    active_lora = await router.route_prompt(prompt_code)
    print(f"-> Selected LoRA: {active_lora}")
    assert active_lora == "code_expert", f"Expected code_expert, got {active_lora}"
    
    print("\\n[Test] Sending prompt resonating with logic...")
    prompt_logic = logic_hvec + (np.random.randn(256).astype(np.float32) * 0.1)
    prompt_logic /= np.linalg.norm(prompt_logic)
    
    active_lora = await router.route_prompt(prompt_logic)
    print(f"-> Selected LoRA: {active_lora}")
    assert active_lora == "logic_expert", f"Expected logic_expert, got {active_lora}"

    print("\\n[Test] Cleanup injected memories...")
    async with cortex._pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = ANY($1::uuid[])", [code_id_1, code_id_2, logic_id_1, logic_id_2])
        
    await cortex.disconnect()
    print("\\nALL MoE ROUTING TESTS PASSED.")

if __name__ == "__main__":
    asyncio.run(test_biomechanic_router())
