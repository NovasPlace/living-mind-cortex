import asyncio
import httpx
import uuid
import time
import numpy as np
from cortex.engine import cortex
from cortex.thermorphic import encode_atom
import json
import logging

logging.basicConfig(level=logging.INFO)

async def test_full_socket():
    print("====================================")
    print("Testing Full API to Mock vLLM Socket")
    print("====================================")
    
    await cortex.connect()
    
    # 1. Inject a code_expert memory so the HSM has thermal mass to resonate with
    print("\\n[Test] Forging synthetic long-term memory for code_expert...")
    code_hvec = encode_atom("def quicksort(arr): if len(arr) <= 1: return arr", dim=256).astype(np.float32)
    code_id = str(uuid.uuid4())
    
    async with cortex._pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO memories (id, content, type, tags, importance, created_at, last_accessed, access_count, emotion, confidence, context, source, linked_ids, metadata, embedding)
            VALUES ($1, 'def quicksort(arr): if len(arr) <= 1: return arr', 'episodic', '{}', 0.49, 1.0, 1.0, 0, 'neutral', 1.0, '', 'experienced', '{}', '{"cognitive_domain": "code_expert"}', $2)
        """, code_id, code_hvec.tobytes())
    
    # 2. Wait for background servers to be ready
    await asyncio.sleep(2)
    
    print("\\n[Test] Sending POST request to Sovereign Daemon (/api/invoke)...")
    payload = {
        "prompt": "def quicksort(arr): if len(arr) <= 1: return arr"
    }
    
    # We hit the local Sovereign daemon (assuming it's running on port 8009)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post("http://localhost:8009/api/invoke", json=payload)
            response.raise_for_status()
            data = response.json()
            
            print(f"\\n--- SOVEREIGN API RESPONSE ---")
            print(json.dumps(data, indent=2))
        except httpx.HTTPError as e:
            print(f"API Error: {e}")
            
    # Cleanup
    async with cortex._pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1::uuid", code_id)
        
    await cortex.disconnect()

if __name__ == "__main__":
    asyncio.run(test_full_socket())
