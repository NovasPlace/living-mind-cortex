import asyncio
from cortex.engine import cortex
async def test():
    await cortex.connect()
    print("cortex pool:", cortex._pool)
asyncio.run(test())
