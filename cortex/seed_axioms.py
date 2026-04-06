import asyncio
import uuid
import time
import json
from cortex.engine import cortex

AXIOMS = [
    {
        "content": "GitHub Trending Heuristic: Repository names are almost always formatted as 'user / repo' and are contained within <h2> or <h3> tags inside <article class='Box-row'> elements.",
        "tags": ["axiom", "domain_knowledge", "github", "navigation"],
        "importance": 1.0
    },
    {
        "content": "Wikipedia Navigation: To search Wikipedia, use the 'Search' input. Definitions are almost always in the first paragraph (<p> tag) of the main content area, following any bolded article title.",
        "tags": ["axiom", "domain_knowledge", "wikipedia", "navigation"],
        "importance": 0.9
    },
    {
        "content": "Reddit Stealth Workaround: Standard Reddit is extremely hostile to automation. Always prefer 'old.reddit.com' for data extraction and navigation. If blocked, use DuckDuckGo Lite to search for the thread URL instead of direct navigation.",
        "tags": ["axiom", "domain_knowledge", "reddit", "stealth", "navigation"],
        "importance": 1.0
    },
    {
        "content": "Interactive Element Heuristic: Form buttons and menu items often lack distinct text. Look for [LINK: ...] markers in your sensory input as these represent high-signal navigation paths.",
        "tags": ["axiom", "domain_knowledge", "general", "navigation"],
        "importance": 0.8
    },
    {
        "content": "Search Proxy Rule: If a website returns a 'Network Security' or 'Access Denied' message, do not keep trying the same URL. Switch to a Search Engine Proxy (DuckDuckGo Lite) to find an alternate path to the content.",
        "tags": ["axiom", "domain_knowledge", "general", "stealth"],
        "importance": 1.0
    }
]

async def seed():
    print("🌱 SEEDING CORTEX WITH AXIOMATIC DOMAIN KNOWLEDGE...")
    await cortex.connect()
    try:
        for axiom in AXIOMS:
            # Fast path: check existence via direct SQL, not the full recall() pipeline
            # (recall() triggers biases, priming, reconsolidation — wrong for a seed check)
            async with cortex._pool.acquire() as conn:
                exists = await conn.fetchval("""
                    SELECT 1 FROM memories
                    WHERE content LIKE $1 AND type = 'semantic'
                    LIMIT 1
                """, axiom["content"][:60] + "%")
            if exists:
                print(f"Skipping existing axiom: {axiom['content'][:60]}...")
                continue

            await cortex.remember(
                content=axiom["content"],
                type="semantic",
                tags=axiom["tags"],
                importance=axiom["importance"],
                source="told",
                emotion="neutral"
            )
            print(f"Seeded axiom: {axiom['content'][:60]}...")
            
        print("✅ SEEDING COMPLETE.")
    finally:
        await cortex.disconnect()

if __name__ == "__main__":
    asyncio.run(seed())
