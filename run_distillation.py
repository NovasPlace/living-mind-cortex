"""
Distillation Runner — seeds the substrate, runs pulses to crystallize
causal nodes, then fires the GRPO training loop.
"""

import cortex.thermorphic as t
import cortex.distillation as d
import numpy as np

print("=" * 60)
print("  SOVEREIGN DISTILLATION RUNNER")
print("=" * 60)

# ── 1. Build and seed a substrate ───────────────────────────────
print("\n[1/4] Seeding substrate...")
sub = t.ThermorphicSubstrate()

memory_seed = [
    # Identity floors (anchor_temperature > 0 = pinned forever)
    ("Sovereign Identity: Zero-Trust agent. Never eval, never exec, never shell=True.", 2.0, 1.0),
    ("Sovereign Directive: Composition over inheritance. Functions under 40 LOC.", 2.0, 0.9),

    # Hot episodic memories (will fuse into causal nodes)
    ("asyncpg create_pool raised InvalidAuthorizationError on peer auth failure", 2.0, 0.0),
    ("peer auth failed because pg_hba.conf uses peer not md5 for local connections", 2.0, 0.0),
    ("fix: add host=localhost to DATABASE_URL forces md5 handshake bypassing peer", 2.0, 0.0),
    ("Zola daemon mission aborted with LLM network partition error at cognition.py", 1.9, 0.0),
    ("Ollama API endpoint returned connection refused on port 11434", 1.9, 0.0),
    ("fix: start ollama serve before launching Zola daemon resolves partition", 1.9, 0.0),
    ("thermorphic phase encoding swap O(N^2) to O(N) phase addition resolved HRR collapse", 1.8, 0.0),
    ("MoVE cross-attention guard now filters episodic injection against identity floors", 1.8, 0.0),
]

for content, temp, anchor in memory_seed:
    sub.inject(content, temperature=temp, anchor_temperature=anchor)

# Wire adjacency so thermal diffusion causes fusion events
nids = list(sub.nodes.keys())
for i, nid in enumerate(nids[:-1]):
    sub.nodes[nid].edges.append(nids[i + 1])

print(f"   Injected {len(sub.nodes)} nodes | identity floors: "
      f"{sum(1 for n in sub.nodes.values() if n.anchor_temperature > 0.0)}")

# ── 2. Run pulses until causal crystals form ─────────────────────
print("\n[2/4] Running pulses to crystallize causal nodes...")
for step in range(50):
    result = sub.pulse()
    crystals = sum(1 for n in sub.nodes.values() if n.immutable)
    causal   = sum(1 for n in sub.nodes.values() if "[CAUSAL]" in n.content and n.immutable)
    if step % 10 == 0 or causal > 0:
        print(f"   pulse={step+1:02d} | nodes={len(sub.nodes)} | crystals={crystals} | causal_frozen={causal}")
    if causal >= 3 and step > 15:   # enough material, stop early
        print(f"   Sufficient causal crystals at pulse {step+1}. Stopping.")
        break

# ── 3. Confirm corpus ────────────────────────────────────────────
print("\n[3/4] Building distillation corpus...")
corpus = d.build_distillation_corpus(sub.nodes)
print(f"   Corpus: {len(corpus)} eligible [CAUSAL] crystals")
if not corpus:
    print("   ❌ Corpus empty — aborting. Check MIN_MATURITY_PULSE or pulse count.")
    raise SystemExit(1)

for i, item in enumerate(corpus[:3]):
    print(f"   [{i}] {item['prompt'][-1]['content'][:80]}")

# ── 4. Fire distillation ────────────────────────────────────────
print("\n[4/4] Firing GRPO distillation...")
print("   Model:", d.GRPO_MODEL_NAME)
print("   Output:", d.OUTPUT_DIR)
print("   WARNING: This writes back to model weights.")
print()
d.run_distillation(sub.nodes)
print("\n✅ Distillation complete.")
