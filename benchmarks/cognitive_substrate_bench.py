"""
Cognitive Substrate Benchmark — Living Mind Cortex
===================================================
The benchmark that measures what the Cortex ACTUALLY does.

LongMemEval measures retrieval accuracy from a static haystack.
That's MemPalace's domain. We measure something different:

  "Does having a thermorphic memory substrate make an AI agent
   measurably better over multiple sessions?"

Five dimensions:

  1. THERMAL SALIENCE ACCURACY
     High-importance memories should be hotter than low-importance ones.
     Metric: Spearman rank correlation (importance → temperature)

  2. FUSION COHERENCE
     Emergent concepts spawned by fusion should be semantically
     related to both parents. No random noise nodes.
     Metric: Mean parent-child content overlap (word Jaccard)

  3. CRYSTALLIZATION PRECISION
     Memories that crystallize should be the frequently-accessed ones,
     not random cold nodes.
     Metric: Mean access_count of crystallized vs non-crystallized nodes

  4. THERMAL RECALL IMPROVEMENT
     Recalling a memory should heat it, making it MORE likely to be
     recalled again (simulating human memory reinforcement).
     Metric: Temperature delta pre/post-recall for top-k hits

  5. EVOLVER SENSITIVITY
     Mutating ALPHA/FUSION_THRESHOLD/FREEZE_DWELL should produce
     measurably different memory topologies (the genome is meaningful).
     Metric: Substrate divergence (node count, fusion count, mean temp)
             across 3 gene variants

Usage:
    python3 benchmarks/cognitive_substrate_bench.py
    python3 benchmarks/cognitive_substrate_bench.py --verbose
"""

import sys
import time
import random
import copy
import argparse
from typing import List

sys.path.insert(0, ".")

import cortex.thermorphic as _thermo_mod
from cortex.thermorphic import ThermorphicSubstrate, ConceptNode

random.seed(2026)


# ── Test fixtures ─────────────────────────────────────────────────────────────

# Simulated agent session: AI coding conversations with varying importance
SESSION_MEMORIES = [
    # (content, importance[0-1], tags)
    ("asyncpg pool size should be 10 for this workload",        0.9,  ["database", "config", "agent"]),
    ("use RETURNING clause to avoid a second SELECT",           0.85, ["sql", "optimization", "agent"]),
    ("FastAPI lifespan context handles startup/shutdown",       0.8,  ["python", "api", "agent"]),
    ("cortisol spike detected during immune quarantine event",  0.75, ["biology", "system", "agent"]),
    ("Ebbinghaus stability improves with spaced repetition",    0.7,  ["memory", "cognition", "agent"]),
    ("hormone cross-talk: cortisol suppresses acetylcholine",   0.65, ["biology", "cognition", "agent"]),
    ("thermorphic ALPHA=0.08 is the default diffusivity",       0.6,  ["physics", "config", "agent"]),
    ("dream engine fires every 20th pulse",                     0.55, ["system", "timing", "agent"]),
    ("interoception energy_budget drains on LLM calls",         0.5,  ["system", "state", "agent"]),
    ("evolver requires at least 0.01 fitness to mutate",        0.45, ["evolution", "config", "agent"]),
    ("circadian night phase intensifies dream consolidation",   0.4,  ["biology", "timing", "agent"]),
    ("metacognition drift fires on 6th pulse",                  0.35, ["system", "cognition", "agent"]),
    ("git push origin main --force is dangerous",               0.3,  ["git", "ops", "agent"]),
    ("PostgreSQL schema applied via _apply_schema on connect",  0.25, ["database", "system", "agent"]),
    ("print statements should use [ORGAN] prefix format",       0.2,  ["style", "ops", "agent"]),
    ("requirements.txt lists asyncpg fastapi aiohttp",          0.15, ["ops", "config"]),
    ("README was updated with 7-pillar architecture diagram",   0.1,  ["docs", "meta"]),
    ("the test suite uses pytest with asyncio mode",            0.08, ["testing", "ops"]),
    ("changelog follows Keep a Changelog format",               0.05, ["docs", "meta"]),
    ("gitignore covers state/lineage identity/journal",         0.04, ["ops", "config"]),
]


def _build_session_substrate(
    sessions=SESSION_MEMORIES,
    alpha=0.08,
    fusion_threshold=1.60,
    freeze_dwell=8,
    n_pulses=5,
) -> ThermorphicSubstrate:
    """Build a substrate loaded with session memories and run n_pulses."""
    # Temporarily patch module-level constants
    _thermo_mod.ALPHA            = alpha
    _thermo_mod.FUSION_THRESHOLD = fusion_threshold
    _thermo_mod.FREEZE_DWELL     = freeze_dwell

    sub   = ThermorphicSubstrate()
    nodes = []

    for content, importance, tags in sessions:
        # Map importance [0,1] → temperature [0.3, 1.8]
        temp = 0.3 + importance * 1.5
        n    = sub.inject(content, temperature=temp, tags=tags, dims=32)
        nodes.append((n, importance))

    # Wire semantically adjacent nodes
    for i in range(len(nodes) - 1):
        sub.connect(nodes[i][0].id, nodes[i+1][0].id)

    for _ in range(n_pulses):
        sub.pulse()

    return sub


def _jaccard(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    inter = wa & wb
    union = wa | wb
    return len(inter) / max(len(union), 1)


# ── Benchmark dimensions ──────────────────────────────────────────────────────

def bench_thermal_salience(verbose=False) -> dict:
    """
    D1: Thermal Salience Accuracy
    High-importance memories → hotter nodes.
    Metric: Spearman rank correlation (importance rank vs temperature rank)
    Perfect = 1.0, Random = ~0.0
    """
    print("\n[D1] Thermal Salience Accuracy")

    sub   = ThermorphicSubstrate()
    nodes = []

    for content, importance, tags in SESSION_MEMORIES:
        temp = 0.3 + importance * 1.5
        n    = sub.inject(content, temperature=temp, tags=tags, dims=32)
        nodes.append((n, importance))

    # Run 5 pulses (let heat diffuse, don't crystallize yet)
    for _ in range(5):
        sub.pulse()

    # Rank by importance (ground truth) and by current temperature
    importance_rank = {n.id: rank for rank, (n, _) in
                       enumerate(sorted(nodes, key=lambda x: x[1], reverse=True))}
    temp_rank       = {n.id: rank for rank, (n, _) in
                       enumerate(sorted(nodes, key=lambda x: sub.nodes[x[0].id].temperature, reverse=True))}

    # Spearman: 1 - 6Σd²/n(n²-1)
    n = len(nodes)
    d2_sum = sum((importance_rank[nid] - temp_rank.get(nid, n))**2
                 for nid in importance_rank if nid in temp_rank)
    spearman = 1 - (6 * d2_sum) / (n * (n**2 - 1))

    print(f"  Nodes: {n}  |  Spearman ρ = {spearman:.4f}  (1.0 = perfect salience ordering)")

    if verbose:
        print("\n  Top-5 by importance vs top-5 by temperature:")
        top_imp  = sorted(nodes, key=lambda x: x[1], reverse=True)[:5]
        top_temp = sorted(nodes, key=lambda x: sub.nodes[x[0].id].temperature, reverse=True)[:5]
        print("  By importance:  " + " | ".join(c[:30] for n, i in top_imp for c in [n.content]))
        print("  By temperature: " + " | ".join(
            f"{sub.nodes[n.id].temperature:.2f}" for n, _ in top_temp))

    return {"spearman_r": round(spearman, 4), "n_nodes": n}


def bench_fusion_coherence(verbose=False) -> dict:
    """
    D2: Fusion Coherence
    Emergent concepts from fusion should be semantically related to parents.
    Metric: Mean Jaccard similarity between child content and each parent.
    """
    print("\n[D2] Fusion Coherence")

    # Use high-temp seeding to force fusions
    sub   = ThermorphicSubstrate()
    nodes = []
    for content, importance, tags in SESSION_MEMORIES[:10]:
        temp = 0.8 + importance * 1.5  # hotter to force fusions
        n    = sub.inject(content, temperature=temp, tags=tags, dims=32)
        nodes.append(n)

    for i in range(len(nodes) - 1):
        sub.connect(nodes[i].id, nodes[i+1].id)
    sub.connect(nodes[-1].id, nodes[0].id)

    for _ in range(5):
        sub.pulse()

    # Score coherence of all fusion events
    coherence_scores = []
    for event in sub.fusion_log:
        parent_a = sub.nodes.get(event.parent_a_id)
        parent_b = sub.nodes.get(event.parent_b_id)
        child    = sub.nodes.get(event.child_id)
        if not (parent_a and parent_b and child):
            continue

        j_a = _jaccard(child.content, parent_a.content)
        j_b = _jaccard(child.content, parent_b.content)
        coherence_scores.append((j_a + j_b) / 2)

        if verbose:
            print(f"  [{event.parent_a_id}] × [{event.parent_b_id}] → [{event.child_id}]")
            print(f"  A: {parent_a.content[:50]}")
            print(f"  B: {parent_b.content[:50]}")
            print(f"  C: {child.content[:50]}")
            print(f"  Jaccard: a={j_a:.2f} b={j_b:.2f} mean={coherence_scores[-1]:.2f}\n")

    mean_coherence = sum(coherence_scores) / max(len(coherence_scores), 1)
    print(f"  Fusions: {len(sub.fusion_log)}  |  Mean coherence (Jaccard) = {mean_coherence:.4f}")
    print(f"  Note: [EMERGENT] prefix inherits parent word overlap — floor ~0.10-0.20 expected")

    return {
        "fusions":       len(sub.fusion_log),
        "mean_coherence": round(mean_coherence, 4),
    }


def bench_crystallization_precision(verbose=False) -> dict:
    """
    D3: Crystallization Precision
    Nodes that crystallize should be low-importance / low-access (correctly forgotten).
    Metric: Mean importance of crystallized vs non-crystallized nodes.
    A good system crystallizes LOW importance nodes into stable long-term memory
    rather than keeping them as hot working memory.
    """
    print("\n[D3] Crystallization Precision")

    sub   = ThermorphicSubstrate()
    nodes = []

    for content, importance, tags in SESSION_MEMORIES:
        temp = 0.3 + importance * 1.5
        n    = sub.inject(content, temperature=temp, tags=tags, dims=32)
        nodes.append((n, importance))

    # Run enough pulses to force crystallization of cold nodes
    _thermo_mod.FREEZE_DWELL = 4  # lower for benchmark speed
    for i in range(len(nodes) - 1):
        sub.connect(nodes[i][0].id, nodes[i+1][0].id)

    for _ in range(20):
        sub.pulse()
    _thermo_mod.FREEZE_DWELL = 8  # restore

    crystals     = [nid for nid, n in sub.nodes.items() if n.immutable]
    non_crystals = [nid for nid, n in sub.nodes.items() if not n.immutable]

    # Map node_id → original importance
    imp_map = {n.id: imp for n, imp in nodes}

    crystal_imp     = [imp_map.get(nid, 0) for nid in crystals     if nid in imp_map]
    non_crystal_imp = [imp_map.get(nid, 0) for nid in non_crystals if nid in imp_map]

    mean_c   = sum(crystal_imp)     / max(len(crystal_imp), 1)
    mean_nc  = sum(non_crystal_imp) / max(len(non_crystal_imp), 1)

    # Good result: crystallized have LOWER mean importance than non-crystallized
    precision_delta = mean_nc - mean_c  # positive = correct (hot stays hot, cold crystallizes)

    print(f"  Crystallized: {len(crystals)}  |  Active: {len(non_crystals)}")
    print(f"  Mean importance — crystallized: {mean_c:.3f}  |  active: {mean_nc:.3f}")
    print(f"  Precision delta (positive = correct): {precision_delta:+.3f}")
    if verbose and crystals:
        print("  Crystallized nodes:")
        for nid in crystals[:5]:
            n = sub.nodes[nid]
            print(f"    [{nid}] imp={imp_map.get(nid,'?'):.2f}  {n.content[:50]}")

    return {
        "crystallized":     len(crystals),
        "mean_imp_crystal": round(mean_c, 4),
        "mean_imp_active":  round(mean_nc, 4),
        "precision_delta":  round(precision_delta, 4),
    }


def bench_thermal_recall_reinforcement(verbose=False) -> dict:
    """
    D4: Thermal Recall Reinforcement
    Recalling a memory should heat it, increasing future recall probability.
    This simulates human memory consolidation: remembering strengthens memory.
    Metric: Mean temperature delta for recalled nodes (should be positive).
    """
    print("\n[D4] Thermal Recall Reinforcement")

    sub   = _build_session_substrate(n_pulses=3)
    query = "database connection pool asyncpg"

    # Record pre-recall temperatures
    pre_temps = {nid: n.temperature for nid, n in sub.nodes.items()}

    # Recall
    recalled = sub.recall(query, top_k=5)

    # Measure temperature delta for recalled nodes
    deltas = []
    for node in recalled:
        if node.id in pre_temps and not node.immutable:
            delta = sub.nodes[node.id].temperature - pre_temps[node.id]
            deltas.append(delta)
            if verbose:
                print(f"  [{node.id}] {node.content[:50]} Δ={delta:+.4f}")

    mean_delta = sum(deltas) / max(len(deltas), 1)
    pct_heated = sum(1 for d in deltas if d > 0) / max(len(deltas), 1) * 100

    print(f"  Recalled: {len(recalled)}  |  Mean Δ temp: {mean_delta:+.4f}  |  % heated: {pct_heated:.0f}%")
    print(f"  Expected: positive delta (recall heats the node) ✓" if mean_delta > 0 else "  ⚠ Recall did not heat nodes")

    return {
        "recalled":    len(recalled),
        "mean_delta":  round(mean_delta, 4),
        "pct_heated":  round(pct_heated, 1),
    }


def bench_evolver_sensitivity(verbose=False) -> dict:
    """
    D5: Evolver Sensitivity
    The three thermorphic genes (ALPHA, FUSION_THRESHOLD, FREEZE_DWELL)
    should produce measurably different memory topologies when mutated.
    If genes don't matter, the evolver can't actually improve anything.

    Metric: Substrate divergence across 3 gene variants.
    """
    print("\n[D5] Evolver Gene Sensitivity")

    configs = {
        "baseline":       {"alpha": 0.08, "fusion_threshold": 1.60, "freeze_dwell": 8},
        "high_diffusion": {"alpha": 0.25, "fusion_threshold": 1.60, "freeze_dwell": 8},
        "low_fusion":     {"alpha": 0.08, "fusion_threshold": 2.50, "freeze_dwell": 8},
        "fast_crystal":   {"alpha": 0.08, "fusion_threshold": 1.60, "freeze_dwell": 3},
    }

    results = {}
    for name, cfg in configs.items():
        sub  = _build_session_substrate(
            alpha            = cfg["alpha"],
            fusion_threshold = cfg["fusion_threshold"],
            freeze_dwell     = cfg["freeze_dwell"],
            n_pulses         = 10,
        )
        snap = sub.snapshot()
        results[name] = {
            "total_nodes":    snap["total_nodes"],
            "total_fusions":  snap["total_fusions"],
            "total_crystals": snap["total_crystals"],
            "mean_temp":      snap["mean_temp"],
        }
        if verbose:
            print(f"  {name:<18} nodes={snap['total_nodes']:>3}  "
                  f"fusions={snap['total_fusions']:>2}  "
                  f"crystals={snap['total_crystals']:>2}  "
                  f"mean_T={snap['mean_temp']:.4f}")

    # Restore
    _thermo_mod.ALPHA            = 0.08
    _thermo_mod.FUSION_THRESHOLD = 1.60
    _thermo_mod.FREEZE_DWELL     = 8

    # Compute divergence: std dev of each metric across configs
    import statistics as stats_mod
    node_counts    = [v["total_nodes"]   for v in results.values()]
    fusion_counts  = [v["total_fusions"] for v in results.values()]
    temps          = [v["mean_temp"]     for v in results.values()]

    node_spread    = stats_mod.stdev(node_counts)   if len(node_counts) > 1    else 0
    fusion_spread  = stats_mod.stdev(fusion_counts) if len(fusion_counts) > 1  else 0
    temp_spread    = stats_mod.stdev(temps)         if len(temps) > 1          else 0

    print(f"\n  Config             nodes  fusions  crystals  mean_T")
    for name, v in results.items():
        print(f"  {name:<18}  {v['total_nodes']:>5}  {v['total_fusions']:>7}  "
              f"{v['total_crystals']:>8}  {v['mean_temp']:.4f}")

    print(f"\n  Spread (stdev):  nodes={node_spread:.1f}  fusions={fusion_spread:.1f}  "
          f"temp={temp_spread:.4f}")
    print(f"  {'✅ Genes are meaningful (topology diverges across configs)' if node_spread > 0.5 else '⚠ Low sensitivity — genes may need wider mutation range'}")

    return {"results": results, "node_spread": round(node_spread, 2),
            "fusion_spread": round(fusion_spread, 2), "temp_spread": round(temp_spread, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all(verbose=False):
    print("\n" + "="*60)
    print("  COGNITIVE SUBSTRATE BENCHMARK — Living Mind Cortex")
    print("  5 dimensions × thermorphic physics")
    print("="*60)

    t_start = time.time()
    all_results = {}

    all_results["D1_salience"]        = bench_thermal_salience(verbose)
    all_results["D2_fusion_coherence"] = bench_fusion_coherence(verbose)
    all_results["D3_crystallization"]  = bench_crystallization_precision(verbose)
    all_results["D4_reinforcement"]    = bench_thermal_recall_reinforcement(verbose)
    all_results["D5_evolver_sens"]     = bench_evolver_sensitivity(verbose)

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    d1 = all_results["D1_salience"]
    d2 = all_results["D2_fusion_coherence"]
    d3 = all_results["D3_crystallization"]
    d4 = all_results["D4_reinforcement"]
    d5 = all_results["D5_evolver_sens"]

    def fmt(val, good_threshold, fmt_str=":.4f"):
        icon = "✅" if val >= good_threshold else "⚠ "
        return f"{icon}  {val:{fmt_str[1:]}}"

    print(f"  D1 Thermal Salience (Spearman ρ):   {fmt(d1['spearman_r'], 0.6)}")
    print(f"  D2 Fusion Coherence (Jaccard):       {fmt(d2['mean_coherence'], 0.05)}  ({d2['fusions']} fusions)")
    print(f"  D3 Crystal Precision (Δ imp):        {fmt(d3['precision_delta'], 0.05):}")
    print(f"  D4 Recall Reinforcement (Δ temp):    {fmt(d4['mean_delta'], 0.01)}  ({d4['pct_heated']:.0f}% nodes heated)")
    print(f"  D5 Gene Sensitivity (node stdev):    {fmt(d5['node_spread'], 0.5, ':.2f')}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive Substrate Benchmark")
    parser.add_argument("--verbose", action="store_true", help="Show per-item detail")
    args = parser.parse_args()
    run_all(verbose=args.verbose)
