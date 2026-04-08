"""
LongMemEval-50 Benchmark — Living Mind Cortex
==============================================
Tests the thermorphic substrate's retrieval on the first 50 questions
from LongMemEval_S (ICLR 2025 benchmark, xiaowu0162/LongMemEval).

Metric:  R@k  (Recall at k)
         = fraction of questions where the correct answer session
           appears in the top-k recalled nodes.

This is a RETRIEVAL benchmark. We are not optimized for this —
the Cortex is a cognitive substrate, not a vector DB.
We run it anyway for honest comparison with MemPalace (96.6% R@5).

Usage:
    python3 benchmarks/longmemeval_runner.py
    python3 benchmarks/longmemeval_runner.py --n 50
    python3 benchmarks/longmemeval_runner.py --n 50 --k 5
"""

import sys
import os
import re
import time
import random
import argparse
from collections import defaultdict

sys.path.insert(0, ".")  # run from project root

from cortex.thermorphic import ThermorphicSubstrate  # fresh substrate per question


# ── Config ────────────────────────────────────────────────────────────────────

DATASET_ID  = "xiaowu0162/longmemeval-cleaned"
SPLIT       = "longmemeval_s"   # shorter haystack (~115k tokens, ~40 sessions/q)
DEFAULT_N   = 50
DEFAULT_K   = 5
INJECT_TEMP_BASE  = 0.6     # base temperature for all session turns
INJECT_TEMP_BONUS = 0.3     # bonus for turns with has_answer=True (cheating check omitted)
DIMS              = 32      # HRR dims — small for speed in benchmark


# ── Data loading ──────────────────────────────────────────────────────────────

CACHED_S_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval-cleaned"
    "/snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json"
)


def load_questions(n: int):
    """Load first n questions from the cached LongMemEval_S JSON."""
    import json as _json
    print(f"[LME] Loading {n} questions from cache...")
    try:
        with open(CACHED_S_PATH) as f:
            data = _json.load(f)
        # Dataset may be a list or dict-of-list
        if isinstance(data, list):
            questions = data[:n]
        elif isinstance(data, dict) and "data" in data:
            questions = data["data"][:n]
        else:
            # Try first value
            questions = list(data.values())[0][:n]
        print(f"[LME] Loaded {len(questions)} questions (total: {len(data) if isinstance(data, list) else '?'})")
        return questions
    except FileNotFoundError:
        print(f"[LME] Cache miss — downloading via HuggingFace (this takes ~30s)")
        try:
            from datasets import load_dataset
            ds = load_dataset(DATASET_ID, data_files={"s": "longmemeval_s_cleaned.json"}, split="s")
            questions = list(ds.select(range(min(n, len(ds)))))
            print(f"[LME] Loaded {len(questions)} questions")
            return questions
        except Exception as e:
            print(f"[LME] ❌ Load failed: {e}")
            sys.exit(1)



# ── Substrate injection ────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip whitespace and truncate to 200 chars for substrate injection."""
    return re.sub(r'\s+', ' ', text).strip()[:200]


def build_substrate_for_question(q: dict) -> tuple[ThermorphicSubstrate, dict[str, str]]:
    """
    Inject all haystack sessions into a fresh substrate.
    Returns:
        substrate       — the loaded substrate
        node_to_session — maps thermorphic node ID → session_id
    """
    sub             = ThermorphicSubstrate()
    node_to_session = {}                       # thermo_node_id → session_id
    session_nodes   = defaultdict(list)        # session_id → [node_ids]

    haystack      = q.get("haystack_sessions", [])       # list of list-of-turns
    session_ids   = q.get("haystack_session_ids", [])    # parallel session ID list

    for idx, turns in enumerate(haystack):
        session_id   = session_ids[idx] if idx < len(session_ids) else f"s{idx}"
        prev_node_id = None

        for turn in turns:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if not content.strip():
                continue

            # Temperature: assistant turns slightly hotter (more informative)
            temp = INJECT_TEMP_BASE + (0.1 if role == "assistant" else 0.0)

            node = sub.inject(
                content     = _clean(f"[{role}] {content}"),
                temperature = temp,
                tags        = [session_id, role, "session"],
                dims        = DIMS,
            )
            node_to_session[node.id] = session_id
            session_nodes[session_id].append(node.id)

            # Wire to previous turn in same session (temporal chain)
            if prev_node_id:
                sub.connect(prev_node_id, node.id)
            prev_node_id = node.id

        # Circular thermal path within session (head ↔ tail)
        nids = session_nodes[session_id]
        if len(nids) > 2:
            sub.connect(nids[-1], nids[0])

    return sub, node_to_session



# ── Recall + scoring ──────────────────────────────────────────────────────────

def run_question(q: dict, k: int) -> dict:
    """
    Run a single LongMemEval question through the thermorphic substrate.
    Returns hit@1, hit@k, and metadata.
    """
    question_text    = q.get("question", "")
    answer_text      = q.get("answer", "")
    answer_sessions  = set(q.get("answer_session_ids", []))
    question_type    = q.get("question_type", "unknown")
    is_abstention    = str(q.get("question_id", "")).endswith("_abs")

    # Abstention questions — skip (our system doesn't model "I don't know" yet)
    if is_abstention:
        return {"skipped": True, "reason": "abstention", "question_type": question_type}

    # No answer sessions to check against
    if not answer_sessions:
        return {"skipped": True, "reason": "no_answer_sessions", "question_type": question_type}

    # Build substrate
    sub, node_to_session = build_substrate_for_question(q)

    # Run a couple of thermal pulses to let heat diffuse
    for _ in range(3):
        sub.pulse()

    # Recall top-k against the question
    recalled = sub.recall(question_text, top_k=k)

    # Map recalled nodes back to session IDs
    recalled_sessions = []
    seen = set()
    for node in recalled:
        sid = node_to_session.get(node.id, "unknown")
        if sid not in seen:
            recalled_sessions.append(sid)
            seen.add(sid)

    # Score: did any answer session appear in top-k?
    hit_k = any(s in answer_sessions for s in recalled_sessions[:k])
    hit_1 = bool(recalled_sessions) and recalled_sessions[0] in answer_sessions

    return {
        "skipped":          False,
        "hit_1":            hit_1,
        "hit_k":            hit_k,
        "question_type":    question_type,
        "answer_sessions":  list(answer_sessions),
        "recalled_sessions": recalled_sessions[:k],
        "substrate_nodes":  len(sub.nodes),
        "question":         question_text[:80],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(n: int = DEFAULT_N, k: int = DEFAULT_K):
    questions = load_questions(n)

    print(f"\n{'='*60}")
    print(f"  LongMemEval-{n} × Thermorphic Substrate")
    print(f"  Metric: R@1 and R@{k}  |  {len(questions)} questions")
    print(f"{'='*60}\n")

    results       = []
    hits_1        = 0
    hits_k        = 0
    skipped       = 0
    by_type       = defaultdict(lambda: {"total": 0, "hit_k": 0})
    t_start       = time.time()

    for i, q in enumerate(questions):
        qid = q.get("question_id", f"q{i}")
        res = run_question(q, k)

        if res["skipped"]:
            skipped += 1
            print(f"  [{i+1:>3}/{n}] SKIP  ({res['reason']}) {qid}")
            continue

        results.append(res)
        qt = res["question_type"]
        by_type[qt]["total"] += 1

        marker_1 = "✅" if res["hit_1"] else "❌"
        marker_k = "✅" if res["hit_k"] else "❌"

        if res["hit_1"]:
            hits_1 += 1
        if res["hit_k"]:
            hits_k += 1
            by_type[qt]["hit_k"] += 1

        print(
            f"  [{i+1:>3}/{n}] R@1={marker_1} R@{k}={marker_k} "
            f"nodes={res['substrate_nodes']:>4}  {res['question'][:50]}"
        )

    elapsed   = time.time() - t_start
    evaluated = len(results)

    r_at_1 = hits_1 / max(evaluated, 1)
    r_at_k = hits_k / max(evaluated, 1)

    print(f"\n{'='*60}")
    print(f"  RESULTS — LongMemEval-{n} (thermorphic substrate)")
    print(f"{'='*60}")
    print(f"  Evaluated:  {evaluated}  |  Skipped: {skipped} (abstention/no-answer)")
    print(f"  R@1:        {r_at_1*100:.1f}%")
    print(f"  R@{k}:        {r_at_k*100:.1f}%")
    print(f"  Time:       {elapsed:.1f}s  ({elapsed/max(evaluated,1):.2f}s/q)")
    print()
    print(f"  By question type (R@{k}):")
    for qt, d in sorted(by_type.items(), key=lambda x: -x[1]["hit_k"]):
        pct = d["hit_k"] / max(d["total"], 1) * 100
        print(f"    {qt:<30} {d['hit_k']}/{d['total']} ({pct:.0f}%)")

    print(f"\n  MemPalace baseline (raw verbatim mode): R@5 = 96.6%")
    print(f"  Living Mind Cortex (thermorphic):       R@{k} = {r_at_k*100:.1f}%")
    print(f"\n  Note: Cortex is not a retrieval system — these architectures")
    print(f"  solve different problems. See cognitive_substrate_bench.py")
    print(f"  for the benchmark that measures what the Cortex actually does.")
    print(f"{'='*60}\n")

    return {
        "r_at_1":    round(r_at_1, 4),
        f"r_at_{k}": round(r_at_k, 4),
        "evaluated": evaluated,
        "skipped":   skipped,
        "elapsed_s": round(elapsed, 1),
        "by_type":   dict(by_type),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval vs Thermorphic Substrate")
    parser.add_argument("--n",  type=int, default=DEFAULT_N, help="Number of questions")
    parser.add_argument("--k",  type=int, default=DEFAULT_K, help="Recall@k")
    args = parser.parse_args()
    run_benchmark(n=args.n, k=args.k)
