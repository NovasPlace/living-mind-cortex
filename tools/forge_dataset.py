"""
Sovereign LoRA Dataset Forger — tools/forge_dataset.py

Synthesizes a pristine code_expert JSONL dataset via a two-pass
Reflective Validator loop:
  1. Generator: Llama-3-8B-Instruct produces a raw trace at temp=0.6
  2. Compiler:  Same model re-passes the trace with a strict-compiler
                persona at temp=0.1 to prune non-performant code.
  3. Gate:      Pure AST validation rejects anything that fails static
                constraints BEFORE it touches the JSONL.

Three-tier distribution:
  40% Semantic Anchor Pairings  (encode_atom aligned)
  40% Polyglot Pipeline Traces  (zero-copy Rust/TS/Python gRPC/WS)
  20% Thermal Dampening         (noisy input → corrected output)

Usage:
  python tools/forge_dataset.py --count 500 --output data/code_expert.jsonl
"""

import ast
import asyncio
import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ── Config ────────────────────────────────────────────────────────────────────

VLLM_BASE          = "http://localhost:8001/v1"
BASE_MODEL         = "meta-llama/Meta-Llama-3-8B-Instruct"
GENERATOR_TEMP     = 0.6
COMPILER_TEMP      = 0.1
MAX_TOKENS_GEN     = 1024
MAX_TOKENS_COMPILE = 1536
MAX_RETRIES        = 3   # reflective validator retries before dropping a sample

# ── Import Whitelist ──────────────────────────────────────────────────────────
# Only these root modules are allowed in any generated code block.
# Anything outside this set trips the hallucination gate.

ALLOWED_IMPORTS = {
    # Python stdlib
    "asyncio", "abc", "ast", "collections", "contextlib", "copy",
    "dataclasses", "enum", "functools", "hashlib", "io", "itertools",
    "json", "logging", "math", "os", "pathlib", "queue", "re",
    "signal", "struct", "sys", "threading", "time", "typing",
    "typing_extensions", "uuid", "weakref",
    # Vetted packages
    "fastapi", "pydantic", "uvicorn", "starlette",
    "asyncpg", "sqlalchemy", "aiohttp", "httpx",
    "numpy", "torch", "transformers", "unsloth",
    "grpc", "protobuf",
    # Rust / FFI bridges (pyo3 surface)
    "ctypes", "cffi", "mmap",
}


# ── AST Strict-Compiler Gates ─────────────────────────────────────────────────

class CompilerViolation(Exception):
    pass


def _extract_code_blocks(text: str) -> list[str]:
    """Pull every fenced ```python ... ``` block from a generation."""
    blocks = []
    lines = text.split("\n")
    inside = False
    buf = []
    for line in lines:
        if line.strip().startswith("```python") or line.strip() == "```py":
            inside = True
            buf = []
        elif line.strip() == "```" and inside:
            inside = False
            blocks.append("\n".join(buf))
        elif inside:
            buf.append(line)
    # If no fenced blocks exist, treat entire text as one block
    if not blocks:
        blocks = [text]
    return blocks


def gate_syntax(code: str) -> None:
    """Gate 1: Hard Python syntax check. Raises CompilerViolation on failure."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise CompilerViolation(f"SYNTAX: {e}")


def gate_imports(code: str) -> None:
    """Gate 2: Reject hallucinated library imports not in the whitelist."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # Syntax gate already fired
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise CompilerViolation(f"IMPORT_HALLUCINATION: '{alias.name}' not in whitelist")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise CompilerViolation(f"IMPORT_HALLUCINATION: from '{node.module}' not in whitelist")


def gate_type_annotations(code: str) -> None:
    """Gate 3: Every function must declare a return type annotation."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is None:
                raise CompilerViolation(
                    f"MISSING_RETURN_TYPE: function '{node.name}' has no -> annotation"
                )


def gate_copy_contamination(code: str, polyglot: bool = False) -> None:
    """
    Gate 4 (polyglot traces only): Reject zero-copy violations.
    Flags .copy(), deepcopy(), list(arr) on buffer-like names.
    """
    if not polyglot:
        return
    COPY_SIGNALS = [".copy()", "deepcopy(", "list(buf", "list(arr", "bytearray(buf"]
    for sig in COPY_SIGNALS:
        if sig in code:
            raise CompilerViolation(f"COPY_CONTAMINATION: zero-copy violation '{sig}' detected")


def run_compiler_gates(code: str, polyglot: bool = False) -> None:
    """Run all AST gates in sequence. Raises CompilerViolation on first failure."""
    gate_syntax(code)
    gate_imports(code)
    gate_type_annotations(code)
    gate_copy_contamination(code, polyglot=polyglot)


# ── Prompt Templates ─────────────────────────────────────────────────────────

SEMANTIC_ANCHOR_TOPICS = [
    ("asynchronous database connection pool", "asyncpg"),
    ("holographic vector superposition retrieval", "numpy"),
    ("FastAPI lifespan context manager startup shutdown", "fastapi"),
    ("thermodynamic memory decay curve implementation", "python"),
    ("zero-trust token validation middleware", "fastapi"),
    ("LRU cache with TTL expiration asyncio", "asyncio"),
    ("circular convolution holographic binding numpy", "numpy"),
    ("PostgreSQL LISTEN NOTIFY pub sub asyncpg", "asyncpg"),
    ("gradient accumulation bfloat16 training loop", "torch"),
    ("LoRA adapter hot swapping inference vllm", "transformers"),
]

POLYGLOT_TOPICS = [
    ("Rust tokio async server exposing a gRPC endpoint consumed by a Python asyncio client using zero-copy bytes buffer", "rust+python"),
    ("TypeScript WebSocket client receiving binary frames from a Python FastAPI server using ArrayBuffer and no JSON serialization", "ts+python"),
    ("Python ctypes FFI calling a C shared library function that operates on a pre-allocated mmap buffer", "python+c"),
    ("Go gRPC server streaming large byte payloads to a Python client using receive-into-preallocated-buffer pattern", "go+python"),
    ("Rust PyO3 extension returning a numpy array view with zero copy to the Python caller", "rust+python"),
    ("C++ ZeroMQ publisher sending packed struct binary frames consumed by Python zmq with struct.unpack no copy", "cpp+python"),
]

DAMPENING_TOPICS = [
    "async function missing await that causes silent race condition",
    "type annotation claiming List[str] but returning Optional[str]",
    "database query inside a hot loop without connection pooling",
    "blocking sleep() in an asyncio coroutine",
    "mutable default argument bug in class __init__",
    "exception swallowing bare except: pass pattern",
    "recursive function without a base case depth guard",
    "global state mutation inside a pure function",
]


def _semantic_anchor_prompt(topic: str, lib: str) -> tuple[str, str]:
    system = (
        "You are code_expert, a hyper-specialized code synthesis engine. "
        "You output only production-grade Python with full type annotations on every function. "
        "No explanations. No markdown prose. Only a single fenced ```python block."
    )
    instruction = (
        f"Implement a complete, production-ready Python module demonstrating: {topic}. "
        f"Use only the {lib} library. "
        "Every function MUST have explicit return type annotations. "
        "No placeholder comments like '# TODO'. Fully implemented."
    )
    return system, instruction


def _polyglot_prompt(description: str, stack: str) -> tuple[str, str]:
    system = (
        "You are code_expert specializing in zero-copy cross-language interoperability. "
        "Output only the implementation. No prose. Use fenced code blocks per language. "
        "NEVER use .copy(), deepcopy(), or list() on buffers. Every Python function needs a return type."
    )
    instruction = (
        f"Implement a complete working example of: {description}. "
        f"Stack: {stack}. "
        "Enforce zero-copy semantics throughout. Use pre-allocated buffers. "
        "Show both sides of the handoff. Real library calls only — no pseudocode."
    )
    return system, instruction


def _dampening_prompt(bug_description: str) -> tuple[str, str]:
    system = (
        "You are a strict compiler. "
        "Given a buggy code pattern, output EXACTLY two fenced Python blocks: "
        "First the BUGGY version, then the CORRECTED version. "
        "Label them # BUGGY and # CORRECTED as inline comments. "
        "All functions must have return type annotations."
    )
    instruction = (
        f"Demonstrate and fix this anti-pattern: {bug_description}. "
        "Show the broken implementation first, then the corrected one. "
        "The correction must be production-safe and fully type-annotated."
    )
    return system, instruction


def _compiler_persona_prompt(raw_code: str) -> tuple[str, str]:
    system = (
        "You are a strict-compiler validator running at temperature 0.1. "
        "Your only job: take the provided code and return a corrected version. "
        "Rules (non-negotiable): "
        "1. Every function must have complete return type annotations. "
        "2. All imports must be from real Python packages — no invented modules. "
        "3. No .copy() or deepcopy() on buffer/array objects. "
        "4. No placeholder comments. Fully implement every stub. "
        "5. Output ONLY a single ```python fenced block. Nothing else."
    )
    instruction = f"Validate and correct this code:\n\n```python\n{raw_code}\n```"
    return system, instruction


# ── HTTP Client ───────────────────────────────────────────────────────────────

async def llm_call(
    client: httpx.AsyncClient,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": BASE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        r = await client.post(f"{VLLM_BASE}/chat/completions", json=payload, timeout=120.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError) as e:
        raise RuntimeError(f"LLM call failed: {e}")


# ── Reflective Validator ──────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    instruction: str
    response: str
    tier: str
    attempts: int


async def reflective_validate(
    client: httpx.AsyncClient,
    system_gen: str,
    instruction: str,
    tier: str,
    polyglot: bool = False,
) -> SynthesisResult | None:
    """
    Two-pass Reflective Validator:
      Pass 1 (Generator): temp=0.6 — creative trace synthesis
      Pass 2 (Compiler):  temp=0.1 — strict-persona pruning
      Gate:               AST validation. Retry up to MAX_RETRIES.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # ── Pass 1: Generator ────────────────────────────────────────
            raw = await llm_call(
                client, system_gen, instruction,
                temperature=GENERATOR_TEMP,
                max_tokens=MAX_TOKENS_GEN,
            )

            # ── Pass 2: Compiler persona ─────────────────────────────────
            compiler_sys, compiler_user = _compiler_persona_prompt(raw)
            compiled = await llm_call(
                client, compiler_sys, compiler_user,
                temperature=COMPILER_TEMP,
                max_tokens=MAX_TOKENS_COMPILE,
            )

            # ── Gate: AST static analysis ────────────────────────────────
            blocks = _extract_code_blocks(compiled)
            # Validate every block — all must pass
            for block in blocks:
                if block.strip():
                    run_compiler_gates(block, polyglot=polyglot)

            print(f"  [✓] {tier} (attempt {attempt}/{MAX_RETRIES})")
            return SynthesisResult(
                instruction=instruction,
                response=compiled,
                tier=tier,
                attempts=attempt,
            )

        except CompilerViolation as cv:
            print(f"  [✗] Gate rejected (attempt {attempt}/{MAX_RETRIES}): {cv}")
        except RuntimeError as re:
            print(f"  [!] LLM error (attempt {attempt}/{MAX_RETRIES}): {re}")
            await asyncio.sleep(2)

    print(f"  [DROPPED] {tier} exhausted {MAX_RETRIES} retries.")
    return None


# ── Dataset orchestrator ──────────────────────────────────────────────────────

async def forge(total: int, output_path: Path) -> None:
    n_anchor   = int(total * 0.40)
    n_polyglot = int(total * 0.40)
    n_dampen   = total - n_anchor - n_polyglot
    
    print(f"\n{'═'*60}")
    print(f"  Sovereign Dataset Forge — target: {total} samples")
    print(f"  Semantic Anchors:  {n_anchor}")
    print(f"  Polyglot Traces:   {n_polyglot}")
    print(f"  Thermal Dampening: {n_dampen}")
    print(f"  Output: {output_path}")
    print(f"{'═'*60}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    dropped = 0

    async with httpx.AsyncClient() as client:
        # Verify vLLM is alive before burning compute
        try:
            await client.get(f"{VLLM_BASE}/models", timeout=5.0)
            print("[+] vLLM endpoint reachable\n")
        except httpx.HTTPError:
            print(f"[FATAL] Cannot reach vLLM at {VLLM_BASE}. Start the inference server first.")
            sys.exit(1)

        with output_path.open("w") as fout:

            # ── Tier 1: Semantic Anchors ──────────────────────────────────
            print(f"── Tier 1: Semantic Anchors ({n_anchor}) ──")
            for i in range(n_anchor):
                topic, lib = random.choice(SEMANTIC_ANCHOR_TOPICS)
                sys_p, inst = _semantic_anchor_prompt(topic, lib)
                result = await reflective_validate(client, sys_p, inst, "anchor")
                if result:
                    record = {
                        "conversations": [
                            {"from": "human",     "value": result.instruction},
                            {"from": "assistant", "value": result.response},
                        ],
                        "meta": {"tier": result.tier, "attempts": result.attempts},
                    }
                    fout.write(json.dumps(record) + "\n")
                    written += 1
                else:
                    dropped += 1

            # ── Tier 2: Polyglot Traces ───────────────────────────────────
            print(f"\n── Tier 2: Polyglot Pipeline Traces ({n_polyglot}) ──")
            for i in range(n_polyglot):
                desc, stack = random.choice(POLYGLOT_TOPICS)
                sys_p, inst  = _polyglot_prompt(desc, stack)
                result = await reflective_validate(client, sys_p, inst, "polyglot", polyglot=True)
                if result:
                    record = {
                        "conversations": [
                            {"from": "human",     "value": result.instruction},
                            {"from": "assistant", "value": result.response},
                        ],
                        "meta": {"tier": result.tier, "attempts": result.attempts},
                    }
                    fout.write(json.dumps(record) + "\n")
                    written += 1
                else:
                    dropped += 1

            # ── Tier 3: Thermal Dampening ─────────────────────────────────
            print(f"\n── Tier 3: Thermal Dampening ({n_dampen}) ──")
            for i in range(n_dampen):
                bug = random.choice(DAMPENING_TOPICS)
                sys_p, inst = _dampening_prompt(bug)
                result = await reflective_validate(client, sys_p, inst, "dampening")
                if result:
                    record = {
                        "conversations": [
                            {"from": "human",     "value": result.instruction},
                            {"from": "assistant", "value": result.response},
                        ],
                        "meta": {"tier": result.tier, "attempts": result.attempts},
                    }
                    fout.write(json.dumps(record) + "\n")
                    written += 1
                else:
                    dropped += 1

    print(f"\n{'═'*60}")
    print(f"  FORGE COMPLETE")
    print(f"  Written: {written} / {total}")
    print(f"  Dropped: {dropped} (gate-rejected or LLM error)")
    print(f"  Yield:   {100 * written / max(total, 1):.1f}%")
    print(f"{'═'*60}\n")


# ── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forge the code_expert LoRA dataset")
    parser.add_argument("--count",  type=int, default=500,
                        help="Total samples to synthesize (default 500)")
    parser.add_argument("--output", type=str, default="data/code_expert.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()

    asyncio.run(forge(args.count, Path(args.output)))
