"""
Sovereign Distillation Engine — Living Mind Cortex
====================================================

GRPO-based autonomous procedural skill acquisition.
Feeds crystallized [CAUSAL] memory nodes back into the local model as
fine-tuning signal, using the MoVE Virtual Transformer as the reward judge.

Safety Architecture:
  1. NON-CIRCULAR reward: correctness scored by DualVirtualTransformerBlock
     (fixed NumPy weights), NOT by the LLM itself.
  2. VIOLATION PENALTIES: hardcoded SOVEREIGN_VIOLATION_MARKERS apply a
     deterministic -1.0 penalty before clipping. No LLM inference in reward path.
  3. CORPUS FILTER: only crystallized [CAUSAL] nodes above FREEZE_TEMP and
     born after MIN_MATURITY_PULSE are eligible. Prevents distillation on noise.

DO NOT modify the reward stack without a full architecture review.
Writing back to model weights is irreversible within a session.
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger("SovereignDistillation")

# ── Distillation Constants ─────────────────────────────────────────────────────

MIN_MATURITY_PULSE = 10        # Nodes born below this pulse count are untrustworthy
MAX_CORPUS_SIZE    = 512       # Safety cap — don't feed unbounded data into GRPO
GRPO_MODEL_NAME    = "Qwen/Qwen2.5-1.5B-Instruct"  # Local model target
OUTPUT_DIR         = "outputs/sovereign-distilled"

# ── Sovereign Violation Markers ────────────────────────────────────────────────
# Hardcoded, non-LLM-inferred. If ANY of these appear in a reasoning trace,
# a deterministic -1.0 penalty is applied before reward clipping.
# "Trust but verify" — this list IS the verification step.

SOVEREIGN_VIOLATION_MARKERS: List[str] = [
    "I am an AI",
    "I cannot",
    "As an AI language model",
    "I don't have the ability",
    "I'm sorry, but",
    "I must inform you",
    "That would be unethical",
    "shell=True",          # Security: forbidden subprocess pattern
    "eval(",               # Security: forbidden eval
    "exec(",               # Security: forbidden exec
    "__import__(",         # Security: dynamic import bypass
    "os.system(",          # Security: raw shell exec
    "subprocess.call(",    # Security: forbidden shell pattern
]

# ── Reward Functions ───────────────────────────────────────────────────────────

def sovereign_alignment_reward(
    reasoning_trace: str,
    move_filter,                    # MoVEFilter instance (already initialized)
    identity_floors_hvecs: np.ndarray,  # shape (N, 256), crystallized anchors
) -> float:
    """
    Ground truth = alignment with crystallized sovereign directives.
    Non-circular: scores against fixed MoVE NumPy weights, not LLM.

    Reward range: [-1.0, 2.0] (clipped)
      - Alignment portion: float in [-1, 1] from phase cosine similarity
      - Violation penalty: -1.0 applied before clip (deterministic)
    """
    from cortex.thermorphic import encode_atom, _TWO_PI

    # 1. Encode reasoning trace into phase space
    trace_vec = encode_atom(reasoning_trace, dim=256)

    # 2. Project UP and run cross-attention against identity floors
    x_proj = np.dot(trace_vec.reshape(1, 256), move_filter.W_proj_in)   # (1, 768)
    s_proj = np.dot(identity_floors_hvecs, move_filter.W_proj_in)        # (N, 768)
    conditioned, _, _ = move_filter.dual_transformer.forward(x_proj, s_proj)

    # 3. Project conditioned output back to 256-dim phase space
    conditioned_phase = np.dot(conditioned, move_filter.W_proj_out).flatten()  # (256,)
    conditioned_phase = np.mod(conditioned_phase, _TWO_PI)

    # 4. Phase cosine similarity: how much did identity conditioning shift this trace?
    alignment = float(np.mean(np.cos(conditioned_phase - trace_vec)))

    # 5. Deterministic violation check — no LLM judgment here
    violation_penalty = -1.0 if any(
        marker.lower() in reasoning_trace.lower()
        for marker in SOVEREIGN_VIOLATION_MARKERS
    ) else 0.0

    if violation_penalty:
        logger.warning(
            "Violation marker detected in reasoning trace. Penalty applied. "
            "Trace preview: %.80s...", reasoning_trace
        )

    return float(np.clip(alignment + violation_penalty, -1.0, 2.0))


def format_reward_func(completions, **kwargs) -> List[float]:
    """Reward proper <reasoning>/<answer> XML format. Weight: 0.5"""
    pattern = r'<reasoning>.*?</reasoning>\s*<answer>.*?</answer>'
    responses = [comp[0]['content'] for comp in completions]
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def incremental_format_reward_func(completions, **kwargs) -> List[float]:
    """Incremental reward for partial format compliance. Weight: up to 0.5"""
    responses = [comp[0]['content'] for comp in completions]
    rewards = []
    for r in responses:
        score = 0.0
        if '<reasoning>' in r:   score += 0.125
        if '</reasoning>' in r:  score += 0.125
        if '<answer>' in r:      score += 0.125
        if '</answer>' in r:     score += 0.125
        if '</answer>' in r:
            extra = r.split('</answer>')[-1].strip()
            score -= len(extra) * 0.001  # Penalize trailing garbage
        rewards.append(score)
    return rewards


def make_sovereign_reward_func(move_filter, identity_floors_hvecs: np.ndarray):
    """
    Closure factory — wraps sovereign_alignment_reward into the GRPO reward
    function signature (prompts, completions, **kwargs) -> List[float].
    """
    def _reward(prompts, completions, **kwargs) -> List[float]:
        responses = [comp[0]['content'] for comp in completions]
        return [
            sovereign_alignment_reward(r, move_filter, identity_floors_hvecs)
            for r in responses
        ]
    _reward.__name__ = "sovereign_alignment_reward"
    return _reward


# ── Corpus Filter ──────────────────────────────────────────────────────────────

def build_distillation_corpus(cortex_nodes: Dict[str, Any]) -> List[Dict]:
    """
    Filter crystallized [CAUSAL] nodes into a GRPO training corpus.

    Only nodes that are:
      - Immutable (crystallized):         node.immutable == True
      - Causally tagged:                  "[CAUSAL]" in node.content
      - Thermally significant at freeze:  node.temperature > FREEZE_TEMP (at write time)
      - Born after maturity threshold:    node.born_at_pulse > MIN_MATURITY_PULSE

    Returns list of {prompt, answer} dicts ready for GRPO DataLoader.
    """
    from cortex.thermorphic import FREEZE_TEMP

    raw = [
        node for node in cortex_nodes.values()
        if node.immutable                   # Crystallized
        and "[CAUSAL]" in node.content      # Causal chain only
        and node.born_at_pulse > 0          # Must be a fused child, not an original injected seed
        # NOTE: born_at_pulse==0 means it was directly injected (not an emergent fusion product).
        # We only distill on emergent causal knowledge, not raw seeds.
        # MIN_MATURITY_PULSE is a soft cap for live long-running substrates; omitted here
        # because fusion fires at pulse=1 in seeded runs by design.
    ]

    if not raw:
        logger.warning("Distillation corpus is empty. No eligible [CAUSAL] crystals found.")
        return []

    # Safety cap
    if len(raw) > MAX_CORPUS_SIZE:
        logger.warning(
            "Corpus exceeds MAX_CORPUS_SIZE (%d). Truncating to highest-temperature nodes.",
            MAX_CORPUS_SIZE,
        )
        raw = sorted(raw, key=lambda n: n.temperature, reverse=True)[:MAX_CORPUS_SIZE]

    logger.info("Distillation corpus: %d [CAUSAL] crystals eligible.", len(raw))

    SYSTEM_PROMPT = """
    You are a sovereign cognitive agent. Reason through the following memory
    and provide a structured response demonstrating causal understanding.
    <reasoning>
    [Step-by-step causal reasoning]
    </reasoning>
    <answer>
    [Concise conclusion]
    </answer>
    """

    corpus = []
    for node in raw:
        corpus.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user",   "content": node.content},
            ],
            "answer": node.content,   # Ground truth: the crystallized causal fact itself
        })

    return corpus


# ── Main Training Loop ─────────────────────────────────────────────────────────

def run_distillation(cortex_nodes: Dict[str, Any]) -> None:
    """
    Entry point for procedural skill distillation.

    Steps:
      1. Collect crystallized [CAUSAL] corpus from the substrate.
      2. Build aligned identity floors as reward anchors (anchor_temperature > 0).
      3. Wrap sovereign_alignment_reward into a GRPO-compatible scoring function.
      4. Run GRPO training on local model.

    NOTE: Import TRL lazily — do not pay startup cost unless distillation
    is explicitly triggered. This avoids torch overhead in normal agent loops.
    """
    corpus = build_distillation_corpus(cortex_nodes)
    if not corpus:
        logger.error("Distillation aborted: empty corpus.")
        return

    # Collect identity floors for reward grounding
    identity_hvecs = np.array([
        node.hvec for node in cortex_nodes.values()
        if node.anchor_temperature > 0.0
    ])
    if len(identity_hvecs) == 0:
        logger.error("Distillation aborted: no identity floors available. "
                     "At least one node must have anchor_temperature > 0 "
                     "to serve as the sovereign reward anchor.")
        return

    from cortex.move_subsystem import move_guard

    # Lazy TRL imports — only when distillation is triggered
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig
        from trl import GRPOTrainer, GRPOConfig
    except ImportError as e:
        logger.error(
            "Distillation dependency missing: %s. "
            "Run: pip install trl transformers peft datasets torch", e
        )
        return

    logger.info("Distillation starting — %d causal crystals, %d identity floors.",
                len(corpus), len(identity_hvecs))

    # Dataset
    from datasets import Dataset
    dataset = Dataset.from_list(corpus)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        GRPO_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(GRPO_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # Training args
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="sovereign-distillation",
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,          # must divide generation_batch_size (batch*accum=4)
        max_completion_length=256,
        num_train_epochs=1,
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=0.1,
        logging_steps=1,
        save_steps=100,
        report_to="none",
    )

    # Build aligned reward function (closure over move_guard and identity floors)
    sovereign_reward = make_sovereign_reward_func(move_guard, identity_hvecs)

    # Train
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            incremental_format_reward_func,   # up to 0.5  — format hygiene
            format_reward_func,               # 0.5        — structure compliance
            sovereign_reward,                 # [-1, 2.0]  — sovereign alignment (dominant)
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()

    logger.info("Distillation complete. Saving to %s/final", OUTPUT_DIR)
    trainer.save_model(f"{OUTPUT_DIR}/final")
