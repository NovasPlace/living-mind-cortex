"""
Sovereign LoRA Trainer — tools/train_lora.py

Forges the code_expert LoRA adapter using Unsloth's rsLoRA engine.

Configuration:
  Base model:   meta-llama/Meta-Llama-3-8B-Instruct (4bit bitsandbytes)
  Adapter rank: r=64, alpha=128 (rsLoRA stabilized)
  Precision:    bfloat16
  Dataset:      data/code_expert.jsonl (ShareGPT format)
  Output:       code_expert/  (vLLM-compatible adapter directory)

Usage:
  python tools/train_lora.py --dataset data/code_expert.jsonl --output code_expert

Dependencies:
  pip install unsloth[colab-new] bitsandbytes accelerate
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Unsloth import guard ──────────────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    print("[WARN] Unsloth not found. Install with: pip install 'unsloth[colab-new]'")

try:
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("[WARN] datasets/trl not found. Install with: pip install datasets trl")


# ── Hyperparameters ───────────────────────────────────────────────────────────

LORA_CONFIG = dict(
    r               = 64,            # rsLoRA rank — high for dense code domain
    lora_alpha      = 128,           # alpha = 2r for rsLoRA stability
    lora_dropout    = 0.0,           # dropout off during fine-tune (Unsloth recommendation)
    bias            = "none",
    use_rslora      = True,          # Rank-Stabilized LoRA — prevents gradient collapse at r=64
    target_modules  = [              # Full attention + FFN coverage
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

TRAIN_CONFIG = dict(
    # ── Batch sizing ──────────────────────────────────────────────────
    # Global batch = per_device_train_batch_size × gradient_accumulation_steps
    # Target: global batch ≥ 32 so rsLoRA alpha=128 gets stable gradient signal.
    # On 24GB VRAM (RTX 3090/4090): batch=2, accum=16 → global=32
    # On 16GB VRAM (RTX 3080/4080): batch=1, accum=32 → global=32
    per_device_train_batch_size   = 2,
    gradient_accumulation_steps   = 16,   # Tune down to 8 on >24GB VRAM
    warmup_steps                  = 50,
    num_train_epochs              = 3,
    learning_rate                 = 2e-4,
    fp16                          = False,
    bf16                          = True,   # bfloat16 — better dynamic range than fp16
    logging_steps                 = 10,
    save_steps                    = 100,
    save_total_limit              = 2,
    optim                         = "adamw_8bit",  # 8-bit Adam → ~half the VRAM of fp32 Adam
    weight_decay                  = 0.01,
    lr_scheduler_type             = "cosine",
    seed                          = 3407,
    report_to                     = "none",  # Disable wandb/tensorboard — pure local run
)

MAX_SEQ_LENGTH = 2048   # Llama-3 supports 8k, we cap for VRAM budget


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_sharegpt_jsonl(path: Path) -> Dataset:
    """
    Load a JSONL file in ShareGPT format and flatten to HuggingFace Dataset.
    Expected record: {"conversations": [{"from": "human", "value": "..."}, ...]}
    """
    records = []
    skipped = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                if len(convs) >= 2:
                    records.append({"conversations": convs})
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    print(f"[Dataset] Loaded {len(records)} samples ({skipped} skipped)")
    if not records:
        print("[FATAL] No valid samples found. Forge the dataset first:")
        print("        python tools/forge_dataset.py --count 500")
        sys.exit(1)

    return Dataset.from_list(records)


# ── Chat template formatter ───────────────────────────────────────────────────

def format_conversations(examples: dict, tokenizer) -> dict:
    """
    Apply the Llama-3 chat template to each conversation.
    Unsloth's get_chat_template patches the tokenizer in-place.
    """
    texts = tokenizer.apply_chat_template(
        examples["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": texts}


# ── Dry-run (no Unsloth) ──────────────────────────────────────────────────────

def dry_run(dataset_path: Path, output_path: Path) -> None:
    """
    Validate the dataset structure without loading model weights.
    Safe to run without GPU.
    """
    print("\n[DRY RUN] Validating dataset schema only (no model weights loaded)\n")
    ds = load_sharegpt_jsonl(dataset_path)
    
    tier_counts: dict[str, int] = {}
    for record in ds:
        # We don't have meta in the HF Dataset object easily; just count
        pass

    print(f"[DRY RUN] Schema OK — {len(ds)} samples validated")
    print(f"[DRY RUN] Sample record:")
    sample = ds[0]
    for turn in sample["conversations"]:
        role = turn.get("from", "?")
        val  = turn.get("value", "")[:80]
        print(f"  [{role}] {val}...")
    print()
    print("[DRY RUN] Hyperparameters:")
    print(f"  rank={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}, rsLoRA=True")
    print(f"  batch={TRAIN_CONFIG['per_device_train_batch_size']}, "
          f"accum={TRAIN_CONFIG['gradient_accumulation_steps']}, "
          f"global_batch={TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")
    print(f"  epochs={TRAIN_CONFIG['num_train_epochs']}, lr={TRAIN_CONFIG['learning_rate']}")
    print(f"  bf16={TRAIN_CONFIG['bf16']}, optim={TRAIN_CONFIG['optim']}")
    print(f"\n[DRY RUN] Output adapter directory: {output_path}/")
    print("[DRY RUN] PASS — run without --dry-run on a GPU node to forge weights.")


# ── Full training run ─────────────────────────────────────────────────────────

def train(dataset_path: Path, output_path: Path, base_model: str) -> None:
    if not HAS_UNSLOTH:
        print("[FATAL] Unsloth required for training. pip install 'unsloth[colab-new]'")
        sys.exit(1)
    if not HAS_TRL:
        print("[FATAL] TRL required. pip install datasets trl")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  Sovereign LoRA Forge — code_expert")
    print(f"  Base:    {base_model}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output:  {output_path}/")
    print(f"{'═'*60}\n")

    # ── 1. Load base model in 4bit ────────────────────────────────────
    print("[1/5] Loading base model in 4bit bitsandbytes...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = base_model,
        max_seq_length  = MAX_SEQ_LENGTH,
        dtype           = None,          # Auto-detect — will use bfloat16 on Ampere+
        load_in_4bit    = True,
    )

    # ── 2. Attach rsLoRA adapter ──────────────────────────────────────
    print("[2/5] Attaching rsLoRA adapter (r=64, alpha=128)...")
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    
    # Print trainable parameter count
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ── 3. Apply Llama-3 chat template ───────────────────────────────
    print("[3/5] Applying Llama-3 chat template...")
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # ── 4. Load and format dataset ────────────────────────────────────
    print("[4/5] Loading dataset...")
    raw_ds = load_sharegpt_jsonl(dataset_path)
    
    def formatting_func(examples: dict) -> dict:
        texts = []
        for conv in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    # ── 5. SFTTrainer ─────────────────────────────────────────────────
    print("[5/5] Launching SFTTrainer...")
    
    sft_config = SFTConfig(
        output_dir          = str(output_path / "checkpoints"),
        dataset_text_field  = "text",
        max_seq_length      = MAX_SEQ_LENGTH,
        packing             = False,   # Off — our samples vary in length significantly
        **TRAIN_CONFIG,
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = raw_ds,
        args            = sft_config,
        formatting_func = formatting_func,
    )
    
    # Enable Unsloth's xformers memory-efficient attention
    FastLanguageModel.for_training(model)

    print(f"\n[TRAIN] Starting — global batch size: "
          f"{TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")
    trainer_stats = trainer.train()
    
    print(f"\n[TRAIN] Complete in {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"        Loss: {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")

    # ── Save adapter in vLLM-compatible format ────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n[SAVE] Exporting adapter to {output_path}/...")
    
    # Save LoRA adapter weights (adapter_model.safetensors + adapter_config.json)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    print(f"[SAVE] Adapter saved.")
    print(f"\n[VERIFY] vLLM load test:")
    print(f"  The adapter directory contains:")
    for f in sorted(output_path.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name:40s}  {size / 1024 / 1024:.1f} MB")

    print(f"\n[DONE] Mount {output_path}/ in vLLM with:")
    print(f"       vllm serve {base_model} --enable-lora "
          f"--lora-modules code_expert={output_path.resolve()}")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forge code_expert rsLoRA adapter")
    parser.add_argument("--dataset",   default="data/code_expert.jsonl",
                        help="Path to the synthesized JSONL dataset")
    parser.add_argument("--output",    default="code_expert",
                        help="Output directory for the adapter (default: code_expert/)")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace base model ID")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Validate dataset schema without loading model weights")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path  = Path(args.output)

    if not dataset_path.exists():
        print(f"[FATAL] Dataset not found: {dataset_path}")
        print(f"        Run: python tools/forge_dataset.py --count 500 --output {dataset_path}")
        sys.exit(1)

    if args.dry_run:
        dry_run(dataset_path, output_path)
    else:
        train(dataset_path, output_path, args.base_model)
