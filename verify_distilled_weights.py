"""
LoRA Verification Suite — Post-Distillation
Loads the Qwen2.5-1.5B model + our new sovereign LoRA weights,
and runs the adversarial state mutation benchmark to ensure we
didn't trade breadth for depth (narrow overfitting).
"""

import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("============================================================")
print("  LOADING DISTILLED CORTEX (LoRA over Qwen2.5-1.5B)")
print("============================================================")

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "outputs/sovereign-distilled/final"

try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"[*] Attaching adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("[*] Model loaded and compiled.")
except Exception as e:
    print(f"[!] Failed to load model: {e}")
    sys.exit(1)

def query_llm_distilled(prompt: str) -> str:
    # Use HF chat template for Qwen Instruct
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice off the prompt
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # Strip any <reasoning> tags just to return the raw answer to the eval script
    if "<answer>" in response:
        response = response.split("<answer>")[-1].split("</answer>")[0].strip()
        
    return response.strip()

print("\n[*] Monkey-patching `query_llm` in benchmarks...")

# Import the benchmark and overwrite its LLM call
sys.path.insert(0, os.path.abspath("."))
from benchmarks import state_mutation_eval

# Override the hardcoded query_llm which normally hits Ollama 11434
state_mutation_eval.query_llm = query_llm_distilled

# Override the deterministic fallback mock (it kicks in if you don't hit localhost:11434)
# We actually just let it use the monkey-patched function directly
print("[*] Running State Mutation Eval (Adversarial Post-Tune Check)...")

state_mutation_eval.run_benchmark()

print("\n✅ Verification pass complete.")
