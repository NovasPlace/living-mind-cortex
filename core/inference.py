import httpx
import json

class SovereignInferenceClient:
    """
    Executes the generation pass, respecting the BiomechanicRouter's LoRA state.
    Assumes vLLM is running locally with --enable-lora.
    """
    def __init__(self, vllm_url="http://localhost:8001/v1"):
        self.vllm_url = vllm_url
        self.base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct" # The persistent substrate

    async def generate(self, prompt: str, adapter_id: str) -> str:
        """
        Pipes the prompt through the appropriate hemisphere.
        """
        # Target the base weights, or target the specific LoRA adapter name
        target_model = self.base_model_name if adapter_id == "base_model" else adapter_id
        
        print(f"[Inference] Executing pass through hemisphere: {target_model}")
        
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.2
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(f"{self.vllm_url}/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except httpx.HTTPError as e:
                print(f"[Inference Error] Substrate failure: {e}")
                return ""
