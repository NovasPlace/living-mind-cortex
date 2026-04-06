"""
Vision Organ — Living Mind
Category: Perception
Analyzes images and screenshots using a local multimodal LLM (moondream).
"""

import base64
import json
import aiohttp

class VisionOrgan:
    def __init__(self):
        self.total_analyzed = 0
        self.model = "moondream"
        self.api_url = "http://localhost:11434/api/generate"

    async def analyze_image(self, image_path: str, prompt: str = "Describe what you see in this screenshot in detail.") -> str:
        """Reads an image from disk and sends it to local Ollama vision model."""
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return f"Vision Error: Could not read image at {image_path}. {e}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.2}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, timeout=60) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.total_analyzed += 1
                        return data.get("response", "").strip()
                    else:
                        err = await resp.text()
                        return f"Vision Error: Ollama API returned {resp.status} - {err}"
        except Exception as e:
            return f"Vision Error: Connection to Ollama failed. {e}"

vision = VisionOrgan()
