import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="Mock vLLM Endpoint")

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 2048
    temperature: float = 0.2

@app.post("/v1/chat/completions")
async def mock_completions(request: ChatRequest):
    print(f"[Mock vLLM] Connection established. Requested adapter: {request.model}")
    
    # Extract the prompt to prove payload integrity
    prompt = request.messages[0]["content"] if request.messages else ""
    
    # Generate the simulated response
    simulated_output = f"Simulated inference executed successfully using hemisphere: {request.model}. Prompt excerpt: '{prompt[:40]}...'"
    
    return {
        "id": "mock-cmpl-123",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": simulated_output
                },
                "finish_reason": "stop"
            }
        ]
    }

if __name__ == "__main__":
    print("Starting Mock vLLM Server on http://localhost:8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
