
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any
import requests
import time
import json

app = FastAPI(title="Aviation RAG Ollama-Compatible API")

RAG_URL = "http://127.0.0.1:8004/chat"
MODEL = "aviation-slm-v3-rag:latest"
BASE_MODEL = "aviation-slm-v3:latest"

def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def get_answer(question):
    try:
        response = requests.post(
            RAG_URL,
            json={"question": question},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "No answer available.")
    except requests.RequestException as error:
        raise HTTPException(status_code=503, detail=str(error))

def validate_model(model):
    if model not in (MODEL, "aviation-slm-v3-rag"):
        raise HTTPException(status_code=404, detail="Model not found")

class GenerateRequest(BaseModel):
    model: str = MODEL
    prompt: str = ""
    stream: bool = False

class ChatRequest(BaseModel):
    model: str = MODEL
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False

@app.get("/")
@app.head("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "running", "model": MODEL, "rag": "enabled"}

@app.get("/api/version")
def version():
    return {"version": "0.9.0"}

@app.get("/api/tags")
def tags():
    return {
        "models": [{
            "name": MODEL,
            "model": MODEL,
            "modified_at": timestamp(),
            "size": 170000000,
            "digest": "aviation-rag-proxy",
            "details": {
                "parent_model": BASE_MODEL,
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "82.38M",
                "quantization_level": "F16"
            }
        }]
    }

@app.post("/api/show")
def show(data: dict):
    validate_model(data.get("model", MODEL))
    return {
        "modelfile": "FROM " + BASE_MODEL,
        "parameters": "temperature 0",
        "template": "",
        "details": {
            "parent_model": BASE_MODEL,
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "82.38M",
            "quantization_level": "F16"
        },
        "model_info": {},
        "capabilities": ["completion"]
    }

@app.post("/api/generate")
def generate(data: GenerateRequest):
    validate_model(data.model)
    answer = get_answer(data.prompt.strip())

    result = {
        "model": MODEL,
        "created_at": timestamp(),
        "response": answer,
        "done": True,
        "done_reason": "stop"
    }

    if data.stream:
        def stream_result():
            yield json.dumps(result, ensure_ascii=False) + "\n"
        return StreamingResponse(
            stream_result(),
            media_type="application/x-ndjson"
        )

    return result

@app.post("/api/chat")
def chat(data: ChatRequest):
    validate_model(data.model)

    user_messages = [
        m.get("content", "")
        for m in data.messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
    ]

    question = user_messages[-1].strip() if user_messages else ""
    answer = get_answer(question)

    result = {
        "model": MODEL,
        "created_at": timestamp(),
        "message": {
            "role": "assistant",
            "content": answer
        },
        "done": True,
        "done_reason": "stop"
    }

    if data.stream:
        def stream_result():
            yield json.dumps(result, ensure_ascii=False) + "\n"
        return StreamingResponse(
            stream_result(),
            media_type="application/x-ndjson"
        )

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=11435)
