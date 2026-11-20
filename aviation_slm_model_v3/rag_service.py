
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import requests

# Import our tested RAG chatbot
sys.path.insert(0, str(Path(__file__).parent))
import aviation_rag_cli as rag

# Use the new Ollama model, not the original model
rag.OLLAMA_MODEL = "aviation-slm-v3-rag:latest"

app = FastAPI(
    title="Aviation SLM V3 RAG Integration",
    description="Verified aviation record retrieval connected to Ollama",
    version="1.0"
)

class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {
        "status": "running",
        "rag": "enabled",
        "ollama_model": rag.OLLAMA_MODEL
    }

@app.post("/chat")
def chat(data: ChatRequest):
    question = data.question.strip()

    if not question:
        return {
            "success": False,
            "answer": "Please enter a question."
        }

    try:
        answer = rag.aviation_chatbot(question)
        return {
            "success": True,
            "question": question,
            "answer": answer
        }
    except Exception as error:
        return {
            "success": False,
            "answer": f"Unable to process the question: {error}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
