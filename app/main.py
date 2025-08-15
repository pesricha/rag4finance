from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
from app.chains.rag_chain import get_ollama_response

app = FastAPI(title="RAG4Finance Chat Backend")

# Allow frontend (Streamlit) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Streamlit dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep only last 20 messages (10 exchanges)
chat_history = deque(maxlen=20)

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.post("/chat")
def chat_endpoint(msg: ChatMessage):
    # Store user message
    chat_history.append({"role": "user", "content": msg.message})

    # Get response from Ollama
    bot_reply = get_ollama_response(user_query=msg.message, chat_history=list(chat_history))
    chat_history.append({"role": "assistant", "content": bot_reply})

    return {"reply": bot_reply, "history": list(chat_history)}

@app.get("/history")
def get_history():
    return {"history": list(chat_history)}

# Run backend:
# uvicorn app.main:app --reload --port 8000
