from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
from app.chains.rag_chain import SmartChatAgent

app = FastAPI(title="RAG4Finance Chat Backend")

# Allow frontend (Streamlit) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Streamlit dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
bot_obj = SmartChatAgent(
    model_name="qwen:1.8b", temperature=0.8, max_chat_history=10, max_plan_steps=2
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

    # Get response from SmartChatAgent
    # TODO: Make User configurable p3

    bot_reply = bot_obj.answer_user_query(question=msg.message)
    chat_history.append({"role": "assistant", "content": bot_reply})

    return {"reply": bot_reply, "history": list(chat_history)}


@app.get("/history")
def get_history():
    return {"history": list(chat_history)}


# Run backend:
# uvicorn app.main:app --reload --port 8000
