# frontend/streamlit_app.py
import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG4Finance Chat", layout="centered")
st.title("💬 RAG4Finance Chat")

# Initialize chat history in Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Type your message..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Send to backend
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json={"message": prompt})
        response.raise_for_status()
        data = response.json()

        bot_reply = data.get("reply", "Error: No reply from server")
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").markdown(bot_reply)

    except Exception as e:
        st.error(f"Error contacting backend: {e}")
