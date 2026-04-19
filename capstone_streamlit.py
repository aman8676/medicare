import streamlit as st
from agent import app
from langgraph.checkpoint.memory import MemorySaver

# ------------------ Setup ------------------

st.set_page_config(page_title="MediCare Chatbot", page_icon="🏥")

st.title("🏥 MediCare Assistant")

# Initialize session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session"

# Optional: initialize checkpointer (if needed later)
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

# ------------------ Display Chat ------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------ Input ------------------

user_input = st.chat_input("Type your question here...")

if user_input:
    # Show user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # ------------------ Agent Call ------------------

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    agent_input = {
        "question": user_input,
        "messages": st.session_state.chat_history[:-1]  # previous context
    }

    try:
        response = app.invoke(agent_input, config=config)
        answer = response.get("answer", "Hmm, couldn't find a proper answer.")

    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    # ------------------ Show Response ------------------

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)