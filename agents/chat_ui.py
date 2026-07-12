"""
Simple local chat UI for testing the NNS assistant in a browser instead of
the terminal. Runs entirely on your machine, talks to whatever model
MODEL_PROVIDER is set to (Ollama by default — free). This is NOT a hosted
site anyone else can reach; it only runs while you have it open locally.

Shows tool calls live as they happen, in a collapsible status box —
the same "Tool #1: ask_hr" style trace you'd see running this in a
terminal, but streamed into the browser. This works by running the agent
on a background thread while the main thread drains a queue of trace
lines and writes them into the status box (Streamlit widgets can only be
updated from the main thread, so the queue is the hand-off point).

Run:  streamlit run chat_ui.py
"""
import queue
import threading

import streamlit as st
from supervisor import handle_request
from trace_log import trace_queue, drain_queue

st.set_page_config(page_title="NNS Assistant", page_icon="🚢")
st.title("NNS Assistant (local test UI)")

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("Ask about HR, Safety, or Operations...")
if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        drain_queue()
        result: dict[str, str] = {}

        def run_agent() -> None:
            result["response"] = handle_request(user_input)

        worker = threading.Thread(target=run_agent, daemon=True)
        worker.start()

        with st.status("Thinking...", expanded=True) as status:
            while worker.is_alive() or not trace_queue.empty():
                try:
                    line = trace_queue.get(timeout=0.1)
                    status.write(line)
                except queue.Empty:
                    continue
            worker.join()
            status.update(label="Done", state="complete", expanded=False)

        response = result["response"]
        st.markdown(response)
    st.session_state.history.append(("assistant", response))