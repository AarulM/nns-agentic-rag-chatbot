"""
Shared trace queue for the chat UI's live "what is the agent doing" view.
Every agent (supervisor + the 3 specialists) puts a line on this queue via
callback_handler whenever it calls a tool. The UI runs the agent call on a
background thread and drains this queue on the main thread, writing each
line into a live st.status() box as it arrives — this reproduces the
"Tool #1: ask_hr" live feed you saw in the terminal, but in the browser.

A plain queue.Queue is used (not a list) because it's thread-safe by
design: the agent call happens on a worker thread, but Streamlit widgets
can only be updated from the main thread, so this is the hand-off point
between the two.
"""
import queue

trace_queue: "queue.Queue[str]" = queue.Queue()


def tracing_callback_handler(**kwargs) -> None:
    tool_use = kwargs.get("event", {}).get("contentBlockStart", {}).get("start", {}).get("toolUse")
    if tool_use:
        trace_queue.put(f"Called tool: `{tool_use['name']}`")


def drain_queue() -> None:
    """Empty out any leftover items before starting a new request."""
    while True:
        try:
            trace_queue.get_nowait()
        except queue.Empty:
            break