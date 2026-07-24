"""
Local chat UI for the NNS assistant. Runs entirely on your machine, talks
to whatever model MODEL_PROVIDER is set to (Ollama by default — free).
This is NOT a hosted site anyone else can reach; it only runs while you
have it open locally.

Tool calls stream into a collapsible status box as they happen — the same
"Tool #1: ask_hr" trace you'd see in a terminal. The agent runs on a
background thread while the main thread drains a queue of trace lines
(Streamlit widgets can only be updated from the main thread, so the queue
is the hand-off point).

Files are attached with the paperclip inside the chat input bar. Any
format is accepted: images and photos go to a vision model, PDFs are read
locally (Bedrock Data Automation is the fallback for scanned ones),
audio/video go to Bedrock Data Automation, spreadsheets are parsed
locally, and anything textual — prose, config, source code in any
language — is read directly with no service call at all.

On infrastructure errors: they are logged to the terminal, NOT shown in
the chat. Someone asking "what is this image about" does not need a
paragraph about AccessDeniedException — they need the answer, which the
app already has. Only failures that change what the user can do (the file
could not be read at all) surface in the conversation.

Run:  streamlit run chat_ui.py
"""
import logging
import os
import queue
import threading

import streamlit as st

from supervisor import handle_request, answer_about_documents
from trace_log import trace_queue, drain_queue
from file_ingest import (
    ExtractionError,
    UnsupportedFileType,
    classify,
    extract,
    publish_to_knowledge_base,
)
from env_check import run_checks, format_results, FAIL

logger = logging.getLogger("nns.chat_ui")
logging.basicConfig(level=logging.INFO)

# Textract key/value extraction instead of the default path. Off by
# default: it costs roughly 5x per page and only helps on structured
# paperwork. An env var rather than a widget, to keep the screen clean.
FORMS_MODE = os.environ.get("FORMS_MODE", "").lower() in ("1", "true", "yes")

st.set_page_config(page_title="NNS Assistant", page_icon="🚢", layout="centered")

if "history" not in st.session_state:
    # (role, markdown) pairs. File results are stored as rendered markdown
    # rather than re-derived, so scrolling back never re-runs an extraction.
    st.session_state.history = []
# --- Environment gate ----------------------------------------------------
# Checked once per session rather than on every rerun: these are network
# calls, and the answer does not change while the app is open.
@st.cache_data(show_spinner=False, ttl=300)
def _environment_report():
    results = run_checks()
    return [r.level for r in results].count(FAIL), format_results(results)


blocking, report = _environment_report()
if blocking:
    st.error(
        f"{blocking} configuration problem(s) — the assistant will not work "
        "until these are fixed.",
        icon=":material/error:",
    )
    with st.expander("Environment report", expanded=True):
        st.code(report, language="text")
    st.stop()


# --- Header --------------------------------------------------------------
st.title("Newport News Shipbuilding Assistant")
st.caption("Huntington Ingalls Industries · internal assistant (local test UI)")


def ingest(name: str, data: bytes) -> tuple[tuple[str, str] | None, str | None]:
    """
    Read one attached file, and index it if the Knowledge Base is reachable.

    Returns (extracted_text, user_facing_error).

    Indexing failures deliberately return no error: the text is in hand, so
    the user can still ask about the file right now. Losing cross-session
    search is an operator problem, and it goes to the log where an operator
    will see it.
    """
    try:
        classify(name, data)
    except UnsupportedFileType as error:
        return None, str(error)

    with st.status(f"Reading {name}...", expanded=False) as status:
        try:
            document = extract(data, name, forms_mode=FORMS_MODE)
            if document.is_empty:
                raise ExtractionError("there was nothing readable in it.")
        except (ExtractionError, UnsupportedFileType) as error:
            logger.warning("Extraction failed for %s: %s", name, error)
            status.update(label=f"Could not read {name}", state="error")
            return None, str(error)

        words = len(document.text.split())
        source = {
            "vision": "read as an image",
            "plaintext": "read directly",
            "spreadsheet": "parsed locally",
            "pdf-text": "read locally (PDF text)",
            "textract": "read with Textract",
            "bda": "read with Bedrock Data Automation",
        }.get(document.extractor, document.extractor)
        if document.metadata.get("from_cache"):
            source = "from cache"
        st.caption(f"{words:,} words · {source}")
        st.text(document.text[:3000])

        try:
            publish_to_knowledge_base(document)
        except (ExtractionError, UnsupportedFileType) as error:
            # Logged, not shown. See the docstring.
            logger.warning("Indexing failed for %s: %s", name, error)

        status.update(label=f"Read {name} ({words:,} words)", state="complete")
        return (document.text, document.extractor), None


# --- Conversation --------------------------------------------------------
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

# Empty-state starters, so a new user has something to click. Each just
# queues a question into the composer; the specialists (HR / Safety /
# Operations) route from there. A spread across all three areas shows the
# range of what the assistant can answer.
SUGGESTIONS = {
    ":material/health_and_safety: PPE for welding": "What PPE do I need for welding?",
    ":material/badge: New hire first week": "What do I need to do in my first week as a new hire?",
    ":material/beach_access: Time off": "How much PTO do I get, and how do I request it?",
    ":material/report: Report an incident": "How do I report a safety incident?",
    ":material/local_shipping: Confined space entry": "What are the rules for confined space entry?",
    ":material/payments: Pay & benefits": "What benefits does the company offer employees?",
}
if not st.session_state.history:
    picked = st.pills(
        "Try one", list(SUGGESTIONS), label_visibility="collapsed", key="starter"
    )
    if picked:
        st.session_state.queued_question = SUGGESTIONS[picked]

# --- Composer ------------------------------------------------------------
# The attach control is the paperclip INSIDE the input bar, right next to
# the text area and the send button (the ChatGPT layout). accept_file
# turns the return value into an object carrying .text and .files, and
# attached files show as removable chips in the bar — no separate dialog.
submission = st.chat_input(
    "Ask a question, or attach a file and ask about it",
    accept_file="multiple",
)
queued = st.session_state.pop("queued_question", None)

if submission is not None:
    user_text = (submission.text or "").strip()
    attachments = [(file.name, file.getvalue()) for file in submission.files]
else:
    user_text = (queued or "").strip()
    attachments = []

if user_text or attachments:
    shown = user_text
    if attachments:
        names = ", ".join(f"`{name}`" for name, _ in attachments)
        shown = f"{shown}\n\n:material/attach_file: {names}".strip()

    st.session_state.history.append(("user", shown))
    with st.chat_message("user"):
        st.markdown(shown)

    with st.chat_message("assistant"):
        # 1) Read anything attached, keeping the text for step 2.
        attached_text: list[tuple[str, str, str]] = []
        reply_parts: list[str] = []
        for name, data in attachments:
            read, error = ingest(name, data)
            if read:
                text, extractor = read
                attached_text.append((name, text, extractor))
            else:
                reply_parts.append(f"I couldn't read **{name}** — {error}")

        # 2) Answer.
        if user_text:
            drain_queue()
            result: dict[str, str] = {}

            def run_agent() -> None:
                # A question asked alongside an attachment is almost always
                # ABOUT that attachment ("what is this image?"), not about
                # HR/Safety/Operations. Answer it from the text just
                # extracted instead of routing to a specialist that would
                # search the Knowledge Base — which does not contain the
                # file yet, and may never if the sync failed.
                if attached_text:
                    result["response"] = answer_about_documents(
                        user_text, attached_text
                    )
                else:
                    result["response"] = handle_request(user_text)

            worker = threading.Thread(target=run_agent, daemon=True)
            worker.start()

            with st.status("Thinking...", expanded=True) as status:
                while worker.is_alive() or not trace_queue.empty():
                    try:
                        status.write(trace_queue.get(timeout=0.1))
                    except queue.Empty:
                        continue
                worker.join()
                status.update(label="Done", state="complete", expanded=False)
            reply_parts.append(result["response"])
        elif attached_text:
            names = ", ".join(f"**{item[0]}**" for item in attached_text)
            reply_parts.append(f"I've read {names}. What would you like to know?")

        reply = "\n\n".join(reply_parts)
        st.markdown(reply)

    st.session_state.history.append(("assistant", reply))
    st.rerun()
