"""
Supervisor agent — the "agents as tools" pattern from Strands. This module
is what AgentCore Runtime would host as the entrypoint.

Run locally:  python supervisor.py
Deploy later: wrap `handle_request` with the AgentCore Runtime entrypoint
decorator per the samples in Build_Plan.md, then `agentcore launch`.
"""
import json
import re
import uuid
from strands import Agent, tool
from model_config import get_model, PROVIDER
from guardrail import apply_guardrail
from hr_agent import hr_agent
from safety_agent import safety_agent
from operations_agent import operations_agent
from web_search import web_search
from aws_config import MEMORY_TABLE
from memory_hook import ShortTermMemoryHookProvider, memory_client

# Long-term memory needs strands.memory (strands-agents >= 1.47, pinned in
# agents/requirements.txt). On an older install, run the app without it
# instead of crashing — short-term memory above still works.
try:
    from strands.memory import MemoryManager
    from memory_store import UserFactStore
except ImportError as e:
    MemoryManager = None
    print(
        f"WARNING: long-term memory disabled ({e}). "
        "Upgrade with: pip install -U 'strands-agents[ollama]'"
    )
from trace_log import tracing_callback_handler, drain_queue, trace_queue

# Static actor for this single-user local test; in a real deployment it
# would be the logged-in employee. The session ID is fresh on every app
# start ON PURPOSE: when every run shared one hardcoded session, each
# startup seeded 5 turns of stale history from ALL previous test chats
# into the model's context, and the small local model would answer those
# old topics instead of the current question. Memory still records and
# recalls everything within the running session.
ACTOR_ID = "local_test_user"
SESSION_ID = f"local_test_{uuid.uuid4().hex[:8]}"


@tool
def ask_hr(question: str) -> str:
    """Route a question to the HR agent (onboarding, benefits, HR policy)."""
    return str(hr_agent(question))


@tool
def ask_safety(question: str) -> str:
    """Route a question to the Safety agent (procedures, PPE, incident reporting)."""
    return str(safety_agent(question))


@tool
def ask_operations(question: str) -> str:
    """Route a question to the Operations agent (SOPs, scheduling, tickets, Jabber)."""
    return str(operations_agent(question))


supervisor = Agent(
    name="supervisor",
    model=get_model(),
    callback_handler=tracing_callback_handler,
    # NOTE: the ROUTING portion of this prompt is tuned to stay stable with
    # llama3.1:8b — keep edits to it minimal (restructuring once made that
    # model loop on tools; renaming the company made every answer a
    # greeting). The web_search paragraph was added for the Bedrock default,
    # where it behaves. Branding lives in the greeting fast-path below, the
    # specialist prompts, and the UI.
    system_prompt=(
        "You are the front-door assistant for a shipbuilding company's internal "
        "chatbot, used by employees, new hires, and interns. Respond warmly and "
        "briefly to greetings and small talk yourself — do not call a tool and "
        "do not refuse just because a message isn't a work question.\n\n"
        "For actual work questions, decide which specialist to call:\n"
        "- ask_hr: onboarding, benefits, PTO, pay, HR policy.\n"
        "- ask_safety: PPE/protective equipment, safety procedures, hazards, "
        "OSHA rules, reporting a safety incident.\n"
        "- ask_operations: SOPs, schedules, calendar, tickets, Jabber messages.\n"
        "PPE and protective-equipment questions always go to ask_safety, never "
        "ask_operations, even if they also mention the shipyard or a work area.\n"
        "Any request to actually DO something — send a Jabber message, open a "
        "ticket, check the calendar — always goes to ask_operations, even if "
        "the topic (like a PTO request) would otherwise go to ask_hr. HR can "
        "only answer policy questions; it cannot file requests or send messages.\n\n"
        "If the question spans more than one area, call more than one tool and "
        "combine the answers.\n\n"
        "For anything OUTSIDE this company's HR, Safety, and Operations — "
        "general knowledge, current events, external companies or products, "
        "public reference facts — use web_search to look it up on the internet, "
        "then answer from what it returns and cite the source. Do NOT use "
        "web_search for internal company policy; that belongs to the "
        "specialists. If a specialist has no answer and the web wouldn't know "
        "either, say so honestly instead of guessing.\n\n"
        "Never mention tool or function names (ask_hr, ask_safety, "
        "ask_operations, search_hr_docs, etc.) in your reply — those are "
        "internal implementation details the user should never see. Do not "
        "narrate what you're about to do (\"I'll pass this along to...\") — "
        "just answer. Base your answer only on what the specialist actually "
        "returned; do not invent additional documents, guides, portals, or "
        "processes that weren't in that result."
    ),
    tools=[ask_hr, ask_safety, ask_operations, web_search],
    hooks=[ShortTermMemoryHookProvider(memory_client, MEMORY_TABLE)],
    # Long-term memory (Strands MemoryManager, DynamoDB-backed — see
    # memory_store.py): facts about the user are auto-extracted in the
    # background and injected back into context on each user turn.
    # search_tool_config=False on purpose: recall happens via injection, so
    # the tool adds nothing but risk — when llama3.1:8b had a search_memory
    # tool, it hallucinated fake tool-result JSON in its replies ("Your
    # supervisor is John Smith"), and extraction then saved that fiction as
    # a real fact. No new tool also means routing behaves exactly as before.
    # Long-term memory (Strands MemoryManager over DynamoDB — see
    # memory_store.py): facts about the user are auto-extracted in the
    # background and injected into context on later turns, across sessions.
    # Bedrock-only, like the guardrail, and after extensive testing — with
    # llama3.1:8b it is strictly net-negative: injected memory derails it
    # (it answered "what is my badge number?" with small talk or a loop of
    # ask_operations calls, whatever the injection format), and as the
    # extraction model it invents facts out of thin air ("supervisor is
    # John Smith") that then poison every later session. Claude recalls
    # cleanly with zero tool derailing. search_tool_config=False because
    # injection already covers recall — a search_memory tool adds nothing
    # but another way for a model to go sideways.
    memory_manager=(
        MemoryManager(
            stores=[UserFactStore(MEMORY_TABLE, ACTOR_ID)],
            search_tool_config=False,
        )
        if PROVIDER == "bedrock" and MemoryManager is not None
        else None
    ),
    state={"actor_id": ACTOR_ID, "session_id": SESSION_ID},
)


# --- Fast path for greetings/small talk ---------------------------------
# The local model sometimes ignores "don't call a tool for greetings" and
# calls ask_hr anyway, wasting a full round-trip through Ollama. Catching
# the obvious cases here in plain Python is instant and 100% reliable — no
# model call happens at all for these, which directly cuts latency too.
_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|what'?s\s*up|how'?s\s*it\s*going|how\s*are\s*you|"
    r"good\s*(morning|afternoon|evening)|thanks|thank\s*you|bye|goodbye)"
    r"\s*(everyone|everybody)?\s*[!.?]*\s*$",
    re.IGNORECASE,
)


# --- Leaked tool-call recovery -------------------------------------------
# llama3.1:8b sometimes emits the routing tool call as literal JSON text
# ('{"name": "ask_operations", "parameters": {...}}') instead of a native
# tool call, so Strands never executes it and the raw JSON becomes the
# "answer". When the reply is exactly that shape, parse it and invoke the
# specialist ourselves so the user still gets a real answer.
_LEAKED_TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(ask_hr|ask_safety|ask_operations)"\s*,\s*'
    r'"parameters"\s*:\s*(\{.*\})\s*\}',
    re.DOTALL,
)

_SPECIALISTS = {
    "ask_hr": hr_agent,
    "ask_safety": safety_agent,
    "ask_operations": operations_agent,
}


def _recover_leaked_tool_call(response: str) -> str:
    match = _LEAKED_TOOL_CALL_PATTERN.search(response)
    if not match:
        return response
    tool_name, raw_params = match.groups()
    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError:
        try:
            # The model often emits \' inside strings, which JSON forbids.
            params = json.loads(raw_params.replace("\\'", "'"))
        except json.JSONDecodeError:
            return response
    question = params.get("question")
    if not isinstance(question, str) or not question:
        return response
    trace_queue.put(f"Recovered leaked tool call → `{tool_name}`")
    return str(_SPECIALISTS[tool_name](question))


# --- Questions about a just-attached file --------------------------------
# Routing these through the supervisor is wrong, and was actively broken:
# "what is this image about" has nothing to do with HR, Safety, or
# Operations, so the router picked a specialist anyway, that specialist
# searched the Knowledge Base, found nothing, and llama3.1:8b looped on
# ask_operations. Worse, the file's text only reaches the KB after an
# ingestion job that takes minutes — so even a perfect router could not
# have answered from it yet.
#
# When the user attaches a file and asks about it in the same breath, the
# answer is already in hand. Read it directly: no tools, no retrieval, no
# waiting on a sync.
_document_qa = Agent(
    name="document_qa",
    # No guardrail: reading an uploaded file must not trip the PROMPT_ATTACK
    # filter on instruction-like content (a .py with a system prompt, a
    # config, an agent framework). PII is already redacted from the text,
    # and answer_about_documents still runs the output through the guardrail.
    model=get_model(guardrail=False),
    callback_handler=tracing_callback_handler,
    system_prompt=(
        "You answer questions about documents the user has just uploaded. "
        "The extracted text of those files is given to you directly.\n\n"
        "Answer only from that text. If it does not contain the answer, say "
        "so plainly — do not guess, and do not invent details that are not "
        "there. If the text looks like a transcript of speech, treat it as "
        "what someone said. If it looks like a description of a picture, "
        "treat it as what the picture shows.\n\n"
        "Be brief and direct. Do not mention tools, extraction, knowledge "
        "bases, or how the text reached you."
    ),
    tools=[],
)

# How much of each attached file goes into the model's context — sized to
# the active model. Claude's window is ~200k tokens, so a whole spreadsheet
# or report fits and the model sees every row; the old flat 6000-char cap
# silently dropped everything past the first ~15 rows of a large file (a
# tracker's A-tier rows never reached the model, so it reported them "not
# visible"). llama3.1:8b runs with num_ctx=8192, where a long file would
# push the question itself out of the window, so it keeps the tight cap.
_MAX_CONTEXT_CHARS_PER_FILE = 200_000 if PROVIDER == "bedrock" else 6_000


# How to introduce each kind of extracted text. Without this the model
# describes the plumbing instead of the content — asked what a photo was,
# it answered "the first file appears to be a description of an image",
# because that is literally what it was handed.
_SECTION_HEADERS = {
    "vision": 'What the image "{name}" shows',
    "bda": 'Contents of "{name}"',
    "textract": 'Form fields read from "{name}"',
    "spreadsheet": 'Spreadsheet "{name}" as CSV',
    "plaintext": 'Contents of the file "{name}"',
}


def answer_about_documents(
    question: str, documents: list[tuple[str, str]] | list[tuple[str, str, str]]
) -> str:
    """
    Answer `question` from text just extracted from attachments.

    `documents` is a list of (filename, extracted_text) or
    (filename, extracted_text, extractor).
    """
    blocked, message = apply_guardrail(question, "INPUT")
    if blocked:
        drain_queue()
        return message

    sections = []
    for entry in documents:
        filename, text = entry[0], entry[1]
        extractor = entry[2] if len(entry) > 2 else ""
        excerpt = text[:_MAX_CONTEXT_CHARS_PER_FILE]
        if len(text) > _MAX_CONTEXT_CHARS_PER_FILE:
            excerpt += "\n[...truncated...]"
        header = _SECTION_HEADERS.get(extractor, 'Contents of "{name}"').format(
            name=filename
        )
        sections.append(f"--- {header} ---\n{excerpt}")
    context = "\n\n".join(sections)

    prompt = f"{context}\n\n--- Question ---\n{question}"
    response = str(_document_qa(prompt))

    _, response = apply_guardrail(response, "OUTPUT")
    return response


def handle_request(user_message: str) -> str:
    if _GREETING_PATTERN.match(user_message):
        drain_queue()
        return (
            "Hi! I'm the Newport News Shipbuilding assistant. "
            "I can help with HR, Safety, or Operations — what do you need?"
        )

    # Guardrail on the way in: refuse harmful/ITAR questions before any
    # model or tool sees them (works in ollama mode too, unlike the
    # guardrail attached to BedrockModel).
    blocked, message = apply_guardrail(user_message, "INPUT")
    if blocked:
        drain_queue()
        return message

    response = _recover_leaked_tool_call(str(supervisor(user_message)))

    # Guardrail on the way out: blocks disallowed responses and anonymizes
    # PII (names/emails/phones/SSNs become {NAME}, {EMAIL}, ...).
    _, response = apply_guardrail(response, "OUTPUT")
    return response


if __name__ == "__main__":
    print("NNS Assistant — type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        print(f"\nAssistant: {handle_request(user_input)}\n")