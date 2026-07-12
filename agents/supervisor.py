"""
Supervisor agent — the "agents as tools" pattern from Strands. This module
is what AgentCore Runtime would host as the entrypoint.

Run locally:  python supervisor.py
Deploy later: wrap `handle_request` with the AgentCore Runtime entrypoint
decorator per the samples in Build_Plan.md, then `agentcore launch`.
"""
import json
import re
from strands import Agent, tool
from model_config import get_model
from guardrail import apply_guardrail
from hr_agent import hr_agent
from safety_agent import safety_agent
from operations_agent import operations_agent
from aws_config import MEMORY_ID
from memory_hook import ShortTermMemoryHookProvider, memory_client
from trace_log import tracing_callback_handler, drain_queue, trace_queue

# Static for now since this is a single-user local test. In a real
# deployment, actor_id/session_id would be set per logged-in employee and
# per conversation instead of hardcoded.
ACTOR_ID = "local_test_user"
SESSION_ID = "local_test_session_2"


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
        "combine the answers. If none of the specialists can answer, say so "
        "honestly instead of guessing.\n\n"
        "Never mention tool or function names (ask_hr, ask_safety, "
        "ask_operations, search_hr_docs, etc.) in your reply — those are "
        "internal implementation details the user should never see. Do not "
        "narrate what you're about to do (\"I'll pass this along to...\") — "
        "just answer. Base your answer only on what the specialist actually "
        "returned; do not invent additional documents, guides, portals, or "
        "processes that weren't in that result."
    ),
    tools=[ask_hr, ask_safety, ask_operations],
    hooks=[ShortTermMemoryHookProvider(memory_client, MEMORY_ID)],
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


def handle_request(user_message: str) -> str:
    if _GREETING_PATTERN.match(user_message):
        drain_queue()
        return "Hey! I'm the NNS assistant — ask me about HR, Safety, or Operations."

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
    print("NNS Assistant — type 'quit' to exit. Memory persists even if you quit and restart.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        print(f"\nAssistant: {handle_request(user_input)}\n")