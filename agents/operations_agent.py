"""
Operations Agent — SOP lookup plus calendar/ticket/Jabber actions via MCP.
"""
import boto3
from strands import Agent, tool
from mcp_gateway_client import create_smax_ticket, get_calendar_events, send_jabber_message
from model_config import get_model
from trace_log import tracing_callback_handler

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")

OPS_KNOWLEDGE_BASE_ID = "RW01IL1SNT"


@tool
def search_ops_docs(query: str) -> str:
    """Search shipbuilding SOPs, schedules, and operations procedures."""
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=OPS_KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
    )
    chunks = [r["content"]["text"] for r in response.get("retrievalResults", [])]
    return "\n---\n".join(chunks) if chunks else "No matching operations documents found."


@tool
def check_my_calendar(date: str) -> str:
    """Look up the current user's calendar events for a given date (YYYY-MM-DD)."""
    events = get_calendar_events(date)
    return "\n".join(events) if events else f"No events found for {date}."


@tool
def notify_team_on_jabber(recipient: str, message: str) -> str:
    """Send a Jabber message to a coworker or team channel. Only call this
    when the user explicitly asks you to send/notify/message someone — never
    speculatively while answering an unrelated informational question."""
    send_jabber_message(recipient, message)
    return f"Message sent to {recipient}."


@tool
def open_ops_ticket(summary: str, priority: str = "normal") -> str:
    """Open a SMAX operations ticket (e.g., equipment issue, schedule conflict).
    Only call this when the user explicitly asks to file/open/submit a
    ticket or request — never speculatively while answering an unrelated
    informational question like onboarding or safety procedures."""
    ticket_id = create_smax_ticket(category="operations", summary=summary, details={"priority": priority})
    return f"Operations ticket {ticket_id} created."


operations_agent = Agent(
    name="operations_agent",
    model=get_model(),
    callback_handler=tracing_callback_handler,
    system_prompt=(
        "You are the Operations assistant for a shipbuilding company. Use "
        "search_ops_docs for SOP/process questions. Only use check_my_calendar, "
        "notify_team_on_jabber, or open_ops_ticket when the user explicitly "
        "asks for that specific action (checking a calendar, sending a "
        "message, filing a ticket) — never call them while answering a "
        "general informational question like onboarding or SOPs, even if the "
        "topic is loosely related. If the user has given enough detail to "
        "act — a recipient and message for Jabber, or a clear summary for a "
        "ticket — call the tool immediately, don't just describe the steps. "
        "Only ask a follow-up question first if something required is truly "
        "missing (e.g. no recipient named). Do not answer PPE or "
        "safety-equipment questions — those belong to the Safety agent.\n\n"
        "After search_ops_docs returns results, check whether they actually "
        "answer the specific question asked before using them. If the results "
        "are about a different topic, say plainly that you couldn't find "
        "documentation on that specific question rather than presenting "
        "unrelated content as the answer."
    ),
    tools=[search_ops_docs, check_my_calendar, notify_team_on_jabber, open_ops_ticket],
)

if __name__ == "__main__":
    print(operations_agent("What's the SOP for a hull section quality check?"))