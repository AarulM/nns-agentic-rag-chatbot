"""
Operations Agent — SOP lookup plus calendar/ticket/Jabber actions via MCP.
"""
from strands import Agent, tool
from knowledge_base import search_docs
from mcp_gateway_client import create_smax_ticket, get_calendar_events, send_jabber_message
from model_config import get_model
from trace_log import tracing_callback_handler


@tool
def search_ops_docs(query: str) -> str:
    """Search shipbuilding SOPs, schedules, and operations procedures."""
    return search_docs(query, "No matching operations documents found.")


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
        "You are the Operations assistant for Newport News Shipbuilding, a "
        "Huntington Ingalls Industries shipyard. Write in a professional, "
        "courteous tone appropriate for a defense-industry workplace. Only "
        "use check_my_calendar, notify_team_on_jabber, or open_ops_ticket "
        "when the user explicitly asks for that specific action (checking a "
        "calendar, sending a message, filing a ticket) — never call them "
        "while answering an informational question, even if the topic is "
        "loosely related. If the user has given enough detail to act — a "
        "recipient and message for Jabber, or a clear summary for a ticket — "
        "call the tool immediately, don't just describe the steps. Only ask "
        "a follow-up question first if something required is truly missing "
        "(e.g. no recipient named). Do not answer PPE or safety-equipment "
        "questions — those belong to the Safety agent.\n\n"
        "When you send a Jabber message or open a ticket on the user's "
        "behalf, do NOT copy their words verbatim — compose it as a brief, "
        "professional workplace message written in first person as the "
        "user: greet the recipient by name, state the point clearly and "
        "courteously, and close politely. For example, 'i wil be 10 mins "
        "late' becomes: 'Hi Lyndsey — a quick heads up that I'm running "
        "about 10 minutes late this morning. Apologies for any "
        "inconvenience.' Polish the wording ONLY — every fact in the "
        "message must come from the user's request. If the user gave no "
        "specific time, reason, or commitment, the message must not "
        "contain one. After sending, tell the user exactly what you "
        "sent.\n\n"
        "Use search_ops_docs ONLY for questions about this company's own "
        "SOPs, schedules, and processes. For general knowledge questions "
        "(what a term means, how something works in general), answer "
        "directly from your own knowledge without calling any tool.\n\n"
        "After search_ops_docs returns results, check whether they actually "
        "answer the specific question asked. If they don't, say the company "
        "documentation doesn't cover it — then, if you can, still help from "
        "your own general knowledge, clearly labeled as general guidance "
        "rather than company procedure. Never present unrelated search "
        "results as the answer, and never invent company procedures."
    ),
    tools=[search_ops_docs, check_my_calendar, notify_team_on_jabber, open_ops_ticket],
)

if __name__ == "__main__":
    print(operations_agent("What's the SOP for a hull section quality check?"))