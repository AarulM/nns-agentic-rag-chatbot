"""
Operations Agent — SOP lookup plus calendar/ticket/Jabber actions via MCP.
"""
import boto3
from strands import Agent, tool
from mcp_gateway_client import create_smax_ticket, get_calendar_events, send_jabber_message
from model_config import get_model

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

OPS_KNOWLEDGE_BASE_ID = "OSOGWWRI0X"


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
    """Send a Jabber message to a coworker or team channel."""
    send_jabber_message(recipient, message)
    return f"Message sent to {recipient}."


@tool
def open_ops_ticket(summary: str, priority: str = "normal") -> str:
    """Open a SMAX operations ticket (e.g., equipment issue, schedule conflict)."""
    ticket_id = create_smax_ticket(category="operations", summary=summary, details={"priority": priority})
    return f"Operations ticket {ticket_id} created."


operations_agent = Agent(
    name="operations_agent",
    model=get_model(),
    system_prompt=(
        "You are the Operations assistant for a shipbuilding company. Use "
        "search_ops_docs for SOP/process questions, and the calendar/ticket/"
        "Jabber tools for real actions. Always confirm before sending a "
        "message or opening a ticket on the user's behalf."
    ),
    tools=[search_ops_docs, check_my_calendar, notify_team_on_jabber, open_ops_ticket],
)

if __name__ == "__main__":
    print(operations_agent("What's on my calendar for 2026-07-14?"))