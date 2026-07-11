"""
Mock backend for SMAX / calendar / Jabber, meant to run as a Lambda
function behind AgentCore Gateway. This is fake, in-memory data standing
in for the real company systems, so you can test the whole Gateway -> MCP
wiring without needing actual network access to SMAX/calendar/Jabber.

How this gets invoked: AgentCore Gateway calls this Lambda whenever an
agent calls one of the three MCP tools we'll register against it
(create_ticket, get_calendar_events, send_jabber_message). Gateway tells
this function *which* tool was called via `context.client_context.custom`
and passes the tool's arguments as `event` (a plain dict matching whatever
input schema we define for that tool in the Gateway Target).

Swap the body of each `if tool_name == ...` branch for a real API call
(SMAX REST API, Exchange/Google Calendar API, Jabber/XMPP client) once
this is deployed somewhere with real access to those systems — the
Gateway wiring and the agent code calling it never have to change.
"""
import uuid

# Fake in-memory "databases". Reset every time Lambda cold-starts — this
# is a mock for testing the wiring, not real persistent storage.
_TICKETS = {}
_CALENDAR = {
    "2026-07-14": ["9:00 AM - Standup", "1:00 PM - Hull Section QA Review"],
    "2026-07-15": ["10:00 AM - Safety Walkthrough"],
}


def lambda_handler(event, context):
    delimiter = "___"
    original_tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
    tool_name = original_tool_name[original_tool_name.index(delimiter) + len(delimiter):]

    if tool_name == "create_ticket":
        ticket_id = f"SMAX-{uuid.uuid4().hex[:6].upper()}"
        _TICKETS[ticket_id] = event
        return {"ticket_id": ticket_id, "status": "created", "details": event}

    if tool_name == "get_calendar_events":
        date = event.get("date")
        events = _CALENDAR.get(date, [])
        return {"date": date, "events": events}

    if tool_name == "send_jabber_message":
        return {
            "status": "sent",
            "recipient": event.get("recipient"),
            "message": event.get("message"),
        }

    return {"error": f"Unknown tool: {tool_name}"}