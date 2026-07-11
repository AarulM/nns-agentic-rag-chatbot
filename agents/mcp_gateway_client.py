"""
Thin wrapper around the AgentCore Gateway MCP endpoint.

In real deployment: AgentCore Gateway exposes SMAX, calendar, and Jabber as
MCP tools (each backed by a Lambda function you write that calls the real
SMAX/Exchange/Jabber APIs). Agents don't call SMAX/Jabber directly — they
call the Gateway's single MCP endpoint, which routes to the right Lambda
and injects the caller's own auth token (AgentCore Identity, OBO exchange),
so a ticket/message is created as *that employee*, not a shared bot account.

This file is a stand-in so hr/safety/operations agents have something to
import while you build the real Gateway + Lambda targets. Replace the
bodies with an MCP client call once your Gateway is stood up, e.g.:

    from mcp.client.streamable_http import streamablehttp_client
    from strands.tools.mcp import MCPClient

    gateway = MCPClient(lambda: streamablehttp_client(
        "https://<your-gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp",
        headers={"Authorization": f"Bearer {user_access_token}"},
    ))
    tools = gateway.list_tools_sync()   # pass these into your Strands Agent(tools=...)
"""


def create_smax_ticket(category: str, summary: str, details: dict) -> str:
    # TODO: call the Gateway MCP tool that wraps the SMAX "create ticket" API
    print(f"[stub] SMAX ticket created: {category=} {summary=} {details=}")
    return "SMAX-000001"


def get_calendar_events(date: str) -> list[str]:
    # TODO: call the Gateway MCP tool that wraps the calendar "list events" API
    print(f"[stub] fetching calendar events for {date}")
    return []


def send_jabber_message(recipient: str, message: str) -> None:
    # TODO: call the Gateway MCP tool that wraps the Jabber/XMPP send-message API
    print(f"[stub] Jabber -> {recipient}: {message}")