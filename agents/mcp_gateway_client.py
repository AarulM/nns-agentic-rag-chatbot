"""
Real MCP client wired to the AgentCore Gateway created by setup_gateway.py
+ finish_gateway_setup.py. Same three function names as the old mock
stub file (create_smax_ticket, get_calendar_events, send_jabber_message)
so safety_agent.py and operations_agent.py don't need to change at all —
only what happens inside these functions changed, from an in-memory fake
to a real call through the Gateway to our mock Lambda.

The Cognito client secret is the one real credential here, so it lives
outside source control: either export COGNITO_CLIENT_SECRET, or put it in
the gitignored agents/gateway_secrets.py (value comes from
setup_gateway.py's final printed output).
"""
import os
import time
import uuid
import requests
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client

# From setup_gateway.py's final printed output.
GATEWAY_URL = "https://nnscompanytoolsgateway-omj3vt66ow.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
COGNITO_DOMAIN = "nns-agentcore-dcnwgvsya"
COGNITO_CLIENT_ID = "23ddablie1urm7mov53i1ltdba"
REGION = "us-east-1"

COGNITO_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET")
if not COGNITO_CLIENT_SECRET:
    try:
        from gateway_secrets import COGNITO_CLIENT_SECRET
    except ImportError:
        raise RuntimeError(
            "No Cognito client secret found. Export COGNITO_CLIENT_SECRET or "
            "create agents/gateway_secrets.py with the value printed by "
            "setup_gateway.py."
        )
SCOPE_STRING = "nns-agentcore-gateway-id/gateway:read nns-agentcore-gateway-id/gateway:write"

_cached_token = None
_cached_token_expiry = 0.0


def _get_access_token() -> str:
    """Gets a fresh Cognito access token, reusing the cached one until it's
    close to expiring (client-credentials tokens usually last ~1 hour)."""
    global _cached_token, _cached_token_expiry
    if _cached_token and time.time() < _cached_token_expiry:
        return _cached_token

    token_url = f"https://{COGNITO_DOMAIN}.auth.{REGION}.amazoncognito.com/oauth2/token"
    response = requests.post(
        token_url,
        data={"grant_type": "client_credentials", "scope": SCOPE_STRING},
        auth=(COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    response.raise_for_status()
    payload = response.json()
    _cached_token = payload["access_token"]
    _cached_token_expiry = time.time() + payload.get("expires_in", 3600) - 60
    return _cached_token


def _call_tool(tool_name_suffix: str, arguments: dict) -> str:
    """Opens a short-lived connection to the Gateway, finds the tool whose
    name ends with tool_name_suffix (AgentCore prefixes each tool's name
    with the Gateway Target's name, e.g. "MockCompanyToolsLambda...create_ticket"),
    calls it, and returns any text result."""
    token = _get_access_token()
    client = MCPClient(lambda: streamablehttp_client(GATEWAY_URL, headers={"Authorization": f"Bearer {token}"}))

    with client:
        tools = client.list_tools_sync()
        match = next((t.tool_name for t in tools if t.tool_name.endswith(tool_name_suffix)), None)
        if match is None:
            available = [t.tool_name for t in tools]
            raise RuntimeError(f"No tool ending in '{tool_name_suffix}' found. Available tools: {available}")

        result = client.call_tool_sync(
            tool_use_id=str(uuid.uuid4()),
            name=match,
            arguments=arguments,
        )

        # Result shape can vary slightly by SDK version — handle both a
        # dict with a "content" list, or an object with a .content attribute.
        content_blocks = result.get("content", []) if isinstance(result, dict) else getattr(result, "content", [])
        texts = [block.get("text") for block in content_blocks if isinstance(block, dict) and "text" in block]
        return "\n".join(t for t in texts if t) or str(result)


def create_smax_ticket(category: str, summary: str, details: dict) -> str:
    # Our registered tool schema only takes category + summary, so fold
    # any extra details (like "location" or "priority") into the summary text.
    extra = ", ".join(f"{k}={v}" for k, v in (details or {}).items())
    full_summary = f"{summary} ({extra})" if extra else summary
    return _call_tool("create_ticket", {"category": category, "summary": full_summary})


def get_calendar_events(date: str) -> list[str]:
    result_text = _call_tool("get_calendar_events", {"date": date})
    return [result_text] if result_text else []


def send_jabber_message(recipient: str, message: str) -> None:
    _call_tool("send_jabber_message", {"recipient": recipient, "message": message})