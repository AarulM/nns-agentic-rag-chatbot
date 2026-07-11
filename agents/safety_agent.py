"""
Safety Agent — answers safety-procedure questions and can file a SMAX
safety ticket via the MCP gateway (see mcp_gateway_client.py).
"""
import boto3
from strands import Agent, tool
from mcp_gateway_client import create_smax_ticket
from model_config import get_model

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

SAFETY_KNOWLEDGE_BASE_ID = "OSOGWWRI0X"


@tool
def search_safety_docs(query: str) -> str:
    """Search shipyard safety procedures and OSHA-related company policy."""
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=SAFETY_KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
    )
    chunks = [r["content"]["text"] for r in response.get("retrievalResults", [])]
    return "\n---\n".join(chunks) if chunks else "No matching safety documents found."


@tool
def report_safety_incident(description: str, location: str) -> str:
    """File a safety incident ticket in SMAX. Use only when the user explicitly
    asks to report/log an incident, not for general safety questions."""
    ticket_id = create_smax_ticket(
        category="safety_incident",
        summary=description,
        details={"location": location},
    )
    return f"Safety incident logged as SMAX ticket {ticket_id}."


safety_agent = Agent(
    name="safety_agent",
    model=get_model(),
    system_prompt=(
        "You are the Safety assistant for a shipbuilding company. Use "
        "search_safety_docs for procedure questions. Only use "
        "report_safety_incident when the user clearly wants to file a report, "
        "and confirm the details back to them before/after filing."
    ),
    tools=[search_safety_docs, report_safety_incident],
)

if __name__ == "__main__":
    print(safety_agent("What PPE is required in the welding bay?"))