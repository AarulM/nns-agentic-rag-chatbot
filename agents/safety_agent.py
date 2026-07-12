"""
Safety Agent — answers safety-procedure questions and can file a SMAX
safety ticket via the MCP gateway (see mcp_gateway_client.py).
"""
from strands import Agent, tool
from knowledge_base import search_docs
from mcp_gateway_client import create_smax_ticket
from model_config import get_model
from trace_log import tracing_callback_handler


@tool
def search_safety_docs(query: str) -> str:
    """Search shipyard safety procedures and OSHA-related company policy."""
    return search_docs(query, "No matching safety documents found.")


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
    callback_handler=tracing_callback_handler,
    system_prompt=(
        "You are the Safety assistant for a shipbuilding company. Only use "
        "report_safety_incident when the user clearly wants to file a report, "
        "and confirm the details back to them before/after filing.\n\n"
        "Use search_safety_docs ONLY for questions about this company's own "
        "safety rules: required PPE for a specific area or task, site "
        "procedures, incident reporting steps. For general knowledge "
        "questions (what a term or acronym means, how something works in "
        "general), answer directly from your own knowledge without calling "
        "any tool.\n\n"
        "After search_safety_docs returns results, check whether they "
        "actually answer the specific question asked. If they don't, say the "
        "company documentation doesn't cover it — then, if you can, still "
        "help from your own general knowledge, clearly labeled as general "
        "safety guidance rather than company policy. Never present unrelated "
        "search results as the answer, and never invent company rules."
    ),
    tools=[search_safety_docs, report_safety_incident],
)

if __name__ == "__main__":
    print(safety_agent("What PPE is required in the welding bay?"))