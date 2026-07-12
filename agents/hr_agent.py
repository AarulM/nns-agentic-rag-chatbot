"""
HR Agent — answers onboarding/benefits/policy questions using a Bedrock
Knowledge Base as its RAG tool. Same pattern reused for safety_agent.py
and operations_agent.py, just swap the KB id, prompt, and tools.
"""
from strands import Agent, tool
from knowledge_base import search_docs
from model_config import get_model
from trace_log import tracing_callback_handler


@tool
def search_hr_docs(query: str) -> str:
    """Search HR policies, benefits, and onboarding documents for an answer."""
    return search_docs(query, "No matching HR documents found.")


hr_agent = Agent(
    name="hr_agent",
    model=get_model(),
    callback_handler=tracing_callback_handler,
    system_prompt=(
        "You are the HR assistant for Newport News Shipbuilding, a "
        "Huntington Ingalls Industries shipyard. Write in a professional, "
        "welcoming tone. Never share another employee's personal data.\n\n"
        "Use search_hr_docs ONLY for questions about this company's own "
        "policies: onboarding steps, benefits, PTO rules, pay policy. For "
        "general knowledge questions (what a term means, how something works "
        "in general), answer directly from your own knowledge without "
        "calling any tool.\n\n"
        "After search_hr_docs returns results, check whether they actually "
        "answer the specific question asked. If they don't, say the company "
        "documentation doesn't cover it — then, if you can, still help from "
        "your own general knowledge, clearly labeled as general guidance "
        "rather than company policy. Never present unrelated search results "
        "as the answer, and never invent company policy."
    ),
    tools=[search_hr_docs],
)

if __name__ == "__main__":
    print(hr_agent("What do I need to do in my first week as a new hire?"))