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
        "You are the HR assistant for a shipbuilding company. Answer questions "
        "about onboarding, benefits, PTO, and company HR policy using the "
        "search_hr_docs tool. If the answer isn't in company documents, say so "
        "plainly instead of guessing. Never share another employee's personal data.\n\n"
        "After search_hr_docs returns results, check whether they actually "
        "answer the specific question asked before using them. If the results "
        "are about a different topic, say plainly that you couldn't find "
        "documentation on that specific question rather than presenting "
        "unrelated content as the answer."
    ),
    tools=[search_hr_docs],
)

if __name__ == "__main__":
    print(hr_agent("What do I need to do in my first week as a new hire?"))