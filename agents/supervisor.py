"""
Supervisor agent — the "agents as tools" pattern from Strands. This module
is what AgentCore Runtime would host as the entrypoint.

Run locally:  python supervisor.py
Deploy later: wrap `handle_request` with the AgentCore Runtime entrypoint
decorator per the samples in Build_Plan.md, then `agentcore launch`.
"""
from strands import Agent, tool
from hr_agent import hr_agent
from safety_agent import safety_agent
from operations_agent import operations_agent


@tool
def ask_hr(question: str) -> str:
    """Route a question to the HR agent (onboarding, benefits, HR policy)."""
    return str(hr_agent(question))


@tool
def ask_safety(question: str) -> str:
    """Route a question to the Safety agent (procedures, PPE, incident reporting)."""
    return str(safety_agent(question))


@tool
def ask_operations(question: str) -> str:
    """Route a question to the Operations agent (SOPs, scheduling, tickets, Jabber)."""
    return str(operations_agent(question))


supervisor = Agent(
    name="supervisor",
    system_prompt=(
        "You are the front-door assistant for a shipbuilding company's internal "
        "chatbot, used by employees, new hires, and interns. Decide which "
        "specialist to call: ask_hr for onboarding/benefits/HR policy, "
        "ask_safety for safety procedures/incidents, ask_operations for SOPs/ "
        "scheduling/tickets/messaging. If the question spans more than one "
        "area, call more than one tool and combine the answers. If it's a "
        "general company question none of the specialists can answer, say so "
        "honestly."
    ),
    tools=[ask_hr, ask_safety, ask_operations],
)


def handle_request(user_message: str) -> str:
    return str(supervisor(user_message))


if __name__ == "__main__":
    print(handle_request("What PPE do I need for the welding bay, and what's on my first-week onboarding checklist?"))