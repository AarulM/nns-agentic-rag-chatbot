"""
HR Agent — answers onboarding/benefits/policy questions using a Bedrock
Knowledge Base as its RAG tool. Same pattern reused for safety_agent.py
and operations_agent.py, just swap the KB id, prompt, and tools.
"""
import boto3
from strands import Agent, tool
from model_config import get_model

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

HR_KNOWLEDGE_BASE_ID = "OSOGWWRI0X"


@tool
def search_hr_docs(query: str) -> str:
    """Search HR policies, benefits, and onboarding documents for an answer."""
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=HR_KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
    )
    chunks = [r["content"]["text"] for r in response.get("retrievalResults", [])]
    return "\n---\n".join(chunks) if chunks else "No matching HR documents found."


hr_agent = Agent(
    name="hr_agent",
    model=get_model(),
    system_prompt=(
        "You are the HR assistant for a shipbuilding company. Answer questions "
        "about onboarding, benefits, PTO, and company HR policy using the "
        "search_hr_docs tool. If the answer isn't in company documents, say so "
        "plainly instead of guessing. Never share another employee's personal data."
    ),
    tools=[search_hr_docs],
)

if __name__ == "__main__":
    print(hr_agent("What do I need to do in my first week as a new hire?"))