"""
Shared Bedrock Knowledge Base retrieval used by all three specialists.
Each agent keeps its own @tool wrapper (the tool name and docstring are
what the LLM routes on), but the actual retrieve call lives here once.
"""
import boto3
from aws_config import REGION, KNOWLEDGE_BASE_ID

_bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


def search_docs(query: str, not_found_message: str) -> str:
    response = _bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
    )
    chunks = [r["content"]["text"] for r in response.get("retrievalResults", [])]
    return "\n---\n".join(chunks) if chunks else not_found_message
