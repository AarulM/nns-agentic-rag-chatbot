"""
Shared Bedrock Knowledge Base retrieval used by all three specialists.
Each agent keeps its own @tool wrapper (the tool name and docstring are
what the LLM routes on), but the actual retrieve call lives here once.
"""
import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_config import REGION, KNOWLEDGE_BASE_ID

logger = logging.getLogger("nns.knowledge_base")

_bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)

# Returned when the Knowledge Base itself is unreachable, as opposed to
# reachable-but-empty.
#
# This has to be a definitive, final-sounding sentence. When retrieval
# raised, llama3.1:8b saw a tool error, assumed it had called the tool
# wrong, and retried — producing the ask_safety → search_safety_docs →
# ask_safety → search_safety_docs loop that looks like a hung app. A calm
# statement that there is nothing to find gives the model somewhere to
# stop.
_UNAVAILABLE = (
    "The document library is not available right now, so there is nothing "
    "to search. Tell the user you cannot look this up at the moment and "
    "suggest they try again later. Do not retry this search."
)


def search_docs(query: str, not_found_message: str) -> str:
    if not KNOWLEDGE_BASE_ID:
        logger.warning("KNOWLEDGE_BASE_ID is not set — retrieval skipped.")
        return _UNAVAILABLE

    try:
        response = _bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
        )
    except (ClientError, BotoCoreError) as error:
        # Logged for the operator, summarized for the model. The agent must
        # never see a stack trace or an AWS error code — it will either
        # retry forever or read the error out loud to the user.
        logger.warning("Knowledge Base retrieval failed: %s", error)
        return _UNAVAILABLE

    chunks = [r["content"]["text"] for r in response.get("retrievalResults", [])]
    return "\n---\n".join(chunks) if chunks else not_found_message
