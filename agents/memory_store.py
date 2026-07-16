"""
Long-term memory for the supervisor agent, built on Strands' MemoryManager
(strands.memory) with DynamoDB as the backend — AgentCore Memory is not
available in AWS GovCloud, and DynamoDB is everywhere.

UserFactStore implements the Strands MemoryStore protocol (search + add).
Facts live in the same DynamoDB table the short-term hook uses, under a
FACT#<actor> partition with no expiry, so they survive across sessions and
app restarts. With extraction=True, Strands automatically distills durable
facts ("user's name is Aarul", "works in Operations") from the conversation
every 5 turns, on a background task — the agent never blocks on it and no
extra tool call is needed to save them. The MemoryManager also injects the
most relevant facts into the model input on each user turn and registers a
search_memory tool for on-demand recall.
"""
import re
import time

import boto3
from boto3.dynamodb.conditions import Key
from strands.memory import ExtractionConfig, ModelExtractor
from strands.memory.types import MemoryEntry, SearchOptions

from aws_config import REGION

# The default extraction distills facts from the WHOLE transcript,
# including the assistant's replies. That poisoned memory in testing: when
# the model hallucinated an answer ("your supervisor is John Smith"),
# extraction saved the hallucination as a real fact, and injection then fed
# it back on later turns. A prompt-only guard ("never extract what the
# assistant said") did NOT hold with llama3.1:8b as the extractor, so
# _UserOnlyExtractor below drops assistant messages structurally — the
# extractor never even sees them. The JSON-array output contract must match
# the default prompt's, because ModelExtractor parses the response as JSON.
_EXTRACTION_PROMPT = (
    "You extract durable facts about the user from their chat messages, to "
    "be remembered across future conversations: their name, role, "
    "department, badge number, supervisor, preferences, and similar stable "
    "details. Ignore questions, small talk, and anything transient.\n"
    "\n"
    "State each fact in the third person, referring to the user as \"the "
    "user\" (e.g. \"The user's badge number is 40213\"). Never quote the "
    "user verbatim — first-person quotes injected back into later chats "
    "read like dialogue and confuse the model.\n"
    "\n"
    'Return ONLY a JSON array of objects, each: {"content": string}. Each '
    "object is one discrete, self-contained fact. If there is nothing worth "
    "remembering, return []."
)


class _UserOnlyExtractor(ModelExtractor):
    """Extracts facts from the user's messages only.

    Assistant text never reaches the extractor, so an assistant
    hallucination can never become a stored fact — guaranteed by structure,
    not by hoping the extraction model follows its prompt.
    """

    async def extract(self, messages, context=None):
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            return []
        return await super().extract(user_messages, context)


class UserFactStore:
    name = "user_facts"
    description = (
        "Durable facts about the employee you are talking to: their name, "
        "role, department, preferences, and past requests."
    )
    max_search_results = 5
    writable = True
    # Distill user-stated facts from the conversation in the background,
    # using the agent's own model (default trigger: every 5 turns, but the
    # synchronous agent() call flushes after each invocation, so in practice
    # facts land every turn).
    extraction = ExtractionConfig(extractor=_UserOnlyExtractor(system_prompt=_EXTRACTION_PROMPT))

    def __init__(self, table_name: str, actor_id: str):
        self._pk = f"FACT#{actor_id}"
        self._table = boto3.resource("dynamodb", region_name=REGION).Table(table_name)

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        limit = (options or {}).get("max_search_results") or self.max_search_results
        response = self._table.query(
            KeyConditionExpression=Key("PK").eq(self._pk),
            ScanIndexForward=False,  # newest facts first
            Limit=200,
        )
        facts = [item["text"] for item in response.get("Items", [])]
        # Simple keyword-overlap ranking — no embeddings or vector store to
        # set up, which keeps this GovCloud-portable. A user accumulates few
        # enough facts that this is plenty; ties keep newest-first order, so
        # with no keyword match the most recent facts are returned.
        # re.findall (not .split()) so "supervisor?" still matches
        # "supervisor" — trailing punctuation broke matching in testing.
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        ranked = sorted(
            facts,
            key=lambda fact: sum(w in fact.lower() for w in words),
            reverse=True,
        )
        return [MemoryEntry(content=fact) for fact in ranked[:limit]]

    async def add(self, content: str, metadata: dict | None = None) -> None:
        # Extraction retries failed batches and re-derives similar facts on
        # later turns, so skip exact duplicates instead of stacking them —
        # they'd crowd out other facts in the 5-entry injection budget.
        existing = await self.search(content, {"max_search_results": 200})
        if any(entry.content == content for entry in existing):
            return
        # No expires_at attribute: the table's TTL only deletes short-term
        # conversation events; facts persist until manually removed.
        self._table.put_item(Item={
            "PK": self._pk,
            "SK": f"{time.time_ns():020d}",
            "text": content,
        })
