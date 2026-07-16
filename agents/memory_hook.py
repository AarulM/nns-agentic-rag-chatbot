"""
Short-term memory hook for the supervisor agent, backed by a DynamoDB
table. Lets the chatbot remember earlier turns in the same conversation
(the user's name, what they already asked, etc.) without you writing any
storage code yourself.

This used to be backed by AgentCore Memory, but that service is not
available in AWS GovCloud (see
https://docs.aws.amazon.com/govcloud-us/latest/UserGuide/govcloud-bedrock-agentcore.html),
so DynamoDBMemoryClient below is a drop-in replacement exposing the same
two calls the hook needs: create_event (save a turn) and get_last_k_turns
(reload recent history). Everything else — the hook, the background write
queue — is unchanged.

Strands calls these two functions automatically at specific points in an
agent's lifecycle ("hooks") — you never call them directly:
  - on_agent_initialized: runs once when the Agent object is created;
    loads recent history and adds it to the conversation.
  - on_message_added: runs every time a new message (user or assistant)
    is added to the conversation; saves it to Memory.
"""
import queue
import threading
import time

import boto3
from boto3.dynamodb.conditions import Key
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AgentInitializedEvent, MessageAddedEvent
from aws_config import REGION

# Conversation events expire after 7 days via the table's TTL, matching the
# old AgentCore event_expiry_days=7.
_EVENT_EXPIRY_SECONDS = 7 * 24 * 3600


class DynamoDBMemoryClient:
    """Drop-in replacement for AgentCore's MemoryClient, backed by DynamoDB.

    memory_id is the DynamoDB table name (MEMORY_TABLE in aws_config.py).
    Each message is one item under a SESSION#actor#session partition, sorted
    by a nanosecond timestamp so history reads back in order.
    """

    def __init__(self, region_name: str):
        self._dynamodb = boto3.resource("dynamodb", region_name=region_name)

    def create_event(self, memory_id: str, actor_id: str, session_id: str,
                     messages: list[tuple[str, str]]):
        table = self._dynamodb.Table(memory_id)
        for text, role in messages:
            table.put_item(Item={
                "PK": f"SESSION#{actor_id}#{session_id}",
                "SK": f"{time.time_ns():020d}",
                "role": role,
                "text": text,
                "expires_at": int(time.time()) + _EVENT_EXPIRY_SECONDS,
            })

    def get_last_k_turns(self, memory_id: str, actor_id: str, session_id: str,
                         k: int) -> list[list[dict]]:
        """Returns the most recent messages, newest-first, in the same shape
        AgentCore returned: a list of turns, each a list of
        {"role": ..., "content": {"text": ...}} messages (here, one message
        per turn). A turn is user + assistant, so k turns = 2*k messages.
        """
        table = self._dynamodb.Table(memory_id)
        response = table.query(
            KeyConditionExpression=Key("PK").eq(f"SESSION#{actor_id}#{session_id}"),
            ScanIndexForward=False,  # newest first
            Limit=2 * k,
        )
        return [
            [{"role": item["role"], "content": {"text": item["text"]}}]
            for item in response.get("Items", [])
        ]


memory_client = DynamoDBMemoryClient(region_name=REGION)

# Memory writes happen on every message and each one is an AWS API call
# (~200-300ms). A single background worker drains this queue so the chat
# never blocks on them, while still writing in conversation order.
_write_queue: "queue.Queue[dict]" = queue.Queue()


def _write_worker():
    while True:
        kwargs = _write_queue.get()
        try:
            memory_client.create_event(**kwargs)
        except Exception as e:
            print(f"WARNING: memory write failed ({e}); continuing without it.")
        finally:
            _write_queue.task_done()


threading.Thread(target=_write_worker, daemon=True).start()


class ShortTermMemoryHookProvider(HookProvider):
    def __init__(self, memory_client: DynamoDBMemoryClient, memory_id: str):
        self.memory_client = memory_client
        self.memory_id = memory_id

    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(MessageAddedEvent, self.on_message_added)
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)

    def on_message_added(self, event: MessageAddedEvent):
        actor_id = event.agent.state.get("actor_id")
        session_id = event.agent.state.get("session_id")
        last = event.agent.messages[-1]
        if last["content"] and last["content"][0].get("text"):
            _write_queue.put(dict(
                memory_id=self.memory_id,
                actor_id=actor_id,
                session_id=session_id,
                messages=[(last["content"][0]["text"], last["role"])],
            ))

    def on_agent_initialized(self, event: AgentInitializedEvent):
        actor_id = event.agent.state.get("actor_id")
        session_id = event.agent.state.get("session_id")
        # This runs at Agent construction, i.e. at app startup. If memory is
        # unreachable (table not created yet, missing IAM permissions, bad
        # credentials), start with an empty history instead of crashing the
        # whole app — memory is a nice-to-have, answering questions isn't.
        try:
            recent_turns = self.memory_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=actor_id,
                session_id=session_id,
                k=5,
            )
        except Exception as e:
            print(
                f"WARNING: could not load conversation memory ({e}). "
                f"Starting without it — if the DynamoDB table '{self.memory_id}' "
                "doesn't exist yet, run: python create_memory.py"
            )
            return
        # Seed real conversation history (event.agent.messages) instead of
        # gluing raw "role: text" lines onto the system prompt. Small local
        # models handle actual prior turns fine, but get confused by a
        # dialogue transcript embedded in their instructions and start
        # parroting it back regardless of the current question — that's
        # what caused the "USER: ... ASSISTANT: ..." echoing bug.
        # get_last_k_turns returns newest-first; seed oldest-first so the
        # model sees the conversation in the order it actually happened.
        for turn in reversed(recent_turns):
            for message in turn:
                role = message["role"].lower()
                if role not in ("user", "assistant"):
                    continue
                event.agent.messages.append(
                    {"role": role, "content": [{"text": message["content"]["text"]}]}
                )
