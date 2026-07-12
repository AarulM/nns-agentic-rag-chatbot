"""
Short-term memory hook for the supervisor agent, backed by AgentCore
Memory. Lets the chatbot remember earlier turns in the same conversation
(the user's name, what they already asked, etc.) without you writing any
storage code yourself.

Strands calls these two functions automatically at specific points in an
agent's lifecycle ("hooks") — you never call them directly:
  - on_agent_initialized: runs once when the Agent object is created;
    loads recent history and adds it to the system prompt.
  - on_message_added: runs every time a new message (user or assistant)
    is added to the conversation; saves it to Memory.
"""
from bedrock_agentcore.memory import MemoryClient
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AgentInitializedEvent, MessageAddedEvent
from aws_config import REGION

memory_client = MemoryClient(region_name=REGION)


class ShortTermMemoryHookProvider(HookProvider):
    def __init__(self, memory_client: MemoryClient, memory_id: str):
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
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=actor_id,
                session_id=session_id,
                messages=[(last["content"][0]["text"], last["role"])],
            )

    def on_agent_initialized(self, event: AgentInitializedEvent):
        actor_id = event.agent.state.get("actor_id")
        session_id = event.agent.state.get("session_id")
        recent_turns = self.memory_client.get_last_k_turns(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            k=5,
        )
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