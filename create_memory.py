"""
Creates (or finds) the AgentCore short-term Memory resource the supervisor
uses to remember conversation history. Safe to re-run — if a memory with
this name already exists, it just prints the existing ID.

Paste the printed MEMORY_ID into agents/aws_config.py.

Run: python create_memory.py
"""
from bedrock_agentcore.memory import MemoryClient

REGION = "us-east-1"
MEMORY_NAME = "NnsSupervisorShortTermMemory"


def main():
    client = MemoryClient(region_name=REGION)

    # Memory IDs are the name plus a random suffix (Name-abc123...).
    for m in client.list_memories():
        memory_id = str(m.get("id", ""))
        if memory_id.startswith(f"{MEMORY_NAME}-"):
            print(f"Memory already exists.\n\nMEMORY_ID = \"{memory_id}\"")
            print("\nPaste into agents/aws_config.py if it isn't there already.")
            return

    print("Creating AgentCore Memory (short-term only, no LLM extraction, no extra model cost)...")
    memory = client.create_memory_and_wait(
        name=MEMORY_NAME,
        strategies=[],  # empty = short-term memory only
        description="Short-term conversation memory for the NNS chatbot supervisor agent",
        event_expiry_days=7,  # how long raw conversation history is kept, up to 365
    )
    print(f"\nMEMORY_ID = \"{memory['id']}\"")
    print("\nPaste into agents/aws_config.py.")


if __name__ == "__main__":
    main()
