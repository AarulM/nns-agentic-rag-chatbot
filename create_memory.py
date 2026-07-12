"""
One-time setup script: creates an AgentCore short-term Memory resource for
the supervisor agent to remember conversation history. Run once with your
existing aarul-dev AWS credentials — no CDK, no new IAM role needed for
local testing (your IAM user already has admin access).

Run: python create_memory.py
"""
import os
from bedrock_agentcore.memory import MemoryClient

REGION = "us-east-1"


def main():
    client = MemoryClient(region_name=REGION)

    print("Creating AgentCore Memory (short-term only, no LLM extraction, no extra model cost)...")
    memory = client.create_memory_and_wait(
        name="NnsSupervisorShortTermMemory",
        strategies=[],  # empty = short-term memory only
        description="Short-term conversation memory for the NNS chatbot supervisor agent",
        event_expiry_days=7,  # how long raw conversation history is kept, up to 365
    )
    print(f"\nMEMORY_ID={memory['id']}")
    print("\nSave this — you'll paste it into agents/memory_hook.py next.")


if __name__ == "__main__":
    main()