"""
Creates (or finds) the DynamoDB table that backs the chatbot's memory:
short-term conversation history AND long-term extracted facts, in one table.

Why DynamoDB and not AgentCore Memory: AgentCore Memory is not available in
AWS GovCloud (see
https://docs.aws.amazon.com/govcloud-us/latest/UserGuide/govcloud-bedrock-agentcore.html),
so memory lives in a plain DynamoDB table instead — available in every
region, including GovCloud. Safe to re-run — if the table already exists,
it just says so.

The table name is fixed (no random suffix), so unlike the old AgentCore
version there is nothing to paste into agents/aws_config.py — the default
there already matches.

Run: python create_memory.py
"""
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
MEMORY_TABLE = os.environ.get("MEMORY_TABLE", "NnsChatbotMemory")


def main():
    client = boto3.client("dynamodb", region_name=REGION)

    try:
        client.describe_table(TableName=MEMORY_TABLE)
        print(f"Table already exists.\n\nMEMORY_TABLE = \"{MEMORY_TABLE}\"")
        print("\nMatches the default in agents/aws_config.py — nothing to paste.")
        return
    except client.exceptions.ResourceNotFoundException:
        pass

    print("Creating DynamoDB memory table (on-demand billing — costs nothing at chat scale)...")
    client.create_table(
        TableName=MEMORY_TABLE,
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    client.get_waiter("table_exists").wait(TableName=MEMORY_TABLE)

    # Conversation events carry an expires_at timestamp and vanish after 7
    # days (same as the old AgentCore event_expiry_days=7). Long-term facts
    # are written without expires_at, so TTL never touches them.
    client.update_time_to_live(
        TableName=MEMORY_TABLE,
        TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
    )

    print(f"\nMEMORY_TABLE = \"{MEMORY_TABLE}\"")
    print("\nMatches the default in agents/aws_config.py — nothing to paste.")


if __name__ == "__main__":
    main()
