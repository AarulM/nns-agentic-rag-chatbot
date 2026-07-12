"""
Fixed continuation of reconnect_gateway.py — waits for the target
deletion to actually finish propagating (not just for the gateway to say
READY) before creating the replacement target.

Run: python finish_reconnect_gateway.py
"""
import time
import boto3

REGION = "us-east-1"
GATEWAY_ID = "nnscompanytoolsgateway-qcngunnqhe"
NEW_LAMBDA_ARN = "arn:aws:lambda:us-east-1:465733921455:function:NnsAgenticRagChatbotStack-McpToolsFunctionBED47FDB-ppHb8U8I29S8"
TARGET_NAME = "MockCompanyToolsLambda"

gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)


def wait_for_gateway_ready():
    for _ in range(24):
        status = gateway_client.get_gateway(gatewayIdentifier=GATEWAY_ID)["status"]
        if status == "READY":
            return
        print(f"Gateway status: {status}, waiting...")
        time.sleep(5)
    raise TimeoutError("Gateway never returned to READY.")


def delete_existing_targets():
    targets = gateway_client.list_gateway_targets(gatewayIdentifier=GATEWAY_ID).get("items", [])
    existing_by_name = {t["name"]: t for t in targets if t["name"] == TARGET_NAME}
    if not existing_by_name:
        print("No existing target with that name — nothing to delete.")
        return

    for t in existing_by_name.values():
        print(f"Deleting target: {t['name']} ({t['targetId']})")
        gateway_client.delete_gateway_target(gatewayIdentifier=GATEWAY_ID, targetId=t["targetId"])

    # Actually wait for it to be gone, not just for the gateway to say READY.
    for attempt in range(24):
        remaining = gateway_client.list_gateway_targets(gatewayIdentifier=GATEWAY_ID).get("items", [])
        if not any(t["name"] == TARGET_NAME for t in remaining):
            print("Confirmed old target is gone.")
            return
        print(f"Target still present (attempt {attempt + 1}), waiting...")
        time.sleep(5)
    raise TimeoutError("Old target never actually disappeared after 2 minutes.")


def create_new_target():
    target_config = {
        "mcp": {
            "lambda": {
                "lambdaArn": NEW_LAMBDA_ARN,
                "toolSchema": {
                    "inlinePayload": [
                        {
                            "name": "create_ticket",
                            "description": "Create a SMAX ticket (safety incident or operations issue).",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string"},
                                    "summary": {"type": "string"},
                                },
                                "required": ["category", "summary"],
                            },
                        },
                        {
                            "name": "get_calendar_events",
                            "description": "Get calendar events for a given date (YYYY-MM-DD).",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"date": {"type": "string"}},
                                "required": ["date"],
                            },
                        },
                        {
                            "name": "send_jabber_message",
                            "description": "Send a Jabber message to a coworker or channel.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "recipient": {"type": "string"},
                                    "message": {"type": "string"},
                                },
                                "required": ["recipient", "message"],
                            },
                        },
                    ]
                },
            }
        }
    }
    credential_config = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
    gateway_client.create_gateway_target(
        gatewayIdentifier=GATEWAY_ID,
        name=TARGET_NAME,
        description="Mock SMAX/calendar/Jabber tools",
        targetConfiguration=target_config,
        credentialProviderConfigurations=credential_config,
    )
    print("Created new target pointing at the new Lambda.")


def main():
    print("Waiting for Gateway to be READY...")
    wait_for_gateway_ready()
    delete_existing_targets()
    wait_for_gateway_ready()
    create_new_target()
    print("\nDone. Gateway URL and Cognito credentials are unchanged from before.")


if __name__ == "__main__":
    main()