"""
Recovery script — picks up where setup_gateway.py crashed. It:
  1. Looks up the Cognito M2M client we already created (so we don't
     create a duplicate).
  2. Waits for the Gateway (already created) to leave "CREATING" status.
  3. Attaches the Lambda target (the step that failed last time).
  4. Prints all 5 values you need for mcp_gateway_client.py.

Run once: python finish_gateway_setup.py
"""
import time
import boto3

REGION = "us-east-1"

# Known from your last run's output — already exist, don't recreate.
USER_POOL_ID = "us-east-1_dcNwgvSya"
COGNITO_DOMAIN = "nns-agentcore-dcnwgvsya"
GATEWAY_ID = "nnscompanytoolsgateway-omj3vt66ow"
GATEWAY_URL = "https://nnscompanytoolsgateway-omj3vt66ow.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"

LAMBDA_ARN = "arn:aws:lambda:us-east-1:465733921455:function:NnsAgenticRagChatbotStack-McpToolsFunctionBED47FDB-rL70B1jiBYTe"
CLIENT_NAME = "nns-agentcore-gateway-client"

cognito = boto3.client("cognito-idp", region_name=REGION)
gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)


def get_existing_m2m_client():
    for c in cognito.list_user_pool_clients(UserPoolId=USER_POOL_ID, MaxResults=60)["UserPoolClients"]:
        if c["ClientName"] == CLIENT_NAME:
            details = cognito.describe_user_pool_client(UserPoolId=USER_POOL_ID, ClientId=c["ClientId"])
            return c["ClientId"], details["UserPoolClient"]["ClientSecret"]
    raise RuntimeError("Couldn't find the M2M client — did setup_gateway.py get further than expected?")


def wait_for_gateway_ready():
    for attempt in range(24):  # up to ~2 minutes
        response = gateway_client.get_gateway(gatewayIdentifier=GATEWAY_ID)
        status = response.get("status", "UNKNOWN")
        print(f"Gateway status: {status}")
        if status != "CREATING":
            if status not in ("READY", "ACTIVE"):
                print("Full response for debugging:", response)
            return status
        time.sleep(5)
    raise TimeoutError("Gateway never left CREATING status after 2 minutes — paste this output back.")


def create_gateway_target():
    target_config = {
        "mcp": {
            "lambda": {
                "lambdaArn": LAMBDA_ARN,
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
    return gateway_client.create_gateway_target(
        gatewayIdentifier=GATEWAY_ID,
        name="MockCompanyToolsLambda",
        description="Mock SMAX/calendar/Jabber tools",
        targetConfiguration=target_config,
        credentialProviderConfigurations=credential_config,
    )


def main():
    print("Looking up existing Cognito M2M client...")
    client_id, client_secret = get_existing_m2m_client()
    print(f"Client ID: {client_id}")

    print("Waiting for Gateway to finish creating...")
    final_status = wait_for_gateway_ready()
    print(f"Gateway reached status: {final_status}")

    if final_status in ("READY", "ACTIVE"):
        print("Attaching Lambda target...")
        create_gateway_target()
        print("Done.")
    else:
        print(f"Gateway status is '{final_status}', not READY/ACTIVE — stopping before creating the target.")
        print("Paste this output back so we can see what went wrong.")
        return

    print("\n--- SAVE ALL OF THIS — you'll need it for mcp_gateway_client.py ---")
    print(f"GATEWAY_URL={GATEWAY_URL}")
    print(f"COGNITO_DOMAIN={COGNITO_DOMAIN}")
    print(f"COGNITO_CLIENT_ID={client_id}")
    print(f"COGNITO_CLIENT_SECRET={client_secret}")
    print(f"COGNITO_USER_POOL_ID={USER_POOL_ID}")


if __name__ == "__main__":
    main()