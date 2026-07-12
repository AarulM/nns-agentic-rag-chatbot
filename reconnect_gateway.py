"""
Reconnect script — run this after any `cdk destroy` + `cdk deploy` cycle.

cdk destroy removes the Lambda function and its IAM role, but your
AgentCore Gateway (created separately via setup_gateway.py, not managed by
CDK) survives and keeps pointing at the old, now-deleted ARNs. This
script updates the Gateway to use the fresh role and re-creates the
Gateway Target pointing at the fresh Lambda. The Gateway URL and Cognito
credentials (domain/client id/secret/user pool id) do NOT change — only
the Lambda-facing wiring does.

Run: python reconnect_gateway.py
"""
import boto3

REGION = "us-east-1"
GATEWAY_ID = "nnscompanytoolsgateway-qcngunnqhe"

# Paste your latest `cdk deploy` outputs here each time you redeploy.
NEW_LAMBDA_ARN = "arn:aws:lambda:us-east-1:465733921455:function:NnsAgenticRagChatbotStack-McpToolsFunctionBED47FDB-ppHb8U8I29S8"
NEW_GATEWAY_ROLE_ARN = "arn:aws:iam::465733921455:role/NnsAgenticRagChatbotStack-GatewayExecutionRole16B5E-tItRYqMeSNna"

gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)


def update_gateway_role():
    gateway = gateway_client.get_gateway(gatewayIdentifier=GATEWAY_ID)
    print("Current gateway details:", gateway)
    gateway_client.update_gateway(
        gatewayIdentifier=GATEWAY_ID,
        name=gateway["name"],
        roleArn=NEW_GATEWAY_ROLE_ARN,
        protocolType=gateway["protocolType"],
        authorizerType=gateway["authorizerType"],
        authorizerConfiguration=gateway["authorizerConfiguration"],
    )
    print("Updated Gateway to use the new IAM role.")


def replace_gateway_target():
    targets = gateway_client.list_gateway_targets(gatewayIdentifier=GATEWAY_ID)
    print("Existing targets:", targets)
    for t in targets.get("items", []):
        print(f"Deleting old target: {t['name']} ({t['targetId']})")
        gateway_client.delete_gateway_target(gatewayIdentifier=GATEWAY_ID, targetId=t["targetId"])

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
        name="MockCompanyToolsLambda",
        description="Mock SMAX/calendar/Jabber tools",
        targetConfiguration=target_config,
        credentialProviderConfigurations=credential_config,
    )
    print("Created new target pointing at the new Lambda.")


def main():
    update_gateway_role()
    replace_gateway_target()
    print("\nDone. Gateway URL and Cognito credentials are unchanged from before.")


if __name__ == "__main__":
    main()