"""
One-time setup script (run with `python setup_gateway.py`, not `cdk deploy`)
that creates:
  1. A Cognito User Pool set up for machine-to-machine (M2M) auth, since
     this is an agent script calling the Gateway, not a human logging in
     through a browser.
  2. An AgentCore Gateway that requires that Cognito auth to call it.
  3. A Gateway Target that registers our mock tools Lambda, exposing its
     three functions (create_ticket, get_calendar_events,
     send_jabber_message) as MCP tools any agent can discover and call.

Run this once. It prints out values at the end you'll need for
mcp_gateway_client.py in the next step — save them somewhere.

Requires: pip install boto3 requests
"""
import time
import boto3
import requests

REGION = "us-east-1"

# From your `cdk deploy` outputs — paste your own values here.
LAMBDA_ARN = "arn:aws:lambda:us-east-1:465733921455:function:NnsAgenticRagChatbotStack-McpToolsFunctionBED47FDB-fxeMHwrrxE5H"
GATEWAY_ROLE_ARN = "arn:aws:iam::465733921455:role/NnsAgenticRagChatbotStack-GatewayExecutionRole16B5E-rLLWKHPpumv0"

USER_POOL_NAME = "nns-agentcore-gateway-pool"
RESOURCE_SERVER_ID = "nns-agentcore-gateway-id"
RESOURCE_SERVER_NAME = "nns-agentcore-gateway-name"
CLIENT_NAME = "nns-agentcore-gateway-client"
SCOPES = [
    {"ScopeName": "gateway:read", "ScopeDescription": "Read access"},
    {"ScopeName": "gateway:write", "ScopeDescription": "Write access"},
]
SCOPE_STRING = f"{RESOURCE_SERVER_ID}/gateway:read {RESOURCE_SERVER_ID}/gateway:write"

cognito = boto3.client("cognito-idp", region_name=REGION)
gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)


def get_or_create_user_pool():
    for pool in cognito.list_user_pools(MaxResults=60)["UserPools"]:
        if pool["Name"] == USER_POOL_NAME:
            return pool["Id"]
    created = cognito.create_user_pool(PoolName=USER_POOL_NAME)
    return created["UserPool"]["Id"]


def ensure_resource_server(user_pool_id):
    try:
        cognito.describe_resource_server(UserPoolId=user_pool_id, Identifier=RESOURCE_SERVER_ID)
    except cognito.exceptions.ResourceNotFoundException:
        cognito.create_resource_server(
            UserPoolId=user_pool_id,
            Identifier=RESOURCE_SERVER_ID,
            Name=RESOURCE_SERVER_NAME,
            Scopes=SCOPES,
        )


def get_or_create_m2m_client(user_pool_id):
    for c in cognito.list_user_pool_clients(UserPoolId=user_pool_id, MaxResults=60)["UserPoolClients"]:
        if c["ClientName"] == CLIENT_NAME:
            details = cognito.describe_user_pool_client(UserPoolId=user_pool_id, ClientId=c["ClientId"])
            return c["ClientId"], details["UserPoolClient"]["ClientSecret"]

    created = cognito.create_user_pool_client(
        UserPoolId=user_pool_id,
        ClientName=CLIENT_NAME,
        GenerateSecret=True,
        AllowedOAuthFlows=["client_credentials"],
        AllowedOAuthScopes=[f"{RESOURCE_SERVER_ID}/gateway:read", f"{RESOURCE_SERVER_ID}/gateway:write"],
        AllowedOAuthFlowsUserPoolClient=True,
        SupportedIdentityProviders=["COGNITO"],
    )["UserPoolClient"]
    return created["ClientId"], created["ClientSecret"]


def ensure_domain(user_pool_id):
    # Client-credentials token requests go through a Cognito "domain" even
    # though no human ever sees a login page for this flow.
    domain_prefix = f"nns-agentcore-{user_pool_id.split('_')[-1].lower()}"
    try:
        cognito.create_user_pool_domain(Domain=domain_prefix, UserPoolId=user_pool_id)
        print(f"Created Cognito domain: {domain_prefix} (waiting a few seconds for it to propagate)")
        time.sleep(10)
    except cognito.exceptions.InvalidParameterException:
        pass  # domain already exists from a previous run
    return domain_prefix


def get_access_token(domain_prefix, client_id, client_secret):
    token_url = f"https://{domain_prefix}.auth.{REGION}.amazoncognito.com/oauth2/token"
    response = requests.post(
        token_url,
        data={"grant_type": "client_credentials", "scope": SCOPE_STRING},
        auth=(client_id, client_secret),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    response.raise_for_status()
    return response.json()["access_token"]


def create_gateway(client_id, discovery_url):
    auth_config = {
        "customJWTAuthorizer": {
            "allowedClients": [client_id],
            "discoveryUrl": discovery_url,
        }
    }
    response = gateway_client.create_gateway(
        name="NnsCompanyToolsGateway",
        roleArn=GATEWAY_ROLE_ARN,
        protocolType="MCP",
        authorizerType="CUSTOM_JWT",
        authorizerConfiguration=auth_config,
        description="Gateway exposing mock SMAX/calendar/Jabber tools",
    )
    return response["gatewayId"], response["gatewayUrl"]


def create_gateway_target(gateway_id):
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
        gatewayIdentifier=gateway_id,
        name="MockCompanyToolsLambda",
        description="Mock SMAX/calendar/Jabber tools",
        targetConfiguration=target_config,
        credentialProviderConfigurations=credential_config,
    )


def main():
    print("Setting up Cognito user pool...")
    user_pool_id = get_or_create_user_pool()
    print(f"User Pool ID: {user_pool_id}")

    ensure_resource_server(user_pool_id)
    client_id, client_secret = get_or_create_m2m_client(user_pool_id)
    domain_prefix = ensure_domain(user_pool_id)
    discovery_url = f"https://cognito-idp.{REGION}.amazonaws.com/{user_pool_id}/.well-known/openid-configuration"

    print("Creating AgentCore Gateway...")
    gateway_id, gateway_url = create_gateway(client_id, discovery_url)
    print(f"Gateway ID: {gateway_id}")
    print(f"Gateway URL: {gateway_url}")

    print("Registering the mock tools Lambda as a Gateway Target...")
    create_gateway_target(gateway_id)

    print("\nSanity-checking that we can get an access token...")
    token = get_access_token(domain_prefix, client_id, client_secret)
    print(f"Got a token (first 20 chars): {token[:20]}...")

    print("\n--- SAVE ALL OF THIS — you'll need it for mcp_gateway_client.py ---")
    print(f"GATEWAY_URL={gateway_url}")
    print(f"COGNITO_DOMAIN={domain_prefix}")
    print(f"COGNITO_CLIENT_ID={client_id}")
    print(f"COGNITO_CLIENT_SECRET={client_secret}")
    print(f"COGNITO_USER_POOL_ID={user_pool_id}")


if __name__ == "__main__":
    main()