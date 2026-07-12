"""
Idempotent setup/repair for everything CDK can't manage yet: the Cognito
M2M auth pieces, the AgentCore Gateway, and its mock-tools Lambda target.

Run it after ANY `cdk deploy` — a fresh account, a new computer, a crashed
half-finished setup, or a redeploy that changed the Lambda/role ARNs.
Every step either finds the existing resource or creates it (and the
Lambda target is replaced outright), so re-running is always safe. The
Lambda/role ARNs are read straight from the CloudFormation stack outputs,
so nothing needs to be pasted in beforehand.

At the end it prints the values to paste into agents/aws_config.py and
the client secret for agents/gateway_secrets.py.

Run: python setup_gateway.py
"""
import time
import boto3
import requests

REGION = "us-east-1"
STACK_NAME = "NnsAgenticRagChatbotStack"

USER_POOL_NAME = "nns-agentcore-gateway-pool"
RESOURCE_SERVER_ID = "nns-agentcore-gateway-id"
RESOURCE_SERVER_NAME = "nns-agentcore-gateway-name"
CLIENT_NAME = "nns-agentcore-gateway-client"
GATEWAY_NAME = "NnsCompanyToolsGateway"
TARGET_NAME = "MockCompanyToolsLambda"
SCOPES = [
    {"ScopeName": "gateway:read", "ScopeDescription": "Read access"},
    {"ScopeName": "gateway:write", "ScopeDescription": "Write access"},
]
SCOPE_STRING = f"{RESOURCE_SERVER_ID}/gateway:read {RESOURCE_SERVER_ID}/gateway:write"

# The single source of truth for the mock tools' MCP schema.
TOOL_SCHEMA = [
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

cognito = boto3.client("cognito-idp", region_name=REGION)
gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)
cloudformation = boto3.client("cloudformation", region_name=REGION)


def get_stack_outputs() -> dict:
    stack = cloudformation.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


# ---------- Cognito ----------
def get_or_create_user_pool():
    for pool in cognito.list_user_pools(MaxResults=60)["UserPools"]:
        if pool["Name"] == USER_POOL_NAME:
            return pool["Id"]
    return cognito.create_user_pool(PoolName=USER_POOL_NAME)["UserPool"]["Id"]


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


# ---------- Gateway ----------
def find_gateway_by_name():
    token = None
    while True:
        kwargs = {"nextToken": token} if token else {}
        page = gateway_client.list_gateways(**kwargs)
        for g in page.get("items", []):
            if g["name"] == GATEWAY_NAME:
                return g["gatewayId"]
        token = page.get("nextToken")
        if not token:
            return None


def get_or_create_gateway(client_id, discovery_url, role_arn):
    """Returns (gateway_id, gateway_url). If the Gateway already exists,
    points it at role_arn (which changes on every cdk destroy/deploy)."""
    auth_config = {
        "customJWTAuthorizer": {
            "allowedClients": [client_id],
            "discoveryUrl": discovery_url,
        }
    }
    gateway_id = find_gateway_by_name()
    if gateway_id:
        existing = gateway_client.get_gateway(gatewayIdentifier=gateway_id)
        print(f"Gateway '{GATEWAY_NAME}' already exists ({gateway_id}); updating its role/auth config.")
        gateway_client.update_gateway(
            gatewayIdentifier=gateway_id,
            name=GATEWAY_NAME,
            roleArn=role_arn,
            protocolType=existing["protocolType"],
            authorizerType=existing["authorizerType"],
            authorizerConfiguration=auth_config,
        )
        return gateway_id, existing["gatewayUrl"]

    created = gateway_client.create_gateway(
        name=GATEWAY_NAME,
        roleArn=role_arn,
        protocolType="MCP",
        authorizerType="CUSTOM_JWT",
        authorizerConfiguration=auth_config,
        description="Gateway exposing mock SMAX/calendar/Jabber tools",
    )
    return created["gatewayId"], created["gatewayUrl"]


def wait_for_gateway_ready(gateway_id):
    for _ in range(24):  # up to ~2 minutes
        status = gateway_client.get_gateway(gatewayIdentifier=gateway_id)["status"]
        if status == "READY":
            return
        print(f"Gateway status: {status}, waiting...")
        time.sleep(5)
    raise TimeoutError("Gateway never reached READY after 2 minutes.")


def replace_gateway_target(gateway_id, lambda_arn):
    """Deletes any existing target with our name (waiting until it's
    actually gone — deletion propagates slowly), then creates a fresh one
    pointing at lambda_arn."""
    targets = gateway_client.list_gateway_targets(gatewayIdentifier=gateway_id).get("items", [])
    for t in targets:
        if t["name"] == TARGET_NAME:
            print(f"Deleting old target: {t['name']} ({t['targetId']})")
            gateway_client.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=t["targetId"])

    for _ in range(24):
        remaining = gateway_client.list_gateway_targets(gatewayIdentifier=gateway_id).get("items", [])
        if not any(t["name"] == TARGET_NAME for t in remaining):
            break
        print("Old target still deleting, waiting...")
        time.sleep(5)
    else:
        raise TimeoutError("Old target never finished deleting after 2 minutes.")

    wait_for_gateway_ready(gateway_id)
    gateway_client.create_gateway_target(
        gatewayIdentifier=gateway_id,
        name=TARGET_NAME,
        description="Mock SMAX/calendar/Jabber tools",
        targetConfiguration={"mcp": {"lambda": {"lambdaArn": lambda_arn, "toolSchema": {"inlinePayload": TOOL_SCHEMA}}}},
        credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}],
    )
    print(f"Target '{TARGET_NAME}' now points at {lambda_arn}")


def main():
    print(f"Reading '{STACK_NAME}' CloudFormation outputs...")
    outputs = get_stack_outputs()
    lambda_arn = outputs["McpToolsLambdaArn"]
    role_arn = outputs["GatewayExecutionRoleArn"]

    print("Setting up Cognito user pool...")
    user_pool_id = get_or_create_user_pool()
    print(f"User Pool ID: {user_pool_id}")
    ensure_resource_server(user_pool_id)
    client_id, client_secret = get_or_create_m2m_client(user_pool_id)
    domain_prefix = ensure_domain(user_pool_id)
    discovery_url = f"https://cognito-idp.{REGION}.amazonaws.com/{user_pool_id}/.well-known/openid-configuration"

    print("Setting up AgentCore Gateway...")
    gateway_id, gateway_url = get_or_create_gateway(client_id, discovery_url, role_arn)
    wait_for_gateway_ready(gateway_id)
    print(f"Gateway ID: {gateway_id}")

    print("Registering the mock tools Lambda as the Gateway Target...")
    replace_gateway_target(gateway_id, lambda_arn)

    print("\nSanity-checking that we can get an access token...")
    token = get_access_token(domain_prefix, client_id, client_secret)
    print(f"Got a token (first 20 chars): {token[:20]}...")

    print("\n--- Paste into agents/aws_config.py ---")
    print(f'GATEWAY_URL = "{gateway_url}"')
    print(f'COGNITO_DOMAIN = "{domain_prefix}"')
    print(f'COGNITO_CLIENT_ID = "{client_id}"')
    print("\n--- Paste into agents/gateway_secrets.py (gitignored) ---")
    print(f'COGNITO_CLIENT_SECRET = "{client_secret}"')


if __name__ == "__main__":
    main()
