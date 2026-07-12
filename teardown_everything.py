"""
One-shot teardown for the whole NNS project — fixes the "cdk destroy leaves
Gateway/Cognito/WAF/Memory dangling" pain from before.

Tears down, in the order that actually works:
  1. Disassociate + delete the WAF Web ACL (must happen before Gateway delete)
  2. Delete Gateway Targets, then the Gateway itself
  3. Delete the AgentCore short-term Memory resource
  4. Delete the Cognito User Pool Domain, App Clients, Resource Server, User Pool
  5. `cdk destroy` for everything CDK manages (S3 bucket, Knowledge Base,
     Lambda, IAM role, Guardrail)

Everything is looked up by name, so this works from any computer with the
right AWS credentials — no per-deployment IDs to paste in. Safe to re-run:
every step checks "does this still exist?" first and skips if it's gone.

Run: python teardown_everything.py
"""
import subprocess
import time
import boto3

REGION = "us-east-1"
GATEWAY_NAME = "NnsCompanyToolsGateway"
WEB_ACL_NAME = "nns-gateway-web-acl"
MEMORY_NAME = "NnsSupervisorShortTermMemory"
USER_POOL_NAME = "nns-agentcore-gateway-pool"

wafv2 = boto3.client("wafv2", region_name=REGION)
gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)
cognito = boto3.client("cognito-idp", region_name=REGION)


def step(title):
    print(f"\n--- {title} ---")


def find_gateway_id():
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


# ---------- 1. WAF ----------
def teardown_waf(gateway_id):
    step("WAF Web ACL")
    if gateway_id:
        account_id = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
        gateway_arn = f"arn:aws:bedrock-agentcore:{REGION}:{account_id}:gateway/{gateway_id}"
        try:
            acl = wafv2.get_web_acl_for_resource(ResourceArn=gateway_arn).get("WebACL")
        except wafv2.exceptions.ClientError:
            acl = None
        if acl:
            print("Disassociating Web ACL from Gateway...")
            wafv2.disassociate_web_acl(ResourceArn=gateway_arn)
        else:
            print("No Web ACL currently associated with the Gateway — skipping disassociate.")
    else:
        print("Gateway already gone — skipping disassociate.")

    existing = [a for a in wafv2.list_web_acls(Scope="REGIONAL").get("WebACLs", []) if a["Name"] == WEB_ACL_NAME]
    if not existing:
        print(f"No Web ACL named '{WEB_ACL_NAME}' — nothing to delete.")
        return

    web_acl = existing[0]
    detail = wafv2.get_web_acl(Name=web_acl["Name"], Scope="REGIONAL", Id=web_acl["Id"])
    lock_token = detail["LockToken"]

    for attempt in range(12):
        try:
            wafv2.delete_web_acl(Name=web_acl["Name"], Scope="REGIONAL", Id=web_acl["Id"], LockToken=lock_token)
            print(f"Deleted Web ACL '{WEB_ACL_NAME}'.")
            return
        except wafv2.exceptions.WAFAssociatedItemException:
            print(f"Web ACL still shows as associated (attempt {attempt + 1}), waiting for disassociation to propagate...")
            time.sleep(5)
    print("WARNING: could not delete Web ACL after 60s — check the console manually.")


# ---------- 2. Gateway ----------
def teardown_gateway(gateway_id):
    step("Gateway + Gateway Targets")
    if not gateway_id:
        print("Gateway already gone — skipping.")
        return

    targets = gateway_client.list_gateway_targets(gatewayIdentifier=gateway_id).get("items", [])
    for t in targets:
        print(f"Deleting target: {t['name']} ({t['targetId']})")
        gateway_client.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=t["targetId"])

    for attempt in range(24):
        remaining = gateway_client.list_gateway_targets(gatewayIdentifier=gateway_id).get("items", [])
        if not remaining:
            break
        print(f"Targets still present (attempt {attempt + 1}), waiting...")
        time.sleep(5)

    print("Deleting Gateway...")
    gateway_client.delete_gateway(gatewayIdentifier=gateway_id)
    print("Gateway deleted.")


# ---------- 3. Memory ----------
def teardown_memory():
    step("AgentCore Memory")
    from bedrock_agentcore.memory import MemoryClient
    memory_client = MemoryClient(region_name=REGION)
    # Memory IDs are the name plus a random suffix (Name-abc123...).
    matches = [
        m["id"] for m in memory_client.list_memories()
        if str(m.get("id", "")).startswith(f"{MEMORY_NAME}-")
    ]
    if not matches:
        print("Memory already gone — skipping.")
        return
    for memory_id in matches:
        try:
            memory_client.delete_memory(memory_id=memory_id)
            print(f"Deleted memory: {memory_id}")
        except Exception as e:
            print(f"WARNING: could not delete memory {memory_id} ({e}). Check the console manually.")


# ---------- 4. Cognito ----------
def teardown_cognito():
    step("Cognito (domain, app clients, resource server, user pool)")
    pools = [p for p in cognito.list_user_pools(MaxResults=60)["UserPools"] if p["Name"] == USER_POOL_NAME]
    if not pools:
        print(f"No user pool named '{USER_POOL_NAME}' — skipping.")
        return
    user_pool_id = pools[0]["Id"]
    print(f"Found User Pool: {user_pool_id}")

    domain = cognito.describe_user_pool(UserPoolId=user_pool_id)["UserPool"].get("Domain")
    if domain:
        print(f"Deleting domain: {domain}")
        cognito.delete_user_pool_domain(Domain=domain, UserPoolId=user_pool_id)

    for c in cognito.list_user_pool_clients(UserPoolId=user_pool_id).get("UserPoolClients", []):
        print(f"Deleting app client: {c['ClientName']} ({c['ClientId']})")
        cognito.delete_user_pool_client(UserPoolId=user_pool_id, ClientId=c["ClientId"])

    for rs in cognito.list_resource_servers(UserPoolId=user_pool_id, MaxResults=50).get("ResourceServers", []):
        print(f"Deleting resource server: {rs['Identifier']}")
        cognito.delete_resource_server(UserPoolId=user_pool_id, Identifier=rs["Identifier"])

    print("Deleting User Pool...")
    cognito.delete_user_pool(UserPoolId=user_pool_id)
    print("Cognito fully torn down.")


# ---------- 5. CDK stack ----------
def teardown_cdk():
    step("CDK stack (S3 bucket, Knowledge Base, Lambda, IAM role, Guardrail)")
    result = subprocess.run(["cdk", "destroy", "--force"], check=False)
    if result.returncode != 0:
        print("WARNING: `cdk destroy` exited non-zero — check the output above.")


def main():
    gateway_id = find_gateway_id()
    teardown_waf(gateway_id)
    teardown_gateway(gateway_id)
    teardown_memory()
    teardown_cognito()
    teardown_cdk()
    print("\nAll done. Everything created for this project — CDK-managed and boto3-managed — has been torn down.")


if __name__ == "__main__":
    main()
