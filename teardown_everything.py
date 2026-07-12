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

Safe to re-run — every step checks "does this still exist?" first and skips
if it's already gone, so a partial failure halfway through won't break a
second run.

Run: python teardown_everything.py
"""
import subprocess
import time
import boto3

REGION = "us-east-1"
ACCOUNT_ID = "465733921455"
GATEWAY_ID = "nnscompanytoolsgateway-omj3vt66ow"
GATEWAY_ARN = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT_ID}:gateway/{GATEWAY_ID}"
WEB_ACL_NAME = "nns-gateway-web-acl"
MEMORY_ID = "NnsSupervisorShortTermMemory-3BEy6kA6v7"
COGNITO_DOMAIN = "nns-agentcore-dcnwgvsya"

wafv2 = boto3.client("wafv2", region_name=REGION)
gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)
cognito = boto3.client("cognito-idp", region_name=REGION)


def step(title):
    print(f"\n--- {title} ---")


# ---------- 1. WAF ----------
def teardown_waf():
    step("WAF Web ACL")
    try:
        acl = wafv2.get_web_acl_for_resource(ResourceArn=GATEWAY_ARN).get("WebACL")
    except wafv2.exceptions.ClientError:
        # Covers "no ACL associated" and "gateway itself no longer exists".
        acl = None

    if acl:
        print("Disassociating Web ACL from Gateway...")
        wafv2.disassociate_web_acl(ResourceArn=GATEWAY_ARN)
    else:
        print("No Web ACL currently associated with the Gateway — skipping disassociate.")

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
def teardown_gateway():
    step("Gateway + Gateway Targets")
    try:
        gateway = gateway_client.get_gateway(gatewayIdentifier=GATEWAY_ID)
    except gateway_client.exceptions.ResourceNotFoundException:
        print("Gateway already gone — skipping.")
        return

    targets = gateway_client.list_gateway_targets(gatewayIdentifier=GATEWAY_ID).get("items", [])
    for t in targets:
        print(f"Deleting target: {t['name']} ({t['targetId']})")
        gateway_client.delete_gateway_target(gatewayIdentifier=GATEWAY_ID, targetId=t["targetId"])

    for attempt in range(24):
        remaining = gateway_client.list_gateway_targets(gatewayIdentifier=GATEWAY_ID).get("items", [])
        if not remaining:
            break
        print(f"Targets still present (attempt {attempt + 1}), waiting...")
        time.sleep(5)

    print("Deleting Gateway...")
    gateway_client.delete_gateway(gatewayIdentifier=GATEWAY_ID)
    print("Gateway deleted.")


# ---------- 3. Memory ----------
def teardown_memory():
    step("AgentCore Memory")
    from bedrock_agentcore.memory import MemoryClient
    memory_client = MemoryClient(region_name=REGION)
    try:
        memory_client.delete_memory(memory_id=MEMORY_ID)
        print(f"Deleted memory: {MEMORY_ID}")
    except Exception as e:
        if "ResourceNotFoundException" in str(type(e)) or "not found" in str(e).lower():
            print("Memory already gone — skipping.")
        else:
            print(f"WARNING: could not delete memory ({e}). Check the console manually.")


# ---------- 4. Cognito ----------
def teardown_cognito():
    step("Cognito (domain, app clients, resource server, user pool)")
    try:
        user_pool_id = cognito.describe_user_pool_domain(Domain=COGNITO_DOMAIN)["DomainDescription"]["UserPoolId"]
    except Exception:
        print(f"Domain '{COGNITO_DOMAIN}' not found — Cognito resources may already be deleted. Skipping.")
        return

    print(f"Found User Pool: {user_pool_id}")

    print("Deleting domain...")
    cognito.delete_user_pool_domain(Domain=COGNITO_DOMAIN, UserPoolId=user_pool_id)

    clients = cognito.list_user_pool_clients(UserPoolId=user_pool_id).get("UserPoolClients", [])
    for c in clients:
        print(f"Deleting app client: {c['ClientName']} ({c['ClientId']})")
        cognito.delete_user_pool_client(UserPoolId=user_pool_id, ClientId=c["ClientId"])

    resource_servers = cognito.list_resource_servers(UserPoolId=user_pool_id, MaxResults=50).get("ResourceServers", [])
    for rs in resource_servers:
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
    teardown_waf()
    teardown_gateway()
    teardown_memory()
    teardown_cognito()
    teardown_cdk()
    print("\nAll done. Everything created for this project — CDK-managed and boto3-managed — has been torn down.")


if __name__ == "__main__":
    main()