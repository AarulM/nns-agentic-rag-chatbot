"""
Attach AWS WAF to the AgentCore Gateway.

Creates a regional WAFv2 Web ACL with two AWS Managed Rule Groups (common
web exploits + known bad inputs) plus a rate-limiting rule, then associates
it with the existing Gateway so every request to the Gateway's public URL
is inspected before it reaches the Lambda target.

Small recurring cost once created (WAF bills per WebACL/month, per rule,
and per million requests inspected — a few dollars a month, nothing like
the Knowledge Base's hourly OpenSearch cost). Nothing here touches Ollama,
your agents, or the Knowledge Base.

Run: python setup_waf.py
"""
import time
import sys

import boto3

sys.path.insert(0, "agents")
from aws_config import REGION, PARTITION  # noqa: E402
GATEWAY_NAME = "NnsCompanyToolsGateway"
WEB_ACL_NAME = "nns-gateway-web-acl"

wafv2 = boto3.client("wafv2", region_name=REGION)
gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)


def find_gateway_arn():
    """Looks the Gateway up by name so nothing needs pasting after a rebuild."""
    account_id = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    token = None
    while True:
        kwargs = {"nextToken": token} if token else {}
        page = gateway_client.list_gateways(**kwargs)
        for g in page.get("items", []):
            if g["name"] == GATEWAY_NAME:
                return g["gatewayId"], (
                    f"arn:{PARTITION}:bedrock-agentcore:{REGION}:{account_id}"
                    f":gateway/{g['gatewayId']}"
                )
        token = page.get("nextToken")
        if not token:
            raise RuntimeError(f"No Gateway named '{GATEWAY_NAME}' — run setup_gateway.py first.")


def wait_for_gateway_ready(gateway_id):
    for _ in range(24):
        status = gateway_client.get_gateway(gatewayIdentifier=gateway_id)["status"]
        if status == "READY":
            return
        print(f"Gateway status: {status}, waiting...")
        time.sleep(5)
    raise TimeoutError("Gateway never returned to READY.")


def create_web_acl():
    existing = wafv2.list_web_acls(Scope="REGIONAL").get("WebACLs", [])
    for acl in existing:
        if acl["Name"] == WEB_ACL_NAME:
            print(f"Web ACL '{WEB_ACL_NAME}' already exists, reusing it.")
            return acl["ARN"]

    response = wafv2.create_web_acl(
        Name=WEB_ACL_NAME,
        Scope="REGIONAL",
        DefaultAction={"Allow": {}},
        Description="Protects the NNS company-tools AgentCore Gateway from common web exploits, bad inputs, and volumetric abuse.",
        VisibilityConfig={
            "SampledRequestsEnabled": True,
            "CloudWatchMetricsEnabled": True,
            "MetricName": "NnsGatewayWebAcl",
        },
        Rules=[
            {
                "Name": "AWS-CommonRuleSet",
                "Priority": 1,
                "OverrideAction": {"None": {}},
                "Statement": {
                    "ManagedRuleGroupStatement": {
                        "VendorName": "AWS",
                        "Name": "AWSManagedRulesCommonRuleSet",
                    }
                },
                "VisibilityConfig": {
                    "SampledRequestsEnabled": True,
                    "CloudWatchMetricsEnabled": True,
                    "MetricName": "CommonRuleSet",
                },
            },
            {
                "Name": "AWS-KnownBadInputs",
                "Priority": 2,
                "OverrideAction": {"None": {}},
                "Statement": {
                    "ManagedRuleGroupStatement": {
                        "VendorName": "AWS",
                        "Name": "AWSManagedRulesKnownBadInputsRuleSet",
                    }
                },
                "VisibilityConfig": {
                    "SampledRequestsEnabled": True,
                    "CloudWatchMetricsEnabled": True,
                    "MetricName": "KnownBadInputs",
                },
            },
            {
                "Name": "RateLimitPerIp",
                "Priority": 3,
                "Action": {"Block": {}},
                "Statement": {
                    "RateBasedStatement": {
                        "Limit": 2000,  # requests per 5-minute window, per IP
                        "AggregateKeyType": "IP",
                    }
                },
                "VisibilityConfig": {
                    "SampledRequestsEnabled": True,
                    "CloudWatchMetricsEnabled": True,
                    "MetricName": "RateLimitPerIp",
                },
            },
        ],
    )
    web_acl_arn = response["Summary"]["ARN"]
    print(f"Created Web ACL: {web_acl_arn}")
    return web_acl_arn


def associate_with_gateway(web_acl_arn, gateway_arn):
    wafv2.associate_web_acl(WebACLArn=web_acl_arn, ResourceArn=gateway_arn)
    print(f"Associated Web ACL with Gateway: {gateway_arn}")


def main():
    gateway_id, gateway_arn = find_gateway_arn()
    print(f"Found Gateway: {gateway_id}")
    print("Waiting for Gateway to be READY...")
    wait_for_gateway_ready(gateway_id)
    web_acl_arn = create_web_acl()
    associate_with_gateway(web_acl_arn, gateway_arn)
    print("\nDone. AWS WAF now inspects every request to your Gateway before it reaches the Lambda target.")


if __name__ == "__main__":
    main()