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
import boto3

REGION = "us-east-1"
ACCOUNT_ID = "465733921455"
GATEWAY_ID = "nnscompanytoolsgateway-omj3vt66ow"
WEB_ACL_NAME = "nns-gateway-web-acl"

GATEWAY_ARN = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT_ID}:gateway/{GATEWAY_ID}"

wafv2 = boto3.client("wafv2", region_name=REGION)
gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)


def wait_for_gateway_ready():
    for _ in range(24):
        status = gateway_client.get_gateway(gatewayIdentifier=GATEWAY_ID)["status"]
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


def associate_with_gateway(web_acl_arn):
    wafv2.associate_web_acl(WebACLArn=web_acl_arn, ResourceArn=GATEWAY_ARN)
    print(f"Associated Web ACL with Gateway: {GATEWAY_ARN}")


def main():
    print("Waiting for Gateway to be READY...")
    wait_for_gateway_ready()
    web_acl_arn = create_web_acl()
    associate_with_gateway(web_acl_arn)
    print("\nDone. AWS WAF now inspects every request to your Gateway before it reaches the Lambda target.")


if __name__ == "__main__":
    main()