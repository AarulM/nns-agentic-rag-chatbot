"""
Every per-deployment AWS identifier the agents need, in one place.

These change on every teardown/rebuild: after `cdk deploy` +
setup_gateway.py + create_memory.py, paste the new values here (each
script prints its lines in paste-ready form). Env vars of the same name
override, so nothing needs editing for a one-off switch. None of these
are secrets — the one real credential (the Cognito client secret) lives
in the gitignored gateway_secrets.py.
"""
import os

REGION = os.environ.get("AWS_REGION", "us-east-1")

# From the `cdk deploy` outputs.
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "RW01IL1SNT")
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID", "j9ikgpkaom8a")
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "1")

# From setup_gateway.py's printed output.
GATEWAY_URL = os.environ.get(
    "GATEWAY_URL",
    "https://nnscompanytoolsgateway-omj3vt66ow.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp",
)
COGNITO_DOMAIN = os.environ.get("COGNITO_DOMAIN", "nns-agentcore-dcnwgvsya")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "23ddablie1urm7mov53i1ltdba")

# From create_memory.py's printed output.
MEMORY_ID = os.environ.get("MEMORY_ID", "NnsSupervisorShortTermMemory-3BEy6kA6v7")
