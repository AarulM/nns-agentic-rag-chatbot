"""
Every per-deployment AWS identifier the agents need, in one place.

All of these come from the environment — put them in a `.env` file at the
repo root (start from `.env.example`) and they load automatically. Nothing
is hardcoded, so a clone of this repo picks up *your* deployment's IDs
rather than someone else's, and the same code runs unchanged against
commercial AWS or GovCloud.

After `cdk deploy` + setup_gateway.py + create_memory.py, paste the new
values into `.env` (each script prints its lines in paste-ready form).
Real shell env vars win over `.env`, so a one-off switch needs no edit.
None of these are secrets — the one real credential (the Cognito client
secret) lives in the gitignored gateway_secrets.py.

Values are read leniently here (missing ones are None/""). Nothing blows
up at import time; env_check.py is what fails fast with a readable error.
"""
import os
from pathlib import Path

# --- .env loading -------------------------------------------------------
# Deliberately hand-rolled rather than pulling in python-dotenv: it is ~15
# lines, and it keeps the dependency list short enough to audit by eye,
# which matters more than usual for a repo destined for an air-gapped
# GovCloud account.
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # A blank line in .env means "I have no value for this", so it must
        # not be exported. Otherwise `BEDROCK_MODEL_ID=` sets the variable
        # to "", and every `os.environ.get(name, sensible_default)` in the
        # codebase silently resolves to "" instead of the default.
        if not value:
            continue
        # Real environment wins, so `KNOWLEDGE_BASE_ID=x streamlit run ...`
        # still overrides the file for a one-off.
        os.environ.setdefault(key, value)


_load_dotenv()

# --- Region and partition ------------------------------------------------
# No default region: guessing one is how a "why is my KB empty?" hour
# starts. env_check.py turns a missing value into a readable error.
REGION = os.environ.get("AWS_REGION", "")

# GovCloud lives in the `aws-us-gov` IAM partition, so every ARN this code
# builds has to be constructed from this rather than the literal "aws".
IS_GOVCLOUD = REGION.startswith("us-gov")
PARTITION = "aws-us-gov" if IS_GOVCLOUD else "aws"

# From the `cdk deploy` outputs.
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
DATA_SOURCE_ID = os.environ.get("DATA_SOURCE_ID", "")
DOCS_BUCKET = os.environ.get("DOCS_BUCKET", "")
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID", "")
# DRAFT always reflects the latest deployed stack config, so guardrail
# edits take effect on the next `cdk deploy` with no version bookkeeping.
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "DRAFT")

# From setup_gateway.py's printed output.
GATEWAY_URL = os.environ.get("GATEWAY_URL", "")
COGNITO_DOMAIN = os.environ.get("COGNITO_DOMAIN", "")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "")

# The DynamoDB table create_memory.py makes (AgentCore Memory isn't
# available in GovCloud, so memory lives in DynamoDB). Fixed name, no
# random suffix — nothing to paste after a rebuild.
MEMORY_TABLE = os.environ.get("MEMORY_TABLE", "NnsChatbotMemory")

# --- Multimodal upload pipeline -----------------------------------------
# Staging bucket for user uploads and Bedrock Data Automation output. Kept
# separate from DOCS_BUCKET so a failed extraction never leaves a partial
# artifact sitting in the Knowledge Base's data source.
UPLOAD_BUCKET = os.environ.get("UPLOAD_BUCKET", "")

# BDA's built-in project handles all four modalities with no setup. Set
# BDA_PROJECT_ARN to point at your own project if you add blueprints.
BDA_PROJECT_ARN = os.environ.get("BDA_PROJECT_ARN", "")

# BDA requires an inference profile ARN. The prefix is partition-specific
# in exactly the way the Bedrock model IDs are ("us." commercial vs
# "us-gov." in GovCloud), so it is derived rather than hardcoded.
BDA_PROFILE_ARN = os.environ.get("BDA_PROFILE_ARN", "")


def bda_project_arn(account_id: str) -> str:
    """AWS-managed default BDA project — no project creation needed."""
    return BDA_PROJECT_ARN or (
        f"arn:{PARTITION}:bedrock:{REGION}:aws:data-automation-project/public-default"
    )


def bda_profile_arn(account_id: str) -> str:
    """Inference profile BDA runs under, in the caller's own account."""
    if BDA_PROFILE_ARN:
        return BDA_PROFILE_ARN
    prefix = "us-gov" if IS_GOVCLOUD else "us"
    return (
        f"arn:{PARTITION}:bedrock:{REGION}:{account_id}"
        f":data-automation-profile/{prefix}.data-automation-v1"
    )
