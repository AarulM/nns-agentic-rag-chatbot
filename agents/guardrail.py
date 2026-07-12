"""
Standalone Bedrock Guardrail enforcement via the ApplyGuardrail API.

The Guardrail in the CDK stack only fires automatically when the model
itself is Bedrock (MODEL_PROVIDER=bedrock). In ollama mode nothing ever
invoked it — so this module calls it directly on the user's message and
the final reply, making the same policies (harmful content, ITAR topic
block, PII anonymization) apply no matter which LLM generated the text.

Empirically (and per the docs): source="INPUT" only evaluates the
blocking policies (topics/content); the PII-anonymize policy only runs
with source="OUTPUT". Costs a fraction of a cent per check.
"""
import boto3
from aws_config import REGION, GUARDRAIL_ID, GUARDRAIL_VERSION

_bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)


def apply_guardrail(text: str, source: str) -> tuple[bool, str]:
    """Runs text through the Guardrail. source is "INPUT" for the user's
    message or "OUTPUT" for the assistant's reply.

    Returns (blocked, text): blocked=True means a DENY/BLOCK policy fired
    and text is the guardrail's canned refusal message; blocked=False
    means text is safe to use — either unchanged, or with PII anonymized
    (e.g. "call {NAME} at {PHONE}").

    Fails open: if the API call itself errors (no AWS creds, network
    down), the original text passes through unchecked rather than
    bricking local testing. Fine for a demo; a production deployment
    would fail closed.
    """
    try:
        response = _bedrock_runtime.apply_guardrail(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
            source=source,
            content=[{"text": {"text": text}}],
        )
    except Exception as e:
        print(f"WARNING: guardrail check failed ({e}); text passed through unchecked.")
        return False, text

    if response.get("action") != "GUARDRAIL_INTERVENED":
        return False, text

    replacement = "".join(o.get("text", "") for o in response.get("outputs", [])) or text

    # Every policy type reports per-item actions inside "assessments"
    # (topicPolicy.topics[], contentPolicy.filters[], ...). Anything
    # BLOCKED means refuse; interventions that are only ANONYMIZED just
    # rewrite the text.
    blocked = any(
        item.get("action") == "BLOCKED"
        for assessment in response.get("assessments", [])
        for policy in assessment.values()
        if isinstance(policy, dict)
        for items in policy.values()
        if isinstance(items, list)
        for item in items
        if isinstance(item, dict)
    )
    return blocked, replacement
