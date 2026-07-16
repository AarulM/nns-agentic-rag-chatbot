"""
Shared model picker for every agent. Flip one environment variable to
switch between free local testing (Ollama) and real Bedrock (costs money).

    export MODEL_PROVIDER=ollama    # default — free, local, for testing
    export MODEL_PROVIDER=bedrock   # real Claude via AWS Bedrock — costs money

Defaults to "ollama" if MODEL_PROVIDER isn't set, on purpose, so you don't
accidentally rack up Bedrock charges while testing.
"""
import os

# Exposed so other modules can gate Bedrock-only features (long-term memory
# in supervisor.py, like the guardrail below) on the active provider.
PROVIDER = os.environ.get("MODEL_PROVIDER", "ollama").lower()


def get_model():
    provider = PROVIDER

    if provider == "ollama":
        from strands.models.ollama import OllamaModel
        return OllamaModel(
            host="http://localhost:11434",
            model_id=os.environ.get("OLLAMA_MODEL_ID", "llama3.1:8b"),
            # An 8B model at Ollama's default temperature (0.8) is a coin
            # flip on tool routing and synthesis, and even 0.1 left enough
            # randomness that one unlucky turn would poison the rest of the
            # conversation. 0 = greedy decoding: same input, same output,
            # every time — reliability matters more than variety here.
            temperature=0.0,
            # Ollama's default context window is tiny (2048 tokens). Once a
            # chat plus tool definitions outgrows it, Ollama silently drops
            # the OLDEST tokens — the system prompt — and the agent visibly
            # derails mid-conversation. 8192 fits comfortably in RAM for an
            # 8B model.
            options={"num_ctx": 8192},
            # Ollama unloads the model after 5 idle minutes by default, so
            # the first question after a coffee break pays a ~15s reload.
            keep_alive="30m",
        )

    if provider == "bedrock":
        from strands.models import BedrockModel
        region = os.environ.get("AWS_REGION", "us-east-1")
        # Commercial: Claude Haiku 4.5 (cheapest Claude tier) via the global
        # inference profile. GovCloud has neither Haiku 4.5 nor "global."
        # profiles — its profiles are prefixed "us-gov." and Sonnet 4.5 is
        # the FedRAMP-authorized Claude there. Either way, export
        # BEDROCK_MODEL_ID to override (list what your region actually has:
        # aws bedrock list-inference-profiles
        #   --query "inferenceProfileSummaries[].inferenceProfileId").
        default_model_id = (
            "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"
            if region.startswith("us-gov")
            else "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        kwargs = dict(
            model_id=os.environ.get("BEDROCK_MODEL_ID", default_model_id),
            region_name=region,
        )
        # Guardrail is optional — only applied if these env vars are set
        # (values come from the GuardrailId/GuardrailVersion CDK outputs).
        # Ollama never sees this; Guardrails are a Bedrock-only feature.
        guardrail_id = os.environ.get("GUARDRAIL_ID")
        guardrail_version = os.environ.get("GUARDRAIL_VERSION")
        if guardrail_id and guardrail_version:
            kwargs["guardrail_id"] = guardrail_id
            kwargs["guardrail_version"] = guardrail_version
            kwargs["guardrail_trace"] = "enabled"
        return BedrockModel(**kwargs)

    raise ValueError(f"Unknown MODEL_PROVIDER: {provider!r}. Use 'ollama' or 'bedrock'.")