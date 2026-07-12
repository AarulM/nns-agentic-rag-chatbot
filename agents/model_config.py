"""
Shared model picker for every agent. Flip one environment variable to
switch between free local testing (Ollama) and real Bedrock (costs money).

    export MODEL_PROVIDER=ollama    # default — free, local, for testing
    export MODEL_PROVIDER=bedrock   # real Claude via AWS Bedrock — costs money

Defaults to "ollama" if MODEL_PROVIDER isn't set, on purpose, so you don't
accidentally rack up Bedrock charges while testing.
"""
import os


def get_model():
    provider = os.environ.get("MODEL_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from strands.models.ollama import OllamaModel
        return OllamaModel(
            host="http://localhost:11434",
            model_id=os.environ.get("OLLAMA_MODEL_ID", "llama3.1:8b"),
            # An 8B model at Ollama's default temperature (0.8) is a coin
            # flip on tool routing and synthesis — some runs it answers
            # perfectly, others it rambles about "question format". Near-zero
            # temperature makes it behave the same way every run.
            temperature=0.1,
            # Ollama's default context window is tiny (2048 tokens). Once a
            # chat plus tool definitions outgrows it, Ollama silently drops
            # the OLDEST tokens — the system prompt — and the agent visibly
            # derails mid-conversation. 8192 fits comfortably in RAM for an
            # 8B model.
            options={"num_ctx": 8192},
        )

    if provider == "bedrock":
        from strands.models import BedrockModel
        kwargs = dict(
            model_id=os.environ.get(
                "BEDROCK_MODEL_ID",
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            ),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
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