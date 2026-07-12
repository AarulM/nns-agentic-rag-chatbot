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
        )

    if provider == "bedrock":
        from strands.models import BedrockModel
        kwargs = dict(
            # Claude Haiku 4.5 (cheapest Claude tier) via the global
            # inference profile. Export BEDROCK_MODEL_ID to use a bigger
            # model, e.g. global.anthropic.claude-sonnet-4-5-20250929-v1:0.
            model_id=os.environ.get(
                "BEDROCK_MODEL_ID",
                "global.anthropic.claude-haiku-4-5-20251001-v1:0",
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