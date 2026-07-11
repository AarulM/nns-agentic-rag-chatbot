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
        )

    if provider == "bedrock":
        from strands.models import BedrockModel
        return BedrockModel(
            model_id=os.environ.get(
                "BEDROCK_MODEL_ID",
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
            ),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {provider!r}. Use 'ollama' or 'bedrock'.")