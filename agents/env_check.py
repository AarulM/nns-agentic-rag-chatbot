"""
Startup validation: fail fast, in English, before anything touches AWS.

Every check here exists because its absence produces a bad error somewhere
else — a raw `NoCredentialsError` traceback, an empty KB that silently
returns nothing, or the one that actually bit during development: a
`us-gov-west-1` region configured against credentials that live in the
commercial `aws` partition, which fails deep inside boto3 with a signing
error that names neither the region nor the account.

Run standalone:   python agents/env_check.py
Imported:         from env_check import require_ready; require_ready()
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from aws_config import (
    REGION,
    IS_GOVCLOUD,
    PARTITION,
    KNOWLEDGE_BASE_ID,
    DATA_SOURCE_ID,
    DOCS_BUCKET,
    UPLOAD_BUCKET,
    GATEWAY_URL,
    COGNITO_DOMAIN,
    COGNITO_CLIENT_ID,
    MEMORY_TABLE,
)

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Result:
    level: str
    check: str
    detail: str
    fix: str = ""


def _env_checks() -> list[Result]:
    results: list[Result] = []

    if not REGION:
        results.append(
            Result(
                FAIL,
                "AWS_REGION",
                "not set",
                "Set AWS_REGION in .env (us-east-1 for commercial, "
                "us-gov-west-1 for GovCloud). There is deliberately no default.",
            )
        )
    else:
        results.append(Result(OK, "AWS_REGION", f"{REGION} (partition: {PARTITION})"))

    # Required to answer anything at all from the Knowledge Base.
    for name, value in (
        ("KNOWLEDGE_BASE_ID", KNOWLEDGE_BASE_ID),
        ("DATA_SOURCE_ID", DATA_SOURCE_ID),
        ("DOCS_BUCKET", DOCS_BUCKET),
    ):
        if value:
            results.append(Result(OK, name, value))
        else:
            results.append(
                Result(
                    FAIL,
                    name,
                    "not set",
                    "Copy it from the `cdk deploy` outputs into .env.",
                )
            )

    # Required only for the multimodal upload path.
    if UPLOAD_BUCKET:
        results.append(Result(OK, "UPLOAD_BUCKET", UPLOAD_BUCKET))
    else:
        results.append(
            Result(
                WARN,
                "UPLOAD_BUCKET",
                "not set — file uploads will be rejected",
                "Create a staging bucket and set UPLOAD_BUCKET in .env.",
            )
        )

    # PII redaction of extracted text. Local only — no AWS call, no cost —
    # so it is checked here in the credential-free section. WARN (not FAIL)
    # on a problem: it only affects the file-upload path, and a broken setup
    # fails closed at upload time with a clear message rather than silently
    # writing raw PII, so it must not block text-only chat from starting.
    results.append(_redaction_check())

    # Required only for the Gateway tools (SMAX / calendar / Jabber).
    gateway_missing = [
        name
        for name, value in (
            ("GATEWAY_URL", GATEWAY_URL),
            ("COGNITO_DOMAIN", COGNITO_DOMAIN),
            ("COGNITO_CLIENT_ID", COGNITO_CLIENT_ID),
        )
        if not value
    ]
    if gateway_missing:
        results.append(
            Result(
                WARN,
                "Gateway config",
                f"missing {', '.join(gateway_missing)} — operations tools will fail",
                "Re-run setup_gateway.py and paste its output into .env.",
            )
        )
    else:
        results.append(Result(OK, "Gateway config", "set"))

    return results


def _redaction_check() -> Result:
    """Is PII redaction of extracted text ready? Local check, no AWS call."""
    import pii_redaction

    if not pii_redaction._enabled():
        return Result(
            WARN,
            "PII redaction",
            "disabled (PII_REDACTION=off) — extracted text is written to S3 raw",
            "Remove PII_REDACTION=off unless you have a specific reason; "
            "uploaded documents can contain names, SSNs, and emails.",
        )
    try:
        import presidio_analyzer  # noqa: F401
        import presidio_anonymizer  # noqa: F401
    except ImportError:
        return Result(
            WARN,
            "PII redaction",
            "enabled, but Presidio is not installed — file uploads will fail",
            "pip install presidio-analyzer presidio-anonymizer "
            "(or set PII_REDACTION=off to skip redaction — not advised).",
        )
    try:
        model = pii_redaction._load_spacy_model()
    except pii_redaction.RedactionUnavailable:
        return Result(
            WARN,
            "PII redaction",
            "enabled, but no spaCy English model is installed — uploads will fail",
            "python -m spacy download en_core_web_lg",
        )
    return Result(OK, "PII redaction", f"on (spaCy model: {model})")


def _identity_in(region: str | None):
    """get_caller_identity against one region, or None if it is rejected."""
    try:
        return boto3.client("sts", region_name=region).get_caller_identity()
    except (ClientError, BotoCoreError):
        return None


def _credential_checks() -> list[Result]:
    try:
        identity = boto3.client("sts", region_name=REGION or None).get_caller_identity()
    except NoCredentialsError:
        return [
            Result(
                FAIL,
                "AWS credentials",
                "none found",
                "export AWS_PROFILE=<your-profile>, or run `aws configure`.",
            )
        ]
    except (ClientError, BotoCoreError) as error:
        # STS rejected the call. "Expired" is the usual cause, but pointing
        # a GovCloud region at commercial credentials fails identically —
        # and telling someone to refresh perfectly good credentials sends
        # them down the wrong path entirely. Probe the other partition
        # before blaming the credentials.
        probe_region = "us-east-1" if IS_GOVCLOUD else "us-gov-west-1"
        elsewhere = _identity_in(probe_region)
        if elsewhere:
            other = elsewhere["Arn"].split(":")[1]
            return [
                Result(
                    FAIL,
                    "AWS credentials",
                    f"valid, but they belong to the {other!r} partition — "
                    f"AWS_REGION={REGION} needs {PARTITION!r}",
                    "These credentials are fine; they just cannot reach this "
                    "region. GovCloud is a separate account in a separate "
                    "partition. Use a profile for the right partition, or "
                    "change AWS_REGION to match the credentials you have.",
                )
            ]
        return [
            Result(
                FAIL,
                "AWS credentials",
                str(error).split("\n")[0],
                "Credentials are present but rejected — they may be expired. "
                "Refresh them (`aws sso login`, or new access keys).",
            )
        ]

    results = [Result(OK, "AWS credentials", f"account {identity['Account']}")]

    # The check that matters most for a GovCloud migration: an ARN's
    # partition tells you which cloud the credentials actually belong to,
    # and a mismatch with AWS_REGION is otherwise a cryptic signing error.
    arn_partition = identity["Arn"].split(":")[1]
    if arn_partition != PARTITION:
        results.append(
            Result(
                FAIL,
                "Partition match",
                f"AWS_REGION={REGION} implies partition {PARTITION!r}, but these "
                f"credentials are in {arn_partition!r} ({identity['Arn']})",
                "GovCloud is a separate account in a separate partition — "
                "commercial credentials cannot reach it. Use a GovCloud "
                "profile, or set AWS_REGION back to a commercial region.",
            )
        )
    else:
        results.append(Result(OK, "Partition match", f"credentials are in {PARTITION}"))

    return results


def _model_checks() -> list[Result]:
    provider = os.environ.get("MODEL_PROVIDER", "ollama").lower()
    if provider != "bedrock":
        return [
            Result(
                OK,
                "Model provider",
                f"{provider} (Bedrock model access not required)",
            )
        ]

    from model_config import resolve_bedrock_model_id

    model_id = resolve_bedrock_model_id()
    try:
        client = boto3.client("bedrock", region_name=REGION)
        profiles = {
            p["inferenceProfileId"]
            for p in client.list_inference_profiles().get("inferenceProfileSummaries", [])
        }
        models = {
            m["modelId"]
            for m in client.list_foundation_models().get("modelSummaries", [])
        }
    except (ClientError, BotoCoreError) as error:
        return [
            Result(
                FAIL,
                "Bedrock model access",
                str(error).split("\n")[0],
                "The credentials cannot list Bedrock models — add "
                "bedrock:ListFoundationModels / ListInferenceProfiles.",
            )
        ]

    if model_id in profiles or model_id in models:
        return [Result(OK, "Bedrock model access", f"{model_id} available")]

    fix = (
        f"{model_id} is not enabled in {REGION}. Available profiles: "
        f"{', '.join(sorted(profiles)) or '(none)'}. Set BEDROCK_MODEL_ID to one of them."
    )
    if IS_GOVCLOUD:
        fix += (
            " In GovCloud you must first accept the model EULA in the LINKED "
            "COMMERCIAL account, then enable the model in GovCloud — see README."
        )
    return [Result(FAIL, "Bedrock model access", f"{model_id} not available", fix)]


def _service_checks() -> list[Result]:
    """Confirm the multimodal services are reachable and permitted."""
    results: list[Result] = []
    if not REGION:
        return results

    try:
        boto3.client("bedrock-data-automation", region_name=REGION).list_data_automation_projects()
        results.append(Result(OK, "Bedrock Data Automation", "reachable"))
    except (ClientError, BotoCoreError) as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
        detail = "access denied" if code == "AccessDeniedException" else str(error).split("\n")[0]
        results.append(
            Result(
                WARN,
                "Bedrock Data Automation",
                detail,
                "File uploads need bedrock:InvokeDataAutomationAsync and "
                "bedrock:GetDataAutomationStatus. BDA is GovCloud US-West only.",
            )
        )

    return results


def run_checks() -> list[Result]:
    results = _env_checks()
    credentials = _credential_checks()
    results += credentials
    # Every remaining check calls AWS, so skip them if credentials are bad
    # rather than emitting a cascade of misleading failures.
    if not any(r.level == FAIL for r in credentials):
        results += _model_checks()
        results += _service_checks()
    return results


def format_results(results: list[Result]) -> str:
    icons = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}
    lines = [f"[{icons[r.level]}] {r.check}: {r.detail}" for r in results]
    problems = [r for r in results if r.level in (WARN, FAIL) and r.fix]
    if problems:
        lines.append("")
        lines.append("How to fix:")
        lines += [f"  - {r.check}: {r.fix}" for r in problems]
    return "\n".join(lines)


def require_ready() -> None:
    """Raise with a readable summary if anything is outright broken."""
    results = run_checks()
    if any(r.level == FAIL for r in results):
        raise RuntimeError(
            "Environment is not ready:\n\n"
            + format_results(results)
            + "\n\nRun `python agents/env_check.py` for the full report."
        )


if __name__ == "__main__":
    results = run_checks()
    print(format_results(results))
    failed = sum(r.level == FAIL for r in results)
    print()
    if failed:
        print(f"{failed} blocking problem(s). Fix the items above, then re-run.")
        sys.exit(1)
    print("Environment looks good.")
