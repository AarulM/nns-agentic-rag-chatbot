"""
Tests for PII redaction of extracted text.

These prove the security-relevant promise directly: PII is gone from the
text the pipeline hands off, and — because the pipeline writes exactly
`document.text` to S3 and the Knowledge Base — from what actually lands in
storage. All offline and free (Presidio is local CPU, no AWS call).

The suite skips cleanly if Presidio or a spaCy English model is not
installed, rather than failing, so a machine that has not run
`python -m spacy download en_core_web_lg` is not blocked.
"""
import io
import os

import pytest

# The engine and a spaCy model are required for these to mean anything.
pytest.importorskip("presidio_analyzer", reason="presidio not installed")
pytest.importorskip("presidio_anonymizer", reason="presidio not installed")

import pii_redaction
import file_ingest
from conftest import PII_NAME, PII_EMAIL, PII_SURVIVES

live_only = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_AWS_TESTS") != "1",
    reason="set RUN_LIVE_AWS_TESTS=1 to run tests that call AWS and cost money",
)

_spacy = pytest.importorskip("spacy.util", reason="spacy not installed")
if not any(_spacy.is_package(m) for m in ("en_core_web_lg", "en_core_web_sm")):
    pytest.skip(
        "no spaCy English model installed (python -m spacy download en_core_web_lg)",
        allow_module_level=True,
    )


# A stand-in personal transcript: the exact shape of the document that once
# leaked a name into plaintext S3. Fake name / SSN / email / phone, plus
# genuine shipyard content that must survive.
TRANSCRIPT = (
    "Shift handoff transcript, Dry Dock 12.\n"
    "Present: Maria Gonzalez and Dave Whitfield.\n"
    "Reach Maria at maria.gonzalez@example.com or 757-555-0142.\n"
    "Her SSN on file is 123-45-6789.\n"
    "Welding on permit NNS-2026-0847 completes Friday 2026-07-15."
)


@pytest.fixture(autouse=True)
def _redaction_on(monkeypatch):
    """Force redaction on regardless of the ambient PII_REDACTION setting."""
    monkeypatch.setenv("PII_REDACTION", "on")


def test_name_ssn_email_and_phone_are_removed():
    result = pii_redaction.redact(TRANSCRIPT)

    # The actual PII values must be gone from the text.
    for leaked in (
        "Maria Gonzalez",
        "Dave Whitfield",
        "maria.gonzalez@example.com",
        "757-555-0142",
        "123-45-6789",
    ):
        assert leaked not in result.text, f"{leaked!r} survived redaction"

    # And replaced with typed placeholders, not deleted — so the text still
    # reads as a sentence for retrieval.
    assert "[PERSON]" in result.text
    assert "[EMAIL_ADDRESS]" in result.text
    assert "[US_SSN]" in result.text

    counts = result.entity_counts
    assert counts.get("PERSON", 0) >= 2
    assert counts.get("US_SSN", 0) >= 1
    assert counts.get("EMAIL_ADDRESS", 0) >= 1


def test_placeholder_ssn_is_still_redacted():
    """
    123-45-6789 is on Presidio's list of canonical placeholder SSNs that its
    built-in recognizer *invalidates*. For a leak-prevention control that is
    the wrong default — a fake-looking SSN is exactly what lands in a doc by
    mistake — so our custom recognizer must catch it anyway.
    """
    result = pii_redaction.redact("The SSN is 123-45-6789 for the record.")
    assert "123-45-6789" not in result.text
    assert "[US_SSN]" in result.text


def test_shipyard_content_is_not_over_redacted():
    """
    LOCATION / DATE_TIME / ORG recognizers are deliberately excluded so the
    technical terms that make the content useful survive.
    """
    result = pii_redaction.redact(TRANSCRIPT)
    assert "Dry Dock 12" in result.text
    assert "NNS-2026-0847" in result.text
    assert "2026-07-15" in result.text


def test_extract_redacts_what_would_be_written_to_s3():
    """
    The end-to-end guarantee: extract() runs fully locally for a .txt, and
    publish_to_knowledge_base writes exactly `document.text` to S3 — so this
    is precisely what would land in the bucket. It must already be redacted,
    and the redaction must be recorded in metadata for audit.
    """
    document = file_ingest.extract(TRANSCRIPT.encode("utf-8"), "handoff.txt")

    assert document.extractor == "plaintext"
    assert "Maria Gonzalez" not in document.text
    assert "123-45-6789" not in document.text
    assert "maria.gonzalez@example.com" not in document.text
    # The audit trail is present without ever recording the removed values.
    assert document.metadata.get("pii_redacted", {}).get("PERSON", 0) >= 1
    assert "Maria" not in str(document.metadata)


def test_redaction_can_be_disabled_for_an_escape_hatch(monkeypatch):
    """PII_REDACTION=off is the documented bypass; it must actually bypass."""
    monkeypatch.setenv("PII_REDACTION", "off")
    result = pii_redaction.redact(TRANSCRIPT)
    assert result.engine == "disabled"
    assert "Maria Gonzalez" in result.text  # untouched


def test_empty_text_is_a_noop():
    assert pii_redaction.redact("").text == ""
    assert pii_redaction.redact("   ").total == 0


def test_redaction_fails_closed_when_engine_cannot_build(monkeypatch):
    """
    The security-critical property: if the engine cannot load (e.g. the
    spaCy model was never downloaded), redaction must RAISE, never fall
    through and pass raw PII to the caller — which would write it to S3.
    """
    monkeypatch.setattr(pii_redaction, "_engines", {})

    def boom():
        raise pii_redaction.RedactionUnavailable("model missing")

    monkeypatch.setattr(pii_redaction, "_build_engines", boom)

    with pytest.raises(pii_redaction.RedactionUnavailable):
        pii_redaction.redact("Contact Maria Gonzalez at maria@example.com.")


def test_load_spacy_model_raises_when_no_model_installed(monkeypatch):
    """With no English model present, model loading must fail loudly."""
    import spacy.util

    monkeypatch.setattr(spacy.util, "is_package", lambda name: False)
    monkeypatch.delenv("PII_SPACY_MODEL", raising=False)
    with pytest.raises(pii_redaction.RedactionUnavailable):
        pii_redaction._load_spacy_model()


# --- Redaction fires through the real extractors, not just .txt ----------
def test_spreadsheet_pii_is_redacted():
    """
    Spreadsheets are parsed locally (no AWS, free), but still flow through
    the same redaction chokepoint. A staff roster is a classic PII carrier.
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    buffer = io.BytesIO()
    pd.DataFrame(
        {
            "Name": [PII_NAME],
            "Email": [PII_EMAIL],
            "Assignment": [f"Welding, {PII_SURVIVES}"],
        }
    ).to_excel(buffer, index=False)

    document = file_ingest.extract(buffer.getvalue(), "roster.xlsx", use_cache=False)

    assert document.extractor == "spreadsheet"
    assert PII_NAME not in document.text
    assert PII_EMAIL not in document.text
    assert "[PERSON]" in document.text
    assert PII_SURVIVES in document.text  # shipyard content survives
    assert document.metadata.get("pii_redacted")


@live_only
def test_live_image_pii_is_redacted(pii_image_file):
    """Vision path (image -> Bedrock multimodal): name/email must be scrubbed."""
    document = file_ingest.extract(
        pii_image_file.read_bytes(), pii_image_file.name, use_cache=False
    )
    assert document.extractor == "vision"
    assert PII_NAME not in document.text, f"name survived: {document.text[:400]}"
    assert PII_EMAIL not in document.text, f"email survived: {document.text[:400]}"
    assert "[PERSON]" in document.text or "[EMAIL_ADDRESS]" in document.text
    assert PII_SURVIVES in document.text  # shipyard content survives
    assert document.metadata.get("pii_redacted")


@live_only
def test_live_pdf_pii_is_redacted(pii_pdf_file):
    """BDA path (PDF -> Bedrock Data Automation): name/email must be scrubbed."""
    document = file_ingest.extract(
        pii_pdf_file.read_bytes(), pii_pdf_file.name, use_cache=False
    )
    assert document.extractor == "bda"
    assert PII_NAME not in document.text, f"name survived: {document.text[:400]}"
    assert PII_EMAIL not in document.text, f"email survived: {document.text[:400]}"
    assert "[PERSON]" in document.text or "[EMAIL_ADDRESS]" in document.text
    assert document.metadata.get("pii_redacted")
