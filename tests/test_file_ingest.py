"""
Smoke tests for the multimodal ingestion pipeline.

Two tiers:

  Offline  — routing, validation, and BDA result parsing. No credentials,
             no cost, always run.
  Live     — real Textract and Bedrock Data Automation calls. Skipped
             unless RUN_LIVE_AWS_TESTS=1, because each one costs money and
             takes up to a few minutes.

Run offline only (default):   pytest tests/
Run everything:               RUN_LIVE_AWS_TESTS=1 pytest tests/

The live tests assert on phrases planted in the fixtures, so a pass means
the service actually read the content — not merely that it returned a
non-empty string.
"""
import os

import pytest

import file_ingest
from conftest import AUDIO_PHRASES, IMAGE_PHRASES

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

live_only = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_AWS_TESTS") != "1",
    reason="set RUN_LIVE_AWS_TESTS=1 to run tests that call AWS and cost money",
)


# --- Offline: routing ----------------------------------------------------
@pytest.mark.parametrize(
    "filename, expected",
    [
        ("scan.png", "image"),
        ("SCAN.JPG", "image"),
        ("photo.jpeg", "image"),
        ("diagram.webp", "image"),
        ("manual.pdf", "document"),
        ("briefing.mp3", "audio"),
        ("briefing.wav", "audio"),
        ("walkthrough.mp4", "video"),
        ("walkthrough.mov", "video"),
        ("dock_status.xlsx", "spreadsheet"),
        # "Any file format" means source code in any language, config, and
        # data files are all just text — no service call, no cost.
        ("notes.txt", "text"),
        ("handler.py", "text"),
        ("App.tsx", "text"),
        ("Main.java", "text"),
        ("query.sql", "text"),
        ("config.yaml", "text"),
        ("data.csv", "text"),
        ("Dockerfile", "text"),
    ],
)
def test_classify_routes_by_extension(filename, expected):
    assert file_ingest.classify(filename) == expected


def test_unknown_extension_falls_back_to_content_sniffing():
    """
    An extension nobody listed must still work if the bytes are text —
    otherwise "supports any format" is only true for formats I remembered.
    """
    assert file_ingest.classify("notes.qqq", b"Crane maintenance Tuesday.") == "text"
    assert file_ingest.looks_like_text(b"plain words") is True
    # A NUL byte in the first block is the classic binary tell.
    assert file_ingest.looks_like_text(b"PK\x03\x04\x00\x00binary") is False


@pytest.mark.parametrize(
    "filename, data",
    [("archive.zip", b"PK\x03\x04\x00\x00\x00"), ("mystery.bin", b"\x00\x01\x02\x03")],
)
def test_classify_rejects_binary_it_cannot_read(filename, data):
    with pytest.raises(file_ingest.UnsupportedFileType) as error:
        file_ingest.classify(filename, data)
    # The message must say what IS accepted, in plain language — it is
    # shown to the user, so no jargon and no extension dump.
    message = str(error.value)
    assert "images" in message and "source-code" in message
    assert "Traceback" not in message


def test_plaintext_skips_aws_entirely():
    """A .txt upload should never pay for a BDA round trip."""
    document = file_ingest.extract(b"Hard hats required in Dry Dock 12.", "note.txt")
    assert document.extractor == "plaintext"
    assert "Dry Dock 12" in document.text
    assert not document.is_empty


def test_forms_mode_rejects_audio_and_video():
    with pytest.raises(file_ingest.UnsupportedFileType):
        file_ingest.extract(b"", "briefing.mp3", forms_mode=True)


# --- Offline: BDA result parsing ----------------------------------------
# These shapes are copied from real service output (see the module
# docstring in file_ingest.py), so the parser is tested without paying for
# a job on every run.
def test_parses_document_pages():
    result = {
        "modality": "DOCUMENT",
        "segments": [
            {
                "pages": [
                    {"representation": {"markdown": "## SAFETY NOTICE\n\nHard hats."}},
                    {"representation": {"text": "Permit NNS-2026-0847"}},
                ]
            }
        ],
    }
    text = file_ingest._text_from_bda(result)
    assert "SAFETY NOTICE" in text
    assert "NNS-2026-0847" in text


def test_parses_audio_transcript():
    result = {
        "modality": "AUDIO",
        "segments": [
            {"audio": {"transcript": {"representation": {"text": "Welding by Friday."}}}}
        ],
    }
    assert file_ingest._text_from_bda(result) == "Welding by Friday."


def test_parses_video_transcript_and_chapter_summaries():
    """Video needs both halves: what was said and what was shown."""
    result = {
        "modality": "VIDEO",
        "segments": [
            {
                "video": {"transcript": {"representation": {"text": "Walkthrough of bay 3."}}},
                "chapters": [{"summary": "A welder inspects a hull seam."}],
            }
        ],
    }
    text = file_ingest._text_from_bda(result)
    assert "Walkthrough of bay 3." in text
    assert "hull seam" in text


def test_cache_key_is_content_addressed_not_name_addressed():
    """
    The same bytes under a different filename must hit the cache; different
    bytes must not. Filename-keyed caching would silently serve stale text
    for an edited file that kept its name.
    """
    a = file_ingest._cache_key(b"same bytes", False, ("FORMS",))
    b = file_ingest._cache_key(b"same bytes", False, ("FORMS",))
    c = file_ingest._cache_key(b"different bytes", False, ("FORMS",))
    assert a == b
    assert a != c


def test_cache_key_separates_extraction_settings():
    """Identical bytes produce different text per extractor, so the key must differ."""
    data = b"bytes"
    bda = file_ingest._cache_key(data, False, ("FORMS",))
    textract = file_ingest._cache_key(data, True, ("FORMS",))
    with_tables = file_ingest._cache_key(data, True, ("FORMS", "TABLES"))
    assert len({bda, textract, with_tables}) == 3


def test_textract_defaults_to_forms_only():
    """
    Textract bills each feature separately per page, so requesting TABLES
    by default would be a permanent ~30% surcharge nobody asked for.
    """
    import inspect

    signature = inspect.signature(file_ingest._text_from_textract)
    assert signature.parameters["features"].default == ("FORMS",)

    extract_signature = inspect.signature(file_ingest.extract)
    assert extract_signature.parameters["features"].default == ("FORMS",)


class _FakeS3:
    """Minimal stand-in for the S3 client used by _delete_prefix."""

    def __init__(self, keys, fail=False):
        self._keys = list(keys)
        self.deleted: list[str] = []
        self.delete_calls = 0
        self._fail = fail

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        keys = self._keys

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                matching = [{"Key": k} for k in keys if k.startswith(Prefix)]
                # Emit in pages of 1000, like the real paginator does.
                for i in range(0, max(len(matching), 1), 1000):
                    yield {"Contents": matching[i : i + 1000]}

        return _Paginator()

    def delete_objects(self, Bucket, Delete):
        self.delete_calls += 1
        if self._fail:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "DeleteObjects"
            )
        self.deleted.extend(o["Key"] for o in Delete["Objects"])
        return {}


def test_delete_prefix_removes_only_matching_keys(monkeypatch):
    """
    Cleaning up one job's staged upload must not touch another job's objects
    or the retained extract-cache.
    """
    fake = _FakeS3(
        [
            "uploads/abc123/report.pdf",
            "uploads/abc123/extra.json",
            "uploads/other/keep.pdf",
            "extract-cache/r1/bda/deadbeef.json",
        ]
    )
    monkeypatch.setitem(file_ingest._clients, "s3", fake)

    file_ingest._delete_prefix("bucket", "uploads/abc123/")

    assert set(fake.deleted) == {"uploads/abc123/report.pdf", "uploads/abc123/extra.json"}


def test_delete_prefix_batches_beyond_the_1000_key_cap(monkeypatch):
    """delete_objects accepts at most 1000 keys, so a big job must flush in batches."""
    keys = [f"bda-output/job/{i}.json" for i in range(1500)]
    fake = _FakeS3(keys)
    monkeypatch.setitem(file_ingest._clients, "s3", fake)

    file_ingest._delete_prefix("bucket", "bda-output/job/")

    assert len(fake.deleted) == 1500
    assert fake.delete_calls == 2  # 1000 + 500


def test_delete_prefix_swallows_errors(monkeypatch):
    """
    Cleanup is best-effort: a delete failure must never surface as an
    extraction failure — the text is already extracted, and the lifecycle
    rule is the backstop.
    """
    fake = _FakeS3(["uploads/x/file.bin"], fail=True)
    monkeypatch.setitem(file_ingest._clients, "s3", fake)

    file_ingest._delete_prefix("bucket", "uploads/x/")  # must not raise


def test_aws_errors_become_readable_not_tracebacks():
    """
    A stale KNOWLEDGE_BASE_ID is the single most likely misconfiguration
    (the IDs change on every teardown/rebuild), and it used to surface in
    the UI as a raw botocore ResourceNotFoundException traceback.
    """
    from botocore.exceptions import ClientError

    error = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
        "StartIngestionJob",
    )
    readable = file_ingest._readable(error, "Starting knowledge base sync")

    assert isinstance(readable, file_ingest.ExtractionError)
    message = str(readable)
    assert "Starting knowledge base sync" in message
    assert "KNOWLEDGE_BASE_ID" in message  # tells you which knob to turn
    assert "Traceback" not in message


def test_empty_result_is_reported_not_silently_indexed():
    document = file_ingest.ExtractedDocument(
        text="   ", filename="blank.png", modality="image", extractor="bda"
    )
    assert document.is_empty
    with pytest.raises(file_ingest.ExtractionError):
        file_ingest.publish_to_knowledge_base(document)


# --- Offline: partition handling ----------------------------------------
def test_arns_follow_the_region_partition(monkeypatch):
    """
    The GovCloud migration hinges on this: ARNs must be built from the
    region, never hardcoded to the commercial `aws` partition.
    """
    import importlib

    import aws_config

    monkeypatch.setenv("AWS_REGION", "us-gov-west-1")
    govcloud = importlib.reload(aws_config)
    assert govcloud.PARTITION == "aws-us-gov"
    assert govcloud.bda_project_arn("123456789012").startswith("arn:aws-us-gov:")
    # GovCloud inference profiles carry the "us-gov." prefix, not "us.".
    assert "us-gov.data-automation-v1" in govcloud.bda_profile_arn("123456789012")

    monkeypatch.setenv("AWS_REGION", "us-east-1")
    commercial = importlib.reload(aws_config)
    assert commercial.PARTITION == "aws"
    assert commercial.bda_project_arn("123456789012").startswith("arn:aws:")
    assert "us.data-automation-v1" in commercial.bda_profile_arn("123456789012")


# --- Live: real AWS calls ------------------------------------------------
@live_only
def test_live_image_via_vision_model(image_file):
    """Images route to a vision model, which reads text AND describes scenes."""
    document = file_ingest.extract(
        image_file.read_bytes(), image_file.name, use_cache=False
    )
    assert not document.is_empty
    assert document.extractor == "vision"
    for phrase in IMAGE_PHRASES:
        assert phrase in document.text, f"OCR missed {phrase!r}: {document.text[:400]}"


@live_only
def test_live_image_via_textract_forms(image_file):
    document = file_ingest.extract(
        image_file.read_bytes(), image_file.name, forms_mode=True, use_cache=False
    )
    assert not document.is_empty
    assert document.extractor == "textract"
    assert "NNS-2026-0847" in document.text
    # The whole reason to choose Textract here is the key/value section.
    assert "Form fields:" in document.text


@live_only
def test_live_photo_with_no_text_is_still_readable(textless_photo):
    """
    Regression: a photo containing no writing used to fail with "no
    readable text", because images were sent to an OCR pipeline. A
    photograph is not a document — it needs describing, not transcribing.
    """
    document = file_ingest.extract(
        textless_photo.read_bytes(), textless_photo.name, use_cache=False
    )
    assert not document.is_empty, "a textless photo must still produce a description"
    assert document.extractor == "vision"
    assert len(document.text.split()) > 15


@live_only
def test_live_pdf_via_bda(pdf_file):
    document = file_ingest.extract(
        pdf_file.read_bytes(), pdf_file.name, use_cache=False
    )
    assert not document.is_empty
    assert "DRY DOCK 12" in document.text


@live_only
def test_live_audio_via_bda(audio_file):
    document = file_ingest.extract(
        audio_file.read_bytes(), audio_file.name, use_cache=False
    )
    assert not document.is_empty
    lowered = document.text.lower()
    for phrase in AUDIO_PHRASES:
        assert phrase in lowered, f"transcript missed {phrase!r}: {document.text[:400]}"


@live_only
def test_live_video_via_bda(video_file):
    document = file_ingest.extract(
        video_file.read_bytes(), video_file.name, use_cache=False
    )
    assert not document.is_empty


@live_only
def test_live_second_upload_hits_cache_and_is_fast(image_file):
    """
    The cost optimization that matters most in practice: users re-upload the
    same permit or shift report repeatedly. The second pass must come from
    S3, not from a fresh (paid) extraction.
    """
    import time

    data = image_file.read_bytes()

    # Prime the cache, then time a fresh BDA run against a cached one.
    file_ingest.extract(data, image_file.name)

    started = time.monotonic()
    cached = file_ingest.extract(data, image_file.name)
    cached_seconds = time.monotonic() - started

    assert cached.metadata.get("from_cache") is True
    assert "DRY DOCK 12" in cached.text
    # A real BDA round trip is ~5-10s; a cache hit is one S3 GET.
    assert cached_seconds < 3, f"cache hit took {cached_seconds:.1f}s — did it miss?"

    # And bypassing the cache must still work, for a forced re-extract.
    fresh = file_ingest.extract(data, image_file.name, use_cache=False)
    assert not fresh.metadata.get("from_cache")
