"""
Multimodal file understanding: turn an uploaded image / PDF / audio / video
into plain text, then hand that text to the same Bedrock Knowledge Base the
typed documents already live in.

Design: **Bedrock Data Automation first.** BDA is one API that covers all
four modalities (OCR for images and PDFs, ASR for audio, transcript +
scene description for video), which beats wiring Textract + Transcribe +
a multimodal model separately — one set of IAM permissions, one polling
loop, one output format to parse. Textract is used only where it genuinely
beats BDA: `FORMS`/`TABLES` extraction of key/value pairs out of
structured paperwork, which BDA's default project returns as prose.

Everything is region- and partition-agnostic: no ARN is spelled out, they
are all built from AWS_REGION (see aws_config.py), so the same code runs
in commercial AWS and in GovCloud's `aws-us-gov` partition.

Verified end-to-end against real AWS (us-east-1) — the BDA status values
and output JSON shapes below are what the service actually returns, not
what the docs describe.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("nns.file_ingest")

from aws_config import (
    REGION,
    IS_GOVCLOUD,
    DOCS_BUCKET,
    UPLOAD_BUCKET,
    KNOWLEDGE_BASE_ID,
    DATA_SOURCE_ID,
    bda_project_arn,
    bda_profile_arn,
)
from pii_redaction import redact

# --- Modality routing ----------------------------------------------------
# BDA's supported extensions per modality. Anything outside this map is
# rejected up front with a readable message rather than failing several
# seconds later inside the service.
EXTENSIONS = {
    # Images go to a vision model rather than BDA — see _describe_image.
    "image": {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"},
    "document": {".pdf", ".docx", ".doc"},
    "audio": {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".amr"},
    "video": {".mp4", ".mov"},
    "spreadsheet": {".xlsx", ".xlsm", ".xls"},
}

# Anything whose bytes are just text — source code in any language,
# config, logs, data. Decoding these locally is instant and free, so they
# never touch AWS. This list is a fast path, not a gate: a file with an
# unknown extension is still tried as text before anything else (see
# classify), which is what makes "any file format" actually true rather
# than "any format I remembered to list".
PLAINTEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    # Config / data
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".properties",
    # Source code
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".cc",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts",
    ".scala", ".pl", ".pm", ".lua", ".r", ".jl", ".dart", ".vue", ".svelte",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".sql", ".graphql",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".tf", ".dockerfile",
    ".makefile", ".gradle", ".ipynb", ".m", ".asm", ".f90", ".vb", ".clj",
    ".ex", ".exs", ".erl", ".hs", ".ml", ".nim", ".zig",
}

# Kept for the file-picker hint. The uploader deliberately does NOT
# restrict by type — see chat_ui.py.
SUPPORTED_EXTENSIONS = sorted(
    {e for exts in EXTENSIONS.values() for e in exts} | PLAINTEXT_EXTENSIONS
)

# BDA terminal states. The docs say "COMPLETED"; the service actually
# returns "Success" — confirmed by running a real job.
_BDA_DONE = "Success"
_BDA_FAILED = {"ClientError", "ServiceError"}

# Cache-schema tag folded into every cache key. Bump it whenever the text we
# store changes shape for the same input — as it did when PII redaction was
# added: entries written before redaction hold un-redacted text, so the tag
# change makes the cache miss them and re-extract (once) into a redacted
# entry, instead of serving raw PII out of an old cache hit.
_CACHE_SCHEMA = "r1"


class UnsupportedFileType(ValueError):
    """Raised for an extension no branch of the pipeline can handle."""


class ExtractionError(RuntimeError):
    """Raised when the pipeline cannot produce usable indexed text."""


# What each AWS error code actually means for someone using this app, so a
# misconfiguration surfaces as a sentence instead of a boto3 traceback.
_ERROR_HELP = {
    "ResourceNotFoundException": (
        "the resource does not exist in this region. Check KNOWLEDGE_BASE_ID "
        "and DATA_SOURCE_ID in .env against the current `cdk deploy` outputs "
        "— they change on every teardown/rebuild."
    ),
    "NoSuchBucket": "the S3 bucket does not exist. Check DOCS_BUCKET/UPLOAD_BUCKET in .env.",
    # Bedrock returns AccessDenied — not ResourceNotFound — when the target
    # resource does not exist, so that a caller cannot probe for which IDs
    # are real. Observed with a bogus KNOWLEDGE_BASE_ID against an account
    # holding AdministratorAccess. Naming only the permission cause sends
    # people to the IAM console for what is usually a stale ID.
    "AccessDeniedException": (
        "either the resource does not exist, or the credentials lack "
        "permission — Bedrock returns the same error for both. Check the "
        "IDs in .env first (they change on every teardown/rebuild); if they "
        "are correct, check IAM."
    ),
    "AccessDenied": "the current credentials lack permission for this call.",
    "ValidationException": "AWS rejected the request as invalid.",
    "ThrottlingException": "AWS is throttling this account; retry in a moment.",
}


def _readable(error: ClientError, doing: str, note: str = "") -> ExtractionError:
    """
    Turn a botocore error into one sentence someone can act on.

    The cause goes first and any reassurance last: these render in a narrow
    Streamlit sidebar, so a long parenthetical at the front pushes the part
    that matters off the bottom of the box.
    """
    code = error.response.get("Error", {}).get("Code", "")
    detail = error.response.get("Error", {}).get("Message", str(error))
    help_text = _ERROR_HELP.get(code, detail)
    message = f"{doing} failed ({code}): {help_text}"
    return ExtractionError(f"{message} {note}".strip())


@dataclass
class ExtractedDocument:
    """Normalized result of running one uploaded file through the pipeline."""

    text: str
    filename: str
    modality: str
    extractor: str  # "bda" | "textract" | "plaintext"
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def looks_like_text(data: bytes) -> bool:
    """
    Whether these bytes are plausibly a text file.

    Used so an unrecognized extension is not an automatic rejection: most
    things people attach that aren't media are text of some kind, and a
    file called `.env.local` or `Dockerfile` or `weird_ext.qqq` should just
    work. A NUL byte in the first block is the classic binary tell.
    """
    head = data[:8192]
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # A truncated multi-byte character at the boundary is fine; real
        # binary will fail well before the end of the block.
        try:
            head[:-4].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def classify(filename: str, data: bytes | None = None) -> str:
    """
    Map a file to one of: text, image, document, audio, video, spreadsheet.

    Extension first, then content sniffing. Passing `data` lets an unknown
    extension still be handled as text if that is what it actually is,
    which is the difference between "supports many formats" and "supports
    whatever you happen to drop on it".
    """
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix in PLAINTEXT_EXTENSIONS:
        return "text"
    for modality, extensions in EXTENSIONS.items():
        if suffix in extensions:
            return modality
    # Extensionless files people really do attach (Dockerfile, Makefile).
    if PurePosixPath(filename).name.lower() in ("dockerfile", "makefile", "rakefile"):
        return "text"
    if data is not None and looks_like_text(data):
        return "text"
    raise UnsupportedFileType(
        f"it isn't a format I can read ({suffix or 'no extension'}). "
        f"I can handle images, PDFs, Office documents, spreadsheets, audio, "
        f"video, and any text or source-code file."
    )


# --- Clients -------------------------------------------------------------
# Built lazily so importing this module never requires credentials — the
# smoke tests and env_check import it before AWS is necessarily reachable.
_clients: dict[str, object] = {}


def _client(service: str):
    if service not in _clients:
        _clients[service] = boto3.client(service, region_name=REGION)
    return _clients[service]


def _account_id() -> str:
    if "_account" not in _clients:
        _clients["_account"] = _client("sts").get_caller_identity()["Account"]
    return _clients["_account"]


def _s3_split(uri: str) -> tuple[str, str]:
    """s3://bucket/some/key -> ("bucket", "some/key")."""
    without_scheme = uri.removeprefix("s3://")
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def _read_json(uri: str) -> dict:
    bucket, key = _s3_split(uri)
    body = _client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def _delete_prefix(bucket: str, prefix: str) -> None:
    """
    Best-effort delete of every object under an S3 prefix.

    Used to purge the raw staged upload and the raw BDA output the instant
    they are no longer needed — both hold unredacted content (the original
    file's bytes, and BDA's pre-redaction extracted text), so they should
    not linger in S3 even for the lifecycle window. A cleanup failure is
    logged, never raised: the extraction already succeeded, and the
    shortened lifecycle rule (see setup_upload_bucket.py) is the backstop.
    """
    try:
        s3 = _client("s3")
        batch: list[dict] = []
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                batch.append({"Key": obj["Key"]})
                if len(batch) == 1000:  # delete_objects caps at 1000 keys.
                    s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                    batch = []
        if batch:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
    except ClientError as error:
        logger.warning("Could not clean up s3://%s/%s: %s", bucket, prefix, error)


# --- Bedrock Data Automation --------------------------------------------
def _run_bda(s3_uri: str, *, timeout: int = 900, poll: float = 5.0) -> dict:
    """Run one BDA job to completion and return its parsed result JSON."""
    account = _account_id()
    job_id = uuid.uuid4().hex

    response = _client("bedrock-data-automation-runtime").invoke_data_automation_async(
        inputConfiguration={"s3Uri": s3_uri},
        # No trailing slash: BDA concatenates directly, and "prefix/" turns
        # into a "prefix//jobid" double slash in the output path.
        outputConfiguration={"s3Uri": f"s3://{UPLOAD_BUCKET}/bda-output/{job_id}"},
        dataAutomationConfiguration={
            "dataAutomationProjectArn": bda_project_arn(account),
            "stage": "LIVE",
        },
        dataAutomationProfileArn=bda_profile_arn(account),
    )
    invocation_arn = response["invocationArn"]

    runtime = _client("bedrock-data-automation-runtime")
    deadline = time.monotonic() + timeout
    while True:
        status = runtime.get_data_automation_status(invocationArn=invocation_arn)
        state = status["status"]
        if state == _BDA_DONE:
            break
        if state in _BDA_FAILED:
            raise ExtractionError(
                f"Bedrock Data Automation failed ({state}): "
                f"{status.get('errorMessage', 'no detail returned')}"
            )
        if time.monotonic() > deadline:
            raise ExtractionError(
                f"Bedrock Data Automation timed out after {timeout}s "
                f"(last status: {state}). Large video files may need a longer timeout."
            )
        time.sleep(poll)

    # job_metadata.json points at one result.json per segment; a long video
    # or a split PDF produces several.
    metadata = _read_json(status["outputConfiguration"]["s3Uri"])
    segments = [
        segment["standard_output_path"]
        for asset in metadata.get("output_metadata", [])
        for segment in asset.get("segment_metadata", [])
        if segment.get("standard_output_path")
    ]
    if not segments:
        raise ExtractionError("BDA returned no output segments for this file.")

    parsed = {
        "segments": [_read_json(path) for path in segments],
        "modality": metadata.get("semantic_modality", ""),
    }

    # The raw BDA output holds the extracted text *before* redaction. It has
    # been parsed into memory now, so delete it immediately rather than
    # leaving unredacted text in S3 for the lifecycle window.
    _delete_prefix(UPLOAD_BUCKET, f"bda-output/{job_id}/")

    return parsed


def _text_from_bda(result: dict) -> str:
    """
    Pull readable text out of BDA's per-modality result JSON.

    The shapes differ by modality and are checked in order of specificity.
    Verified against real service output:
      document/image -> pages[].representation.{markdown,text}
      audio          -> audio.transcript.representation.text
      video          -> video.transcript.representation.text, chapters[]
    """
    parts: list[str] = []

    for segment in result["segments"]:
        # Audio and video transcripts.
        for key in ("audio", "video"):
            transcript = (
                segment.get(key, {}).get("transcript", {}).get("representation", {})
            )
            if text := transcript.get("text"):
                parts.append(text)

        # Video scene summaries add the visual half of the content that a
        # transcript alone misses.
        for chapter in segment.get("chapters", []):
            if summary := chapter.get("summary"):
                parts.append(summary)

        # Documents and images: markdown preserves headings and tables,
        # which survive chunking better than a flat text dump.
        for page in segment.get("pages", []):
            representation = page.get("representation", {})
            if text := (representation.get("markdown") or representation.get("text")):
                parts.append(text)

        # Standalone image summary (BDA's IMAGE modality, when the default
        # project classifies a photo as a scene rather than a document).
        image = segment.get("image", {})
        if summary := image.get("summary"):
            parts.append(summary)
        for line in image.get("text_lines", []):
            if text := line.get("text"):
                parts.append(text)

    return "\n\n".join(part.strip() for part in parts if part and part.strip())


# --- Images: vision model rather than OCR --------------------------------
# Bedrock Data Automation's default project treats an image as a DOCUMENT
# and runs OCR on it. That is exactly right for a scanned permit and
# exactly wrong for a photograph: a picture of a tower has no text, so OCR
# returns nothing and the upload fails with "no readable text" even though
# the image is perfectly legible to a human.
#
# A multimodal model reads both — it describes the scene AND transcribes
# any text in it — and at Haiku rates costs roughly a tenth of a BDA image
# job. So images go here by default, and BDA keeps the modalities where it
# is genuinely the best tool (PDF, audio, video).
_VISION_PROMPT = (
    "Describe this image in detail. Cover what it shows, any notable objects, "
    "people, places, or landmarks, and the setting. If the image contains any "
    "text, signage, labels, numbers, or handwriting, transcribe all of it "
    "verbatim. Write plain prose with no preamble."
)

# Bedrock/Anthropic reject a Converse image whose longest side exceeds
# 8000px ("dimensions exceed max allowed size") or whose bytes are too large
# ("Input is too long for requested model"). Real photos routinely trip one
# or both — that is what made a normal phone/download JPEG fail with an
# opaque ValidationException. 1568px is Claude's optimal long edge, so
# downscaling to it stays well under the cap AND cuts vision-token cost with
# no quality loss for description.
_MAX_IMAGE_EDGE = 1568


def _vision_model_id() -> str:
    """
    A vision-capable model in this region.

    Deliberately independent of MODEL_PROVIDER: the chat brain may be a
    local Ollama text model, but describing an image still needs a
    multimodal one, and Bedrock is where that lives.
    """
    if explicit := os.environ.get("VISION_MODEL_ID"):
        return explicit
    if IS_GOVCLOUD:
        # Sonnet 4.5 is the FedRAMP-authorized multimodal Claude there.
        return "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"
    return "global.anthropic.claude-haiku-4-5-20251001-v1:0"


def _normalize_image(data: bytes, filename: str) -> bytes:
    """
    Decode any image and re-encode it as a Bedrock-safe JPEG.

    One local PIL pass does four jobs at once: it enforces the 8000px
    dimension cap and keeps the byte size small (the two limits that made
    real photos fail), converts to RGB so odd encodings (CMYK, palette,
    alpha, progressive) don't reach the model, and normalizes every input
    format — png/webp/gif/bmp/tiff/heic-as-jpeg — to a single one the
    Converse API always accepts. Downscaling to 1568px on the long edge is
    free quality-wise for a description task and cheaper on vision tokens.
    """
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as error:
        suffix = PurePosixPath(filename).suffix.lower()
        raise ExtractionError(
            f"I couldn't decode that image ({suffix or 'unknown format'})."
        ) from error

    longest = max(image.size)
    if longest > _MAX_IMAGE_EDGE:
        scale = _MAX_IMAGE_EDGE / longest
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def _describe_image(data: bytes, filename: str) -> str:
    jpeg = _normalize_image(data, filename)

    response = _client("bedrock-runtime").converse(
        modelId=_vision_model_id(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": jpeg}}},
                    {"text": _VISION_PROMPT},
                ],
            }
        ],
        inferenceConfig={"maxTokens": 1000, "temperature": 0},
    )
    blocks = response["output"]["message"]["content"]
    return "\n".join(b["text"] for b in blocks if "text" in b).strip()


# --- Spreadsheets --------------------------------------------------------
def _text_from_spreadsheet(data: bytes, filename: str) -> str:
    """
    Every sheet flattened to markdown tables.

    Local and free — an .xlsx is a zip of XML, not something that needs a
    document-understanding service.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ExtractionError(
            "reading spreadsheets needs pandas and openpyxl installed."
        ) from error

    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    except Exception as error:
        raise ExtractionError(f"I couldn't open that spreadsheet ({error}).") from error

    parts = []
    for name, frame in sheets.items():
        if frame.empty:
            continue
        # CSV rather than to_markdown(): markdown tables would need the
        # `tabulate` package, and CSV embeds just as well while keeping the
        # dependency list shorter.
        parts.append(f"## Sheet: {name}\n\n{frame.to_csv(index=False).strip()}")
    return "\n\n".join(parts)


# --- Textract ------------------------------------------------------------
def _text_from_textract(data: bytes, features: tuple[str, ...] = ("FORMS",)) -> str:
    """
    Key/value pairs (and optionally tables) out of a structured form.

    Synchronous `analyze_document` only — it takes single-page images and
    single-page PDFs up to 5 MB. Multi-page PDFs need the async
    StartDocumentAnalysis flow, which is why forms mode is opt-in and BDA
    stays the default path.

    `features` defaults to FORMS alone because **Textract bills each
    feature separately, per page** — FORMS and TABLES together cost the sum
    of both, not one blended rate. Asking for TABLES you never read is a
    silent ~30% surcharge on every page. Pass ("FORMS", "TABLES") only when
    the tables actually matter.
    """
    response = _client("textract").analyze_document(
        Document={"Bytes": data}, FeatureTypes=list(features)
    )
    blocks = {block["Id"]: block for block in response["Blocks"]}

    def words_of(block: dict) -> str:
        out: list[str] = []
        for relationship in block.get("Relationships", []):
            if relationship["Type"] != "CHILD":
                continue
            for child_id in relationship["Ids"]:
                child = blocks.get(child_id, {})
                if child.get("BlockType") == "WORD":
                    out.append(child["Text"])
                elif child.get("BlockType") == "SELECTION_ELEMENT":
                    if child.get("SelectionStatus") == "SELECTED":
                        out.append("[X]")
        return " ".join(out)

    lines = [b["Text"] for b in response["Blocks"] if b["BlockType"] == "LINE"]

    pairs: list[str] = []
    for block in response["Blocks"]:
        if block["BlockType"] != "KEY_VALUE_SET":
            continue
        if "KEY" not in block.get("EntityTypes", []):
            continue
        key_text = words_of(block)
        value_text = ""
        for relationship in block.get("Relationships", []):
            if relationship["Type"] == "VALUE":
                for value_id in relationship["Ids"]:
                    value_text = words_of(blocks.get(value_id, {}))
        if key_text:
            pairs.append(f"{key_text} {value_text}".strip())

    sections = ["\n".join(lines)]
    if pairs:
        sections.append("Form fields:\n" + "\n".join(pairs))
    return "\n\n".join(section for section in sections if section.strip())


# --- PII redaction -------------------------------------------------------
def _redacted(document: "ExtractedDocument") -> "ExtractedDocument":
    """
    Scrub PII out of a document's text before it can be cached or indexed.

    This is the single chokepoint that sits upstream of *both* durable
    writes: the extract-cache put below and the Knowledge Base put in
    publish_to_knowledge_base. Redacting here means neither store ever holds
    the raw value. The per-entity counts are recorded in metadata so the
    redaction is auditable without logging what was removed.
    """
    result = redact(document.text)
    document.text = result.text
    if result.total:
        document.metadata["pii_redacted"] = result.entity_counts
    return document


def _cache_key(data: bytes, forms_mode: bool, features: tuple[str, ...]) -> str:
    """
    Deterministic key for one (file content, extraction settings) pair.

    Content-addressed, not filename-addressed: the same scan re-uploaded
    under a different name must still hit the cache, and an edited file
    with the same name must miss it. The settings are part of the key
    because Textract-with-tables and BDA produce different text for
    identical bytes.
    """
    digest = hashlib.sha256(data).hexdigest()
    mode = f"textract-{'-'.join(features)}" if forms_mode else "bda"
    return f"extract-cache/{_CACHE_SCHEMA}/{mode}/{digest}.json"


def _cache_get(key: str) -> dict | None:
    try:
        return _read_json(f"s3://{UPLOAD_BUCKET}/{key}")
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404", "AccessDenied"):
            return None
        raise
    except (ValueError, KeyError):
        # A corrupt cache entry must never block an upload — just re-extract.
        return None


def _cache_put(key: str, document: "ExtractedDocument") -> None:
    try:
        _client("s3").put_object(
            Bucket=UPLOAD_BUCKET,
            Key=key,
            Body=json.dumps(
                {
                    "text": document.text,
                    "modality": document.modality,
                    "extractor": document.extractor,
                    "metadata": document.metadata,
                }
            ).encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError:
        # Caching is an optimization. Failing to write it is not a reason
        # to fail an extraction that already succeeded and was paid for.
        pass


# --- PDFs: local text first, BDA only for scanned ones -------------------
# Digital PDFs (reports, workbooks, forms with a real text layer) carry
# their text directly, so pypdf reads them locally — free, instant, and
# immune to the "Document format is invalid or not supported" error BDA
# throws on some perfectly readable PDFs. Only scanned/image-only PDFs need
# BDA's OCR, so a near-empty local result falls through to BDA (see
# extract()). The floor guards against a PDF that is one stray character of
# text over an otherwise-scanned document.
_MIN_PDF_TEXT_CHARS = 32


def _text_from_pdf(data: bytes) -> str:
    """Local text from a digital PDF, or "" if there's no usable text layer."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf is a pinned dependency
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception:
        # Malformed or encrypted PDF pypdf can't parse — let BDA try.
        return ""
    text = "\n\n".join(part for part in parts if part)
    return text if len(text) >= _MIN_PDF_TEXT_CHARS else ""


def extract(
    data: bytes,
    filename: str,
    *,
    forms_mode: bool = False,
    features: tuple[str, ...] = ("FORMS",),
    use_cache: bool = True,
) -> ExtractedDocument:
    """
    Extract text from one uploaded file.

    forms_mode routes single-page images/PDFs through Textract instead of
    BDA — use it for structured paperwork (permits, inspection checklists)
    where the key/value pairs matter more than the prose. Note that
    Textract is roughly 5-6x BDA's per-page rate, so it is the deliberate
    exception, not the default.

    Results are cached in S3 by content hash. Re-uploading a file anyone
    has already processed costs one S3 GET (fractions of a cent) instead of
    a fresh extraction — which matters because the natural user behaviour
    here is to re-upload the same shift report or permit repeatedly.
    """
    modality = classify(filename, data)

    # Text of any kind — prose, config, source code in any language — never
    # needed a service call in the first place. It still gets redacted: a
    # pasted transcript or a config file is as likely to carry PII as any
    # scanned form, and this is the exact path a personal transcript took
    # into plaintext S3 once.
    if modality == "text":
        return _redacted(
            ExtractedDocument(
                text=data.decode("utf-8", errors="replace"),
                filename=filename,
                modality=modality,
                extractor="plaintext",
            )
        )

    if forms_mode and modality not in ("image", "document"):
        raise UnsupportedFileType(
            f"form mode only applies to images and PDFs, not {modality} files."
        )

    cache_key = _cache_key(data, forms_mode, features)
    if use_cache and UPLOAD_BUCKET:
        if cached := _cache_get(cache_key):
            return ExtractedDocument(
                text=cached["text"],
                filename=filename,
                modality=cached.get("modality", modality),
                extractor=cached.get("extractor", "cache"),
                metadata={**cached.get("metadata", {}), "from_cache": True},
            )

    is_pdf = modality == "document" and PurePosixPath(filename).suffix.lower() == ".pdf"

    if modality == "spreadsheet":
        document = ExtractedDocument(
            text=_text_from_spreadsheet(data, filename),
            filename=filename,
            modality=modality,
            extractor="spreadsheet",
        )
    elif is_pdf and not forms_mode and (pdf_text := _text_from_pdf(data)):
        # Digital PDF with a real text layer — read locally, no BDA, no cost.
        # A scanned PDF yields no text here and falls through to BDA below.
        document = ExtractedDocument(
            text=pdf_text, filename=filename, modality=modality, extractor="pdf-text"
        )
    elif modality == "image" and not forms_mode:
        # Vision model, not OCR — see the comment above _describe_image.
        try:
            text = _describe_image(data, filename)
        except ClientError as error:
            raise _readable(error, f"Reading the image {filename}") from error
        document = ExtractedDocument(
            text=text, filename=filename, modality=modality, extractor="vision"
        )
    elif forms_mode:
        try:
            text = _text_from_textract(data, features)
        except ClientError as error:
            raise _readable(error, f"Textract analysis of {filename}") from error
        document = ExtractedDocument(
            text=text, filename=filename, modality=modality, extractor="textract"
        )
    else:
        if not UPLOAD_BUCKET:
            raise ExtractionError(
                "UPLOAD_BUCKET is not set — BDA reads its input from S3, so a "
                "staging bucket is required. See .env.example."
            )

        upload_dir = f"uploads/{uuid.uuid4().hex}"
        key = f"{upload_dir}/{PurePosixPath(filename).name}"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        try:
            _client("s3").put_object(
                Bucket=UPLOAD_BUCKET, Key=key, Body=data, ContentType=content_type
            )
        except ClientError as error:
            raise _readable(error, f"Uploading {filename} to {UPLOAD_BUCKET}") from error

        try:
            result = _run_bda(f"s3://{UPLOAD_BUCKET}/{key}")
        except ClientError as error:
            raise _readable(error, f"Extracting text from {filename}") from error
        finally:
            # The staged original holds the raw file and its PII. BDA has
            # finished reading it by the time we get here (success or
            # failure), so delete it now instead of letting it sit in S3.
            _delete_prefix(UPLOAD_BUCKET, f"{upload_dir}/")
        document = ExtractedDocument(
            text=_text_from_bda(result),
            filename=filename,
            modality=modality,
            extractor="bda",
            metadata={"bda_modality": result["modality"], "source_key": key},
        )

    # Redact before caching, so the cache itself only ever holds redacted
    # text — a cache hit on a later upload serves the scrubbed version.
    document = _redacted(document)

    if use_cache and UPLOAD_BUCKET and not document.is_empty:
        _cache_put(cache_key, document)
    return document


# --- Knowledge Base hand-off --------------------------------------------
def publish_to_knowledge_base(document: ExtractedDocument) -> str:
    """
    Write extracted text into the KB's S3 data source and start an
    ingestion job, so the new content is chunked, embedded, and searchable
    by exactly the same path as the typed documents.

    The `.metadata.json` sidecar is a Bedrock KB convention: its attributes
    ride along on every chunk, which is what lets an answer cite the
    original filename and modality instead of an opaque chunk ID.
    """
    if document.is_empty:
        raise ExtractionError(
            f"{document.filename}: extraction produced no text; nothing to index."
        )
    if not DOCS_BUCKET:
        raise ExtractionError("DOCS_BUCKET is not set — see .env.example.")

    stem = PurePosixPath(document.filename).stem
    key = f"uploads/{uuid.uuid4().hex[:8]}-{stem}.txt"
    s3 = _client("s3")

    try:
        s3.put_object(
            Bucket=DOCS_BUCKET,
            Key=key,
            Body=document.text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        s3.put_object(
            Bucket=DOCS_BUCKET,
            Key=f"{key}.metadata.json",
            Body=json.dumps(
                {
                    "metadataAttributes": {
                        "source_filename": document.filename,
                        "source_modality": document.modality,
                        "extractor": document.extractor,
                    }
                }
            ).encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError as error:
        raise _readable(error, f"Writing {document.filename} to {DOCS_BUCKET}") from error

    if not (KNOWLEDGE_BASE_ID and DATA_SOURCE_ID):
        raise ExtractionError(
            "KNOWLEDGE_BASE_ID / DATA_SOURCE_ID not set — the file was "
            "uploaded but no ingestion job could be started. See .env.example."
        )

    try:
        job = _client("bedrock-agent").start_ingestion_job(
            knowledgeBaseId=KNOWLEDGE_BASE_ID, dataSourceId=DATA_SOURCE_ID
        )
    except ClientError as error:
        # A sync already running is not a failure — the file is in the
        # bucket and the in-flight job will pick it up.
        if error.response["Error"]["Code"] == "ConflictException":
            return "ingestion-already-running"
        # Anything else: the text IS safely in S3, so say so. Otherwise the
        # message reads like the upload was lost, and someone re-uploads
        # and pays for the extraction twice.
        raise _readable(
            error,
            "Starting knowledge base sync",
            note=(
                f"The extracted text is already saved at "
                f"s3://{DOCS_BUCKET}/{key}, so it will be picked up by the "
                f"next successful sync — no need to re-upload."
            ),
        ) from error

    return job["ingestionJob"]["ingestionJobId"]
