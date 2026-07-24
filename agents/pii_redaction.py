"""
Strip personally-identifiable information out of extracted text *before* it
is ever written to S3 or handed to the Knowledge Base.

Why this exists: everything the file pipeline pulls out of an upload lands in
two durable places — the extract-cache in the staging bucket and the KB's
S3 data source — and both are retained (the cache indefinitely). A single
uploaded personal transcript once put someone's name into plaintext in S3
that way. Redacting at the pipeline chokepoint means the sensitive value
never reaches storage in the first place, rather than being scrubbed after
the fact.

Engine: **Microsoft Presidio running locally**, not AWS Comprehend. This is a
deliberate cost choice — Presidio is pure local CPU (a spaCy NER model plus
regex recognizers), so it adds **zero AWS spend**: no per-character
Comprehend `DetectPiiEntities` charge on every file, no new IAM permission,
no network call. Plain regex alone was also rejected: regex cannot catch a
person's *name*, and a name is the exact PII that leaked here. Presidio's
NER does catch names; regex handles the structured identifiers (SSN, cards,
etc.) where it is strongest.

Partition-agnostic by construction: there is no AWS call in this file, so it
behaves identically in commercial AWS and in GovCloud's `aws-us-gov`
partition — which also makes it the right layer for the ITAR posture, since
nothing sensitive leaves the box to be redacted.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# --- What we redact, and what we deliberately do not ---------------------
# An allowlist, not "everything Presidio can find". The uploads here are
# overwhelmingly shipyard technical content — permits, shift reports, config
# files, source code — and Presidio's LOCATION / DATE_TIME / ORGANIZATION /
# NRP recognizers would shred exactly the terms that make that content
# useful: "Dry Dock 12", an inspection date, "NNS", a hull number. So the
# default set is the genuinely-sensitive entities only. Override with
# PII_ENTITIES (comma-separated) if a deployment needs a different balance.
_DEFAULT_ENTITIES = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "US_ITIN",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
    "IBAN_CODE",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "MEDICAL_LICENSE",
    "CRYPTO",
)

# Presidio scores each hit 0..1. Phone numbers with no surrounding context
# come in around 0.4, so the floor sits just below that. Everything we care
# most about — names (0.85), email (1.0), SSN (0.95) — clears it easily.
# Lower means "redact more aggressively"; over-redaction is the safe failure
# for a leak-prevention control, so we err low rather than high.
_DEFAULT_THRESHOLD = 0.35

# spaCy model for the name-detection NER. `_lg` is the most accurate of the
# English models and is what this was tuned against; `_sm` is the fallback
# so a machine that only has the small model still runs (with slightly worse
# name recall) rather than failing outright.
_MODEL_CANDIDATES = ("en_core_web_lg", "en_core_web_sm")


def _enabled() -> bool:
    """Redaction is on unless explicitly switched off via PII_REDACTION."""
    return os.environ.get("PII_REDACTION", "on").strip().lower() not in (
        "off",
        "0",
        "false",
        "no",
    )


def _configured_entities() -> list[str]:
    override = os.environ.get("PII_ENTITIES", "").strip()
    if override:
        return [e.strip() for e in override.split(",") if e.strip()]
    return list(_DEFAULT_ENTITIES)


def _threshold() -> float:
    try:
        return float(os.environ.get("PII_SCORE_THRESHOLD", _DEFAULT_THRESHOLD))
    except ValueError:
        return _DEFAULT_THRESHOLD


@dataclass
class RedactionResult:
    """
    Outcome of redacting one block of text.

    `entity_counts` is what makes the control observable — it rides along in
    the ExtractedDocument metadata so an operator can see that redaction ran
    and what it removed, without ever logging the removed values themselves.
    """

    text: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    engine: str = "presidio"  # or "disabled"

    @property
    def total(self) -> int:
        return sum(self.entity_counts.values())


# --- Engine construction (lazy, built once) ------------------------------
# Importing Presidio and loading the spaCy model costs a second or two and a
# few hundred MB, so it happens on first use, not at import — same rationale
# as the lazy boto3 clients in file_ingest. A lock keeps two Streamlit
# reruns from building it twice.
_engines: dict[str, object] = {}
_lock = threading.Lock()


class RedactionUnavailable(RuntimeError):
    """Presidio (or its spaCy model) could not be loaded while redaction is on."""


def _load_spacy_model() -> str:
    import spacy.util

    override = os.environ.get("PII_SPACY_MODEL", "").strip()
    candidates = (override, *_MODEL_CANDIDATES) if override else _MODEL_CANDIDATES
    for model in candidates:
        if model and spacy.util.is_package(model):
            return model
    raise RedactionUnavailable(
        "No spaCy English model is installed, so names cannot be detected. "
        "Install one with:  python -m spacy download en_core_web_lg\n"
        "(or set PII_REDACTION=off to disable redaction — not recommended, "
        "as extracted text is written to S3 unredacted.)"
    )


def _build_engines() -> tuple[object, object, str]:
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
    except ImportError as error:  # pragma: no cover - install-time failure
        raise RedactionUnavailable(
            "PII redaction is enabled but Presidio is not installed. "
            "Install it with:  pip install presidio-analyzer presidio-anonymizer\n"
            "(or set PII_REDACTION=off to disable — not recommended.)"
        ) from error

    model = _load_spacy_model()
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model}],
        }
    )
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(), supported_languages=["en"]
    )

    # Presidio's built-in US_SSN recognizer deliberately *invalidates* the
    # three canonical placeholder SSNs (123-45-6789, 987-65-4320,
    # 078-05-1120) because they are published examples. For a redaction
    # control that guard is backwards: a placeholder-shaped SSN is exactly
    # what ends up in a document by mistake, and we want it gone. This extra
    # recognizer catches any SSN-shaped string regardless — layered over the
    # built-in, so real SSNs are still caught by both.
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_SSN",
            name="ssn_shape_recognizer",
            patterns=[
                Pattern("ssn-delimited", r"\b[0-9]{3}[- ][0-9]{2}[- ][0-9]{4}\b", 0.6),
                Pattern("ssn-9-digit", r"\b[0-9]{9}\b", 0.3),
            ],
            context=["ssn", "social security", "social-security"],
        )
    )

    return analyzer, AnonymizerEngine(), model


def _get_engines() -> tuple[object, object, str]:
    if "analyzer" not in _engines:
        with _lock:
            if "analyzer" not in _engines:
                analyzer, anonymizer, model = _build_engines()
                _engines["analyzer"] = analyzer
                _engines["anonymizer"] = anonymizer
                _engines["model"] = model
                logger.info("PII redaction ready (spaCy model: %s)", model)
    return _engines["analyzer"], _engines["anonymizer"], _engines["model"]


def redact(text: str) -> RedactionResult:
    """
    Return `text` with PII replaced by typed placeholders, e.g. a name
    becomes `[PERSON]` and an address `[EMAIL_ADDRESS]`.

    Typed placeholders (rather than a blanket `***`) keep the text useful for
    retrieval and answering: "the [PERSON] who signed the permit" still reads
    as a sentence about a person, so a chunk stays meaningful after redaction.

    No-ops on empty text and when PII_REDACTION is off. Raises
    RedactionUnavailable if the engine cannot be built while redaction is on
    — fail-closed, so a broken install never silently writes raw PII to S3.
    """
    if not text or not text.strip():
        return RedactionResult(text=text)
    if not _enabled():
        return RedactionResult(text=text, engine="disabled")

    from presidio_anonymizer.entities import OperatorConfig

    analyzer, anonymizer, _ = _get_engines()
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=_configured_entities(),
        score_threshold=_threshold(),
    )
    if not results:
        return RedactionResult(text=text)

    counts: dict[str, int] = {}
    operators = {}
    for r in results:
        counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
        operators.setdefault(
            r.entity_type,
            OperatorConfig("replace", {"new_value": f"[{r.entity_type}]"}),
        )

    anonymized = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators
    )
    return RedactionResult(text=anonymized.text, entity_counts=counts)
