"""
Tests for the local, credential-free portion of env_check — specifically the
PII-redaction readiness check, which must WARN (not FAIL, not silently pass)
in every broken-setup case so a coworker cloning fresh is told what to fix.

These make no AWS calls.
"""
import sys

import pytest

pytest.importorskip("presidio_analyzer", reason="presidio not installed")

import env_check
import pii_redaction


def test_redaction_check_ok_when_ready(monkeypatch):
    """The model is installed in this environment, so the check should pass."""
    monkeypatch.setenv("PII_REDACTION", "on")
    result = env_check._redaction_check()
    assert result.level == env_check.OK
    assert "on" in result.detail


def test_redaction_check_warns_when_disabled(monkeypatch):
    """PII_REDACTION=off is a real risk (raw text to S3) — surface it, don't hide it."""
    monkeypatch.setenv("PII_REDACTION", "off")
    result = env_check._redaction_check()
    assert result.level == env_check.WARN
    assert "disabled" in result.detail
    assert result.fix  # tells the operator how to re-enable


def test_redaction_check_warns_when_model_missing(monkeypatch):
    """The exact fresh-clone failure: Presidio installed, spaCy model not."""
    monkeypatch.setenv("PII_REDACTION", "on")

    def no_model():
        raise pii_redaction.RedactionUnavailable("no model")

    monkeypatch.setattr(pii_redaction, "_load_spacy_model", no_model)
    result = env_check._redaction_check()
    assert result.level == env_check.WARN
    assert "spacy" in result.fix.lower()  # points at `spacy download`


def test_redaction_check_warns_when_presidio_missing(monkeypatch):
    """If Presidio itself is absent, still WARN with an install hint."""
    monkeypatch.setenv("PII_REDACTION", "on")
    # Force `import presidio_analyzer` to raise ImportError.
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    result = env_check._redaction_check()
    assert result.level == env_check.WARN
    assert "presidio" in result.fix.lower()


def test_redaction_check_never_fails_the_whole_app(monkeypatch):
    """
    Redaction problems only affect file uploads, so the check must never
    return FAIL — that would block text-only chat from starting via
    require_ready(). WARN is the ceiling here.
    """
    for value in ("on", "off"):
        monkeypatch.setenv("PII_REDACTION", value)
        assert env_check._redaction_check().level in (env_check.OK, env_check.WARN)
