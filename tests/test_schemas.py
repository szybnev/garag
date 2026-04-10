"""Invariants for the `Document` Pydantic contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import Document


def _doc(**overrides: object) -> Document:
    defaults: dict[str, object] = {
        "doc_id": "mitre_attack:T1059.001",
        "source": "mitre_attack",
        "url": "https://attack.mitre.org/techniques/T1059/001/",
        "title": "Command and Scripting Interpreter: PowerShell",
        "text": "Adversaries may abuse PowerShell commands and scripts for execution.",
        "metadata": {"stix_type": "attack-pattern", "platforms": ["Windows"]},
    }
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


def test_valid_document_roundtrip() -> None:
    doc = _doc()
    assert doc.source == "mitre_attack"
    assert doc.doc_id.startswith("mitre_attack:")
    assert doc.metadata["platforms"] == ["Windows"]


def test_document_is_frozen() -> None:
    doc = _doc()
    with pytest.raises(ValidationError):
        doc.title = "mutated"  # type: ignore[misc]


def test_unknown_source_rejected() -> None:
    with pytest.raises(ValidationError):
        _doc(source="unknown", doc_id="unknown:x")  # type: ignore[arg-type]


def test_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        _doc(text="")


def test_empty_title_rejected() -> None:
    with pytest.raises(ValidationError):
        _doc(title="")


def test_doc_id_prefix_must_match_source() -> None:
    with pytest.raises(ValidationError):
        _doc(doc_id="owasp:A01_2021", source="mitre_attack")


def test_all_five_sources_accepted() -> None:
    cases = [
        ("mitre_attack", "mitre_attack:T1059", "https://attack.mitre.org/techniques/T1059/"),
        ("mitre_atlas", "mitre_atlas:AML.T0000", "https://atlas.mitre.org/techniques/AML.T0000"),
        ("owasp", "owasp:A01_2021", "https://owasp.org/Top10/A01_2021-Broken_Access_Control/"),
        ("hackerone", "hackerone:120", "https://hackerone.com/reports/120"),
        ("man_pages", "man_pages:nmap", "https://linux.die.net/man/1/nmap"),
    ]
    for source, doc_id, url in cases:
        doc = _doc(source=source, doc_id=doc_id, url=url)  # type: ignore[arg-type]
        assert doc.source == source
        assert doc.doc_id == doc_id


def test_url_may_be_none() -> None:
    doc = _doc(url=None)
    assert doc.url is None


def test_metadata_defaults_empty() -> None:
    doc = Document(
        doc_id="owasp:A01_2021",
        source="owasp",
        title="Broken Access Control",
        text="Access control enforces policy...",
    )
    assert doc.metadata == {}
