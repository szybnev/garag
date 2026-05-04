"""Invariants for the `Document`, `Chunk`, and public API Pydantic contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import Chunk, Citation, Document, QueryRequest, QueryResponse


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


# ---------- Chunk ----------


def _chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "chunk_id": "mitre_attack:T1059.001::0",
        "doc_id": "mitre_attack:T1059.001",
        "source": "mitre_attack",
        "url": "https://attack.mitre.org/techniques/T1059/001/",
        "title": "Command and Scripting Interpreter: PowerShell",
        "text": "Adversaries may abuse PowerShell commands and scripts.",
        "char_start": 0,
        "char_end": 55,
        "token_count": 10,
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def test_valid_chunk() -> None:
    chunk = _chunk()
    assert chunk.chunk_id == "mitre_attack:T1059.001::0"
    assert chunk.token_count == 10


def test_chunk_is_frozen() -> None:
    chunk = _chunk()
    with pytest.raises(ValidationError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_chunk_id_must_start_with_doc_id() -> None:
    with pytest.raises(ValidationError):
        _chunk(chunk_id="mitre_attack:T9999::0")  # doc_id mismatch


def test_chunk_id_prefix_must_match_source() -> None:
    with pytest.raises(ValidationError):
        _chunk(
            chunk_id="owasp:A01::0",
            doc_id="owasp:A01",
            source="mitre_attack",
        )


def test_chunk_char_end_must_be_ge_char_start() -> None:
    with pytest.raises(ValidationError):
        _chunk(char_start=100, char_end=50)


def test_chunk_token_count_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _chunk(token_count=0)


def test_chunk_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        _chunk(text="")


# ---------- Citation ----------


def test_citation_valid() -> None:
    c = Citation(
        chunk_id="owasp:A03_2021-Injection::6",
        source="owasp",
        url="https://owasp.org/Top10/A03_2021-Injection/",
        quote="SQL injection occurs when untrusted input is passed to a SQL query.",
    )
    assert c.source == "owasp"
    assert c.url is not None


def test_citation_url_optional() -> None:
    c = Citation(
        chunk_id="man_pages:nmap::0", source="man_pages", quote="nmap -sU scans UDP ports."
    )
    assert c.url is None


def test_citation_quote_max_length() -> None:
    long_quote = "x" * 700
    with pytest.raises(ValidationError):
        Citation(chunk_id="x:y", source="owasp", quote=long_quote)


def test_citation_empty_quote_rejected() -> None:
    with pytest.raises(ValidationError):
        Citation(chunk_id="x:y", source="owasp", quote="")


def test_citation_is_frozen() -> None:
    c = Citation(chunk_id="x:y", source="owasp", quote="something")
    with pytest.raises(ValidationError):
        c.quote = "mutated"  # type: ignore[misc]


# ---------- QueryRequest ----------


def test_query_request_minimum() -> None:
    req = QueryRequest(query="What is T1059?")
    assert req.top_k == 12


def test_query_request_top_k_bounds() -> None:
    QueryRequest(query="x" * 10, top_k=1)
    QueryRequest(query="x" * 10, top_k=20)
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 10, top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 10, top_k=21)


def test_query_request_query_length() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="a")
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001)


def test_query_request_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="What is T1059?", unknown="field")  # type: ignore[call-arg]


# ---------- QueryResponse ----------


def test_query_response_minimal() -> None:
    resp = QueryResponse(
        answer="T1059 is a MITRE ATT&CK technique for command-line interpreter abuse.",
        confidence=0.92,
    )
    assert resp.citations == []
    assert resp.used_chunks == []


def test_query_response_with_citations() -> None:
    cit = Citation(
        chunk_id="mitre_attack:T1059::0",
        source="mitre_attack",
        url="https://attack.mitre.org/techniques/T1059/",
        quote="Adversaries may abuse command-line interpreters.",
    )
    resp = QueryResponse(
        answer="MITRE T1059 covers command interpreter abuse.",
        citations=[cit],
        confidence=0.88,
        used_chunks=["mitre_attack:T1059::0"],
    )
    assert len(resp.citations) == 1
    assert resp.used_chunks == ["mitre_attack:T1059::0"]


def test_query_response_confidence_bounds() -> None:
    QueryResponse(answer="ok", confidence=0.0)
    QueryResponse(answer="ok", confidence=1.0)
    with pytest.raises(ValidationError):
        QueryResponse(answer="ok", confidence=1.1)
    with pytest.raises(ValidationError):
        QueryResponse(answer="ok", confidence=-0.1)


def test_query_response_empty_answer_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryResponse(answer="", confidence=0.5)


def test_query_response_json_schema_is_serialisable() -> None:
    """Sanity check that the schema can be passed to Ollama `format=...`."""
    schema = QueryResponse.model_json_schema()
    assert schema["type"] == "object"
    assert "answer" in schema["properties"]
    assert "citations" in schema["properties"]
    assert "confidence" in schema["properties"]
    assert "used_chunks" in schema["properties"]
    assert "latency_ms" in schema["properties"]


def test_query_response_latency_ms_default_none() -> None:
    resp = QueryResponse(answer="ok", confidence=0.5)
    assert resp.latency_ms is None


def test_query_response_latency_ms_roundtrip() -> None:
    timings = {"dense": 12.3, "sparse": 4.5, "fusion": 0.2, "rerank": 250.0, "gen": 4800.0}
    resp = QueryResponse(answer="ok", confidence=0.9, latency_ms=timings)
    assert resp.latency_ms is not None
    assert resp.latency_ms["dense"] == pytest.approx(12.3)
    assert set(resp.latency_ms) == set(timings)
    dumped = resp.model_dump()
    assert dumped["latency_ms"]["gen"] == pytest.approx(4800.0)


def test_query_response_latency_ms_accepts_arbitrary_keys() -> None:
    """Free-form stage names so future stages (guardrails_in/out) don't need schema changes."""
    resp = QueryResponse(
        answer="ok",
        confidence=0.5,
        latency_ms={"guardrails_in": 3.1, "total": 5000.0},
    )
    assert resp.latency_ms == {"guardrails_in": 3.1, "total": 5000.0}


def test_query_response_extra_top_level_key_still_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryResponse(answer="ok", confidence=0.5, unknown_field=42)  # type: ignore[call-arg]
