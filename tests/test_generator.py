"""Mock-based tests for `Generator`.

We don't want the unit suite to spin up Ollama or download a 14B model,
so we swap the underlying `httpx.Client` for one wired to
`httpx.MockTransport`. That lets us assert the request payload the
generator builds (model, messages, `think=false`, `format` schema) and
drive the parser against canned responses without any network.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.rag import ScoredChunk
from app.rag.generator import (
    SYSTEM_PROMPT,
    GenerationError,
    Generator,
    _format_context,
    _GeneratedResponse,
)
from app.schemas import QueryResponse


def _chunk(cid: str, text: str = "example chunk text") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        score=1.0,
        source="mitre_attack",
        title=f"title-{cid}",
        text=text,
        url=f"https://example.test/{cid}",
        doc_id="mitre_attack:T0",
    )


def _valid_response_json(chunk_id: str = "mitre_attack:T0::0") -> str:
    return json.dumps(
        {
            "answer": "PowerShell can be abused by adversaries for execution.",
            "citations": [
                {
                    "chunk_id": chunk_id,
                    "quote": "Adversaries may abuse PowerShell commands.",
                }
            ],
            "confidence": 0.9,
            "used_chunks": [chunk_id],
        }
    )


def _make_generator(
    handler: httpx.MockTransport,
    **overrides: Any,
) -> Generator:
    client = httpx.Client(transport=handler)
    return Generator(
        base_url="http://ollama.test:11434",
        model="qwen3.5:test",
        provider="ollama",
        client=client,
        **overrides,
    )


def test_generate_parses_valid_response() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": _valid_response_json()}},
        )

    gen = _make_generator(httpx.MockTransport(handler))
    chunks = [_chunk("mitre_attack:T0::0")]
    result = gen.generate("What is PowerShell abuse?", chunks)

    assert isinstance(result, QueryResponse)
    assert result.confidence == pytest.approx(0.9)
    assert result.used_chunks == ["mitre_attack:T0::0"]
    assert len(result.citations) == 1
    # source + url hydrated from the retrieved chunk, not emitted by the LLM
    assert result.citations[0].source == "mitre_attack"
    assert result.citations[0].url == "https://example.test/mitre_attack:T0::0"
    assert captured["url"] == "http://ollama.test:11434/api/chat"


def test_generate_payload_has_think_false_and_format_schema() -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"content": _valid_response_json()}},
        )

    gen = _make_generator(httpx.MockTransport(handler))
    gen.generate("query", [_chunk("mitre_attack:T0::0")])

    payload = payloads[0]
    assert payload["model"] == "qwen3.5:test"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == _GeneratedResponse.model_json_schema()
    assert payload["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "mitre_attack:T0::0" in payload["messages"][1]["content"]
    assert payload["options"]["temperature"] == pytest.approx(gen.temperature)
    assert payload["options"]["seed"] == gen.seed


def test_generate_openai_compat_payload_and_response() -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": _valid_response_json(),
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gen = Generator(
        base_url="http://lmstudio.test:1234/v1",
        model="qwen/qwen3.6-35b-a3b",
        provider="openai_compat",
        client=client,
    )
    result = gen.generate("query", [_chunk("mitre_attack:T0::0")])

    assert result.confidence == pytest.approx(0.9)
    payload = payloads[0]
    assert payload["model"] == "qwen/qwen3.6-35b-a3b"
    assert payload["stream"] is False
    assert payload["max_tokens"] == gen.num_predict
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == (
        _GeneratedResponse.model_json_schema()
    )


def test_generate_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    gen = _make_generator(httpx.MockTransport(handler))
    with pytest.raises(GenerationError, match="Ollama /api/chat failed"):
        gen.generate("q", [_chunk("mitre_attack:T0::0")])


def test_generate_raises_on_empty_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": ""}})

    gen = _make_generator(httpx.MockTransport(handler))
    with pytest.raises(GenerationError, match="empty message content"):
        gen.generate("q", [_chunk("mitre_attack:T0::0")])


def test_generate_raises_on_invalid_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "not a json { at all"}},
        )

    gen = _make_generator(httpx.MockTransport(handler))
    with pytest.raises(GenerationError, match="not valid JSON"):
        gen.generate("q", [_chunk("mitre_attack:T0::0")])


def test_generate_raises_on_schema_mismatch() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        bad = json.dumps({"answer": "hi", "confidence": 2.5})
        return httpx.Response(200, json={"message": {"content": bad}})

    gen = _make_generator(httpx.MockTransport(handler))
    with pytest.raises(GenerationError, match="does not match _GeneratedResponse schema"):
        gen.generate("q", [_chunk("mitre_attack:T0::0")])


def test_generate_drops_ungrounded_citations() -> None:
    """LLM-emitted chunk_ids not in the retrieved set get filtered out."""
    payload_json = json.dumps(
        {
            "answer": "Partially grounded answer.",
            "citations": [
                {"chunk_id": "mitre_attack:T0::0", "quote": "real quote"},
                {"chunk_id": "mitre_attack:HALLUCINATED::9", "quote": "fake"},
            ],
            "confidence": 0.7,
            "used_chunks": [
                "mitre_attack:T0::0",
                "mitre_attack:HALLUCINATED::9",
            ],
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": payload_json}})

    gen = _make_generator(httpx.MockTransport(handler))
    result = gen.generate("q", [_chunk("mitre_attack:T0::0")])

    assert [c.chunk_id for c in result.citations] == ["mitre_attack:T0::0"]
    assert result.used_chunks == ["mitre_attack:T0::0"]


def test_format_context_lists_all_chunks() -> None:
    chunks = [_chunk("a", "alpha"), _chunk("b", "bravo")]
    rendered = _format_context(chunks)
    assert "[1]" in rendered
    assert "[2]" in rendered
    assert "alpha" in rendered
    assert "bravo" in rendered
    assert "chunk_id=a" in rendered
    assert "chunk_id=b" in rendered


def test_format_context_truncates_long_text() -> None:
    big = "x" * 5000
    rendered = _format_context([_chunk("big", big)])
    assert len(rendered) < 2000
    assert rendered.endswith("…")


def test_format_context_empty() -> None:
    assert _format_context([]) == "(no sources retrieved)"
