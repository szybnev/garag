"""Mock-based tests for `Judge` — parallel to `test_generator.py`.

Uses `httpx.MockTransport` so the tests never talk to a real Ollama.
Exercises payload shape (think=false, temperature=0, format schema)
and parser error paths.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.rag import ScoredChunk
from app.rag.judge import (
    JUDGE_SYSTEM_PROMPT,
    Judge,
    JudgeError,
    _JudgeVerdict,
)
from app.schemas import Citation, QueryResponse


def _chunk(cid: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        score=1.0,
        source="mitre_attack",
        title=f"title-{cid}",
        text=f"text of {cid}",
        url=f"https://example.test/{cid}",
        doc_id="mitre_attack:T0",
    )


def _candidate() -> QueryResponse:
    return QueryResponse(
        answer="PowerShell can be abused by adversaries.",
        citations=[
            Citation(
                chunk_id="mitre_attack:T0::0",
                source="mitre_attack",
                url="https://example.test/T0",
                quote="Adversaries may abuse PowerShell commands.",
            )
        ],
        confidence=0.9,
        used_chunks=["mitre_attack:T0::0"],
    )


def _valid_verdict_json() -> str:
    return json.dumps(
        {
            "faithfulness": 2,
            "correctness": 2,
            "citation_support": 2,
            "rationale": "Every claim directly supported by chunk T0::0.",
        }
    )


def _make_judge(handler: httpx.MockTransport) -> Judge:
    return Judge(
        base_url="http://ollama.test:11434",
        model="qwen3.5:judge-test",
        client=httpx.Client(transport=handler),
    )


def test_judge_parses_valid_verdict() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": _valid_verdict_json()}})

    judge = _make_judge(httpx.MockTransport(handler))
    verdict = judge.judge(
        question="What is PowerShell abuse?",
        golden="PowerShell is abused by adversaries for execution.",
        candidate=_candidate(),
        chunks=[_chunk("mitre_attack:T0::0")],
    )
    assert isinstance(verdict, _JudgeVerdict)
    assert verdict.faithfulness == 2
    assert verdict.correctness == 2
    assert verdict.citation_support == 2


def test_judge_payload_has_think_false_and_temperature_zero() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": _valid_verdict_json()}})

    judge = _make_judge(httpx.MockTransport(handler))
    judge.judge(
        question="q",
        golden="g",
        candidate=_candidate(),
        chunks=[_chunk("mitre_attack:T0::0")],
    )
    payload = captured["payload"]
    assert payload["model"] == "qwen3.5:judge-test"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == _JudgeVerdict.model_json_schema()
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 42
    assert payload["messages"][0] == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    assert "PowerShell" in payload["messages"][1]["content"]


def test_openai_compat_judge_payload_and_response() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _valid_verdict_json(),
                        }
                    }
                ]
            },
        )

    judge = Judge(
        provider="openai_compat",
        base_url="http://lmstudio.test:1234/v1",
        model="zai-org/glm-4.7-flash",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    verdict = judge.judge(
        question="q",
        golden="g",
        candidate=_candidate(),
        chunks=[_chunk("mitre_attack:T0::0")],
    )

    assert verdict.faithfulness == 2
    assert captured["url"] == "http://lmstudio.test:1234/v1/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == "zai-org/glm-4.7-flash"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 300
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == _JudgeVerdict.model_json_schema()


def test_judge_raises_on_schema_mismatch() -> None:
    """Out-of-range faithfulness must be rejected by Pydantic validator."""

    def handler(_request: httpx.Request) -> httpx.Response:
        bad = json.dumps(
            {
                "faithfulness": 3,
                "correctness": 1,
                "citation_support": 1,
                "rationale": "too high",
            }
        )
        return httpx.Response(200, json={"message": {"content": bad}})

    judge = _make_judge(httpx.MockTransport(handler))
    with pytest.raises(JudgeError, match="does not match _JudgeVerdict schema"):
        judge.judge(
            question="q",
            golden="g",
            candidate=_candidate(),
            chunks=[_chunk("mitre_attack:T0::0")],
        )


def test_judge_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    judge = _make_judge(httpx.MockTransport(handler))
    with pytest.raises(JudgeError, match="Ollama /api/chat failed"):
        judge.judge(
            question="q",
            golden="g",
            candidate=_candidate(),
            chunks=[_chunk("mitre_attack:T0::0")],
        )
