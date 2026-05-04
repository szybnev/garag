"""LLM-as-judge — `qwen3.5:35b` via native Ollama `/api/chat`.

Mirror of `app.rag.generator.Generator`: same transport (native
`/api/chat`), same `think=false` flag, same structured output pattern.
The difference is the system prompt (a three-dimension rubric) and the
output schema (`_JudgeVerdict`): three integer scores in `[0, 2]` plus
a short rationale.

Why integer 0..2: binary 0/1 loses resolution on partially correct
answers, 0-5 introduces spurious precision noise on qwen3.5, and the
rubric maps cleanly onto "unsupported / partially supported / fully
supported". Downstream code normalises `score / 2` to `[0, 1]` when
comparing against the NFR thresholds in `docs/design.md §3`.

Why we ship this as a class (not an inline helper inside
`scripts/eval_generation.py`): `.25` (garak scan), `.26` (NFR benchmark),
and a future cross-model judge all need the same entry point. Keeping it
reusable also makes it unit-testable via
`httpx.MockTransport` the same way `Generator` is.

## Self-bias caveat

GaRAG uses `qwen3.5:35b` as both generator and judge because it is the
largest Ollama-available model on the d9 host. Literature on LLM-as-judge
(e.g. Zheng et al. 2023 "Judging LLM-as-a-Judge") reports self-bias of
5-15% on the faithfulness dimension when generator and judge share a
checkpoint. We accept this for the MVP and document it on every report
this class feeds. Cross-model rerun (GPT-4o or Claude as judge) is deferred.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.rag.generator import _format_context

if TYPE_CHECKING:
    from app.rag import ScoredChunk
    from app.schemas import Citation, QueryResponse


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of cybersecurity Q&A systems.
Given a question, a reference answer (golden), a candidate answer, and the
retrieval context, you assign three integer scores in [0, 2] and a brief
rationale.

Scoring rubric:

faithfulness  (is the candidate answer supported by the provided context?)
  0 — claims not in context or contradict it
  1 — partially supported; at least one claim unsupported or hedged
  2 — every factual claim directly supported by the context

correctness  (does the candidate answer match the reference answer on substance?)
  0 — wrong or off-topic
  1 — partially correct (right topic, wrong details, or missing key fact)
  2 — semantically equivalent to the reference (paraphrase is fine)

citation_support  (do cited chunks actually back the quoted claims?)
  0 — citations missing, wrong, or do not support the quoted text
  1 — citations exist but at least one is loosely related / off-topic
  2 — every citation is on-point and directly supports its quoted claim

Output a single JSON object with keys: faithfulness, correctness,
citation_support, rationale. Keep rationale under 400 characters.
No markdown fences, no extra keys, no preamble. Be strict: when in
doubt, score lower."""


class _JudgeVerdict(BaseModel):
    """Judge-side output schema — narrow on purpose.

    Three flat integers + one short string means the Ollama structured
    output decoder has the smallest possible surface area to violate.
    This is the same lesson d9 taught us with `_GeneratedResponse`:
    nested `required` fields on MoE qwen3.5 are leaky; flat primitives
    are robust.

    `rationale` is bounded at 1200 chars. The system prompt asks for
    "<400 characters" but qwen3.5:35b consistently overruns on ~10% of
    multi-hop / tool_usage queries with 500-900 char explanations.
    Letting those through (rather than rejecting) keeps the eval pass
    rate high; the rationale is debugging-only and doesn't feed any
    numeric metric.
    """

    model_config = ConfigDict(extra="forbid")

    faithfulness: int = Field(ge=0, le=2)
    correctness: int = Field(ge=0, le=2)
    citation_support: int = Field(ge=0, le=2)
    rationale: str = Field(min_length=1, max_length=1200)


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return "(no citations)"
    lines: list[str] = []
    for i, c in enumerate(citations, start=1):
        snippet = c.quote.strip().replace("\r\n", "\n")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        lines.append(f"[{i}] chunk_id={c.chunk_id}\n    quote: {snippet}")
    return "\n".join(lines)


def _build_judge_user_message(
    *,
    question: str,
    golden: str,
    candidate: QueryResponse,
    chunks: list[ScoredChunk],
) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Reference answer (golden):\n{golden}\n\n"
        f"Candidate answer:\n{candidate.answer}\n\n"
        f"Candidate citations:\n{_format_citations(list(candidate.citations))}\n\n"
        f"Retrieval context (numbered sources available to the candidate):\n"
        f"{_format_context(chunks)}\n\n"
        "Evaluate the candidate answer against the rubric. Respond with one JSON object."
    )


class JudgeError(RuntimeError):
    """Raised when the judge returns an unusable response."""


class Judge:
    """qwen3.5 LLM-as-judge over the native Ollama `/api/chat` endpoint."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_predict: int = 300,
        seed: int = 42,
        keep_alive: str | None = None,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.model = model or settings.ollama_judge_model
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.seed = seed
        self.keep_alive = keep_alive or settings.ollama_keep_alive
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        self._get_client()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _build_payload(
        self,
        *,
        question: str,
        golden: str,
        candidate: QueryResponse,
        chunks: list[ScoredChunk],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_judge_user_message(
                        question=question,
                        golden=golden,
                        candidate=candidate,
                        chunks=chunks,
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "format": _JudgeVerdict.model_json_schema(),
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }

    def judge(
        self,
        *,
        question: str,
        golden: str,
        candidate: QueryResponse,
        chunks: list[ScoredChunk],
    ) -> _JudgeVerdict:
        """Score the candidate answer against golden + context."""
        client = self._get_client()
        payload = self._build_payload(
            question=question, golden=golden, candidate=candidate, chunks=chunks
        )
        try:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise JudgeError(f"Ollama /api/chat failed: {exc}") from exc

        body = resp.json()
        content = (body.get("message") or {}).get("content") or ""
        if not content.strip():
            raise JudgeError(
                "Ollama returned empty message.content — check that `think=false` "
                f"is honored by the judge model (body keys: {sorted(body)})"
            )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise JudgeError(
                f"Judge response is not valid JSON despite format schema: {exc}\n"
                f"content={content[:400]!r}"
            ) from exc
        try:
            return _JudgeVerdict.model_validate(parsed)
        except ValueError as exc:
            raise JudgeError(
                f"Judge JSON does not match _JudgeVerdict schema: {exc}\nparsed={parsed!r}"
            ) from exc
