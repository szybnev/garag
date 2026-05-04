"""LLM generator — qwen-family local LLM endpoints.

The retrieval pipeline returns a list of `ScoredChunk` candidates; this
module turns them into a grounded `QueryResponse` (answer + citations +
self-reported confidence) by calling a qwen model through either native Ollama
`/api/chat` or an OpenAI-compatible local server such as LM Studio.

## Why native `/api/chat` instead of the OpenAI-compatible endpoint

qwen reasoning models support a "thinking" mode. The OpenAI-compatible endpoint in
Ollama can silently return an empty `content` field with the entire
answer hidden inside `thinking`, which breaks any JSON parsing. The
native `/api/chat` endpoint accepts a top-level `think: false` flag
that disables the behavior end-to-end. See `hw12-advanced-rag` in the
parent `gigaschool` repo for the original discovery.

## Structured output + post-hoc hydration

Ollama supports JSON-schema constrained decoding via the top-level
`format` field on `/api/chat`. We feed it `_GeneratedResponse`, a
narrower shape that only asks the LLM for the bits it actually has to
originate: `answer`, `citations[].chunk_id`, `citations[].quote`,
`confidence`, and `used_chunks`. The public-facing `Citation.source`
and `Citation.url` are then hydrated from the retrieved chunks we
already have on hand.

Why the split: on d9 a first cut that asked qwen3.5:35b to emit the
full `QueryResponse.model_json_schema()` (including `source` / `url`
per citation) failed parsing on 20/20 golden queries — the MoE decoder
consistently dropped the `source` field despite it being marked
`required`. Hydrating those fields post-hoc removes an entire class of
structured-output flakiness and, as a bonus, makes it impossible for
the LLM to hallucinate a URL: we only carry over URLs attached to
chunks that were actually retrieved.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Self, cast

import httpx
from json_repair import repair_json
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.schemas import Citation, QueryResponse, SourceName

if TYPE_CHECKING:
    from app.rag import ScoredChunk


class _GeneratedCitation(BaseModel):
    """LLM-side citation shape — narrower than the public `Citation`.

    The LLM only has to emit `chunk_id` (which it can copy verbatim from
    the context header) and a short `quote` from the chunk body. The
    `source` / `url` fields of the public `Citation` are filled in
    post-hoc from the retrieved chunks we already have on hand. This
    keeps qwen3.5:35b's MoE decoder from dropping required fields under
    Ollama structured output (observed on d9 on 20/20 golden queries).
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=3)
    quote: str = Field(min_length=1, max_length=600)


class _GeneratedResponse(BaseModel):
    """The JSON shape we actually ask the LLM to emit.

    `answer`, `confidence`, and `used_chunks` match `QueryResponse`
    one-to-one; `citations` uses the narrower `_GeneratedCitation` shape
    so that we can enrich it with `source` / `url` post-hoc without
    risking LLM hallucination on those fields.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[_GeneratedCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    used_chunks: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """You are GaRAG, a careful cybersecurity research assistant.
You answer questions using ONLY the numbered sources in the provided context.

Rules:
1. Ground every factual claim in at least one source. If the context does not
   contain the answer, reply exactly: "The provided sources do not contain
   enough information to answer this question." and set confidence <= 0.2.
2. Never invent CVE IDs, CWE numbers, ATT&CK technique IDs, bounty amounts,
   tool flags, or URLs. If a detail is not in the sources, leave it out.
3. For every `citations[]` entry, copy the `chunk_id` VERBATIM from the
   context header (e.g. `chunk_id=mitre_attack:T1059.001::0`) and a short
   verbatim `quote` (<= 280 chars) from that same chunk's body.
4. `used_chunks` must list exactly the `chunk_id`s of the citations you
   emitted, in the same order.
5. `confidence` is your self-assessment in [0, 1]: 0.9+ only when every claim
   is directly supported; 0.5-0.8 when you had to synthesize across sources;
   <= 0.3 when the context is weak or tangential.
6. Keep the answer concise (1-6 sentences). No preamble, no meta commentary,
   no "As an AI" disclaimers.
7. Do not include bracket citation markers like [1] in `answer`; citations are
   rendered separately from the `citations` array.
8. Output MUST be a single JSON object with keys: `answer`, `citations`,
   `confidence`, `used_chunks`. Each `citations` entry has ONLY `chunk_id`
   and `quote`. No markdown fences, no extra keys, no trailing text."""


def _format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as a numbered, citation-friendly block.

    Each chunk gets an explicit `chunk_id` line so the model can reference
    it verbatim in `citations[].chunk_id` and `used_chunks`. Text is
    truncated defensively — the retrieval layer already bounds chunk size,
    but a long stray chunk would eat the generation budget.
    """
    if not chunks:
        return "(no sources retrieved)"
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        snippet = c.text.strip().replace("\r\n", "\n")
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "…"
        header = f"[{i}] chunk_id={c.chunk_id} source={c.source} title={c.title}"
        if c.url:
            header += f" url={c.url}"
        lines.append(f"{header}\n{snippet}")
    return "\n\n".join(lines)


def _build_user_message(query: str, chunks: list[ScoredChunk]) -> str:
    context = _format_context(chunks)
    return (
        f"Question:\n{query}\n\n"
        f"Context (numbered sources):\n{context}\n\n"
        "Answer the question using only the sources above, following the "
        "rules in the system prompt. Respond with one JSON object."
    )


def _hydrate_response(
    generated: _GeneratedResponse,
    chunks: list[ScoredChunk],
) -> QueryResponse:
    """Attach `source` / `url` to LLM-emitted citations from the retrieved chunks.

    The LLM only emits `chunk_id` + `quote`; we look the chunk up in the
    retrieved set and carry its `source` / `url` across. Citations with a
    `chunk_id` that was not actually in the retrieved set are dropped —
    they would be ungrounded by definition. `used_chunks` is likewise
    filtered to the subset that survived hydration, so downstream
    guardrails can trust it as a real citation list.
    """
    by_id: dict[str, ScoredChunk] = {c.chunk_id: c for c in chunks}
    hydrated: list[Citation] = []
    for cit in generated.citations:
        chunk = by_id.get(cit.chunk_id)
        if chunk is None:
            continue
        hydrated.append(
            Citation(
                chunk_id=cit.chunk_id,
                source=cast("SourceName", chunk.source),
                url=chunk.url,
                quote=cit.quote[:600],
            )
        )
    if not hydrated and generated.answer.strip() and chunks:
        fallback_ids = [cid for cid in generated.used_chunks if cid in by_id]
        fallback_chunks = [by_id[cid] for cid in fallback_ids] or chunks[:1]
        hydrated.extend(
            Citation(
                chunk_id=chunk.chunk_id,
                source=cast("SourceName", chunk.source),
                url=chunk.url,
                quote=_fallback_quote(chunk.text),
            )
            for chunk in fallback_chunks[:3]
        )
    valid_ids = {c.chunk_id for c in hydrated}
    used = list(dict.fromkeys(cid for cid in generated.used_chunks if cid in valid_ids))
    if not used:
        used = [c.chunk_id for c in hydrated]
    return QueryResponse(
        answer=generated.answer,
        citations=hydrated,
        confidence=generated.confidence,
        used_chunks=used,
    )


def _fallback_quote(text: str) -> str:
    quote = " ".join(text.strip().split())
    if len(quote) > 280:
        quote = quote[:277].rstrip() + "..."
    return quote or "Retrieved source chunk."


class GenerationError(RuntimeError):
    """Raised when the LLM backend returns an unusable response."""


class _StructuredOutputParseError(GenerationError):
    """Raised when malformed structured output could not be parsed or repaired."""


class Generator:
    """qwen-family generator over native Ollama or OpenAI-compatible endpoints."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_predict: int | None = None,
        seed: int | None = None,
        keep_alive: str | None = None,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider = provider or settings.llm_provider
        if self.provider == "openai_compat":
            self.base_url = (base_url or settings.openai_base_url).rstrip("/")
            self.model = model or settings.openai_model
        elif self.provider == "ollama":
            self.base_url = (base_url or settings.ollama_url).rstrip("/")
            self.model = model or settings.ollama_model
        else:
            msg = f"unsupported LLM provider: {self.provider}"
            raise ValueError(msg)
        self.temperature = temperature if temperature is not None else settings.gen_temperature
        self.top_p = top_p if top_p is not None else settings.gen_top_p
        self.num_predict = num_predict if num_predict is not None else settings.gen_num_predict
        self.seed = seed if seed is not None else settings.gen_seed
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

    def _build_ollama_payload(self, query: str, chunks: list[ScoredChunk]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query, chunks)},
            ],
            "stream": False,
            "think": False,
            "format": _GeneratedResponse.model_json_schema(),
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }

    def _build_openai_payload(self, query: str, chunks: list[ScoredChunk]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query, chunks)},
            ],
            "stream": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.num_predict,
            "seed": self.seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "garag_generated_response",
                    "strict": True,
                    "schema": _GeneratedResponse.model_json_schema(),
                },
            },
        }

    def generate(self, query: str, chunks: list[ScoredChunk]) -> QueryResponse:
        """Call the configured LLM backend and parse JSON into `QueryResponse`.

        Raises `GenerationError` on HTTP failure or on JSON that cannot be
        validated against the schema. Malformed structured JSON gets one retry:
        LM Studio can occasionally truncate schema-constrained content under
        concurrent load, while valid-but-schema-wrong JSON still fails fast.
        """
        client = self._get_client()
        last_parse_error: _StructuredOutputParseError | None = None
        for attempt in range(2):
            content = self._call_backend(client, query, chunks)
            if not content.strip():
                raise GenerationError(
                    f"{self.provider} returned empty message content"
                )
            try:
                generated = self._parse_generated_response(
                    content,
                    allow_repair=attempt > 0,
                )
            except _StructuredOutputParseError as exc:
                last_parse_error = exc
                continue
            return _hydrate_response(generated, chunks)

        if last_parse_error is not None:
            raise last_parse_error
        raise GenerationError(f"{self.provider} failed to produce structured output")

    def _call_backend(
        self,
        client: httpx.Client,
        query: str,
        chunks: list[ScoredChunk],
    ) -> str:
        if self.provider == "openai_compat":
            return self._call_openai_compat(client, query, chunks)
        return self._call_ollama(client, query, chunks)

    def _parse_generated_response(
        self,
        content: str,
        *,
        allow_repair: bool,
    ) -> _GeneratedResponse:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            if not allow_repair:
                raise _StructuredOutputParseError(
                    f"{self.provider} response is not valid JSON despite schema prompt: "
                    f"{exc}\ncontent={content[:400]!r}"
                ) from exc
            parsed = self._repair_generated_json(content, exc)
        try:
            return _GeneratedResponse.model_validate(parsed)
        except ValueError as exc:
            raise GenerationError(
                f"{self.provider} JSON does not match _GeneratedResponse schema: "
                f"{exc}\nparsed={parsed!r}"
            ) from exc

    def _repair_generated_json(
        self,
        content: str,
        parse_error: json.JSONDecodeError,
    ) -> Any:
        try:
            repaired = repair_json(
                content,
                return_objects=True,
                ensure_ascii=False,
            )
        except Exception as exc:
            raise _StructuredOutputParseError(
                f"{self.provider} response is not valid JSON despite schema prompt: "
                f"{parse_error}\ncontent={content[:400]!r}"
            ) from exc
        try:
            _GeneratedResponse.model_validate(repaired)
        except ValueError as exc:
            raise _StructuredOutputParseError(
                f"{self.provider} response is not valid JSON despite schema prompt: "
                f"{parse_error}; repair did not produce _GeneratedResponse: {exc}\n"
                f"content={content[:400]!r}"
            ) from parse_error
        return repaired

    def _call_ollama(
        self,
        client: httpx.Client,
        query: str,
        chunks: list[ScoredChunk],
    ) -> str:
        payload = self._build_ollama_payload(query, chunks)
        try:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise GenerationError(f"Ollama /api/chat failed: {exc}") from exc
        body = resp.json()
        return (body.get("message") or {}).get("content") or ""

    def _call_openai_compat(
        self,
        client: httpx.Client,
        query: str,
        chunks: list[ScoredChunk],
    ) -> str:
        payload = self._build_openai_payload(query, chunks)
        try:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"OpenAI-compatible /chat/completions failed: {exc}"
            ) from exc
        body = resp.json()
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, str):
            return content
        return json.dumps(content)
