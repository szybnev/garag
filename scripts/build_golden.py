"""Build the golden Q/A set from chunks via LLM-assisted generation.

For each sampled chunk we ask `qwen3.5:35b` (or whatever `--judge-model`
points to) to write **one** Q/A pair grounded in that chunk's text. The
prompt enforces a category (factual / tool_usage / multi_hop) and JSON
output. We do basic validation (non-empty fields, no obvious echo of the
prompt) and append to `data/golden/golden_set_v1.jsonl`.

Stratified sampling layout (default `--target 25` for d5):

- 10 factual:    mitre_attack, mitre_atlas, owasp
- 8  tool_usage: man_pages
- 7  multi_hop:  mitre_attack (longer techniques)

`--target 50` extends this on d6 with more pairs from each category.
The script is **append-only** by default — re-running it adds new pairs
without overwriting existing ones, identified by `qid`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.parquet"
GOLDEN_FILE = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set_v1.jsonl"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:35b"

CATEGORY_SPECS = {
    "factual": {
        "sources": ["mitre_attack", "mitre_atlas", "owasp"],
        "min_tokens": 60,
        "instruction": (
            "Write a SHORT factual question whose answer is a single fact "
            "(name, ID, definition, classification) extractable from this snippet. "
            "Avoid yes/no questions."
        ),
    },
    "tool_usage": {
        "sources": ["man_pages"],
        "min_tokens": 60,
        "instruction": (
            "Write a SHORT how-to question about using this command-line tool. "
            "It must reference a specific flag, option, or invocation pattern "
            "described in the snippet."
        ),
    },
    "multi_hop": {
        "sources": ["mitre_attack", "owasp"],
        "min_tokens": 120,
        "instruction": (
            "Write a question that requires combining at least TWO distinct facts "
            "from the snippet (for example: a technique AND its mitigation, or a "
            "vulnerability AND its consequence)."
        ),
    },
}

CATEGORY_TARGETS_25 = {"factual": 10, "tool_usage": 8, "multi_hop": 7}
CATEGORY_TARGETS_50 = {"factual": 20, "tool_usage": 15, "multi_hop": 15}


def _system_prompt() -> str:
    return (
        "You are a cybersecurity educator generating evaluation questions for a RAG "
        "system. You receive a text snippet and a category. You output exactly one "
        "JSON object with keys: question, answer, category. The answer must be "
        "directly supported by the snippet. Keep both fields concise (≤2 sentences)."
    )


def _user_prompt(category: str, source: str, title: str, text: str) -> str:
    spec = CATEGORY_SPECS[category]
    return (
        f"Category: {category}\n"
        f"Source: {source}\n"
        f"Title: {title}\n\n"
        f"Snippet:\n```\n{text.strip()[:2500]}\n```\n\n"
        f"Instruction: {spec['instruction']}\n\n"
        f"Output ONE JSON object only, no markdown, no commentary."
    )


def _call_ollama(
    client: httpx.Client,
    model: str,
    system: str,
    user: str,
) -> str:
    resp = client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "top_p": 0.9, "seed": 42},
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("message", {}).get("content", ""))


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_qa(raw: str) -> dict[str, str] | None:
    raw = raw.strip()
    if not raw:
        return None
    obj = _try_load_json(raw)
    if not isinstance(obj, dict):
        return None
    q = str(obj.get("question", "")).strip()
    a = str(obj.get("answer", "")).strip()
    if len(q) < 5 or len(a) < 3 or q == a:
        return None
    return {"question": q, "answer": a}


def _try_load_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _stratified_sample(
    chunks: pd.DataFrame,
    targets: dict[str, int],
    rng: random.Random,
) -> list[tuple[str, dict[str, Any]]]:
    """Pick (category, chunk_row) pairs to send to the LLM."""
    samples: list[tuple[str, dict[str, Any]]] = []
    for category, requested in targets.items():
        spec = CATEGORY_SPECS[category]
        pool = chunks[
            chunks["source"].isin(spec["sources"]) & (chunks["token_count"] >= spec["min_tokens"])
        ]
        n_take = requested
        if len(pool) < requested:
            print(f"[warn] {category}: pool only has {len(pool)}, need {requested}")
            n_take = len(pool)
        picked = pool.sample(n=n_take, random_state=rng.randint(0, 2**31 - 1))
        for _, row in picked.iterrows():
            samples.append((category, row.to_dict()))
    return samples


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[obj["qid"]] = obj
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=CHUNKS_FILE)
    ap.add_argument("--output", type=Path, default=GOLDEN_FILE)
    ap.add_argument("--target", type=int, default=25, choices=[25, 50])
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"[load] {len(df)} chunks from {args.input}")

    existing = _load_existing(args.output)
    print(f"[existing] {len(existing)} qid in {args.output}")

    targets = CATEGORY_TARGETS_25 if args.target == 25 else CATEGORY_TARGETS_50
    # subtract already-collected per-category to avoid duplication
    have = {"factual": 0, "tool_usage": 0, "multi_hop": 0}
    for obj in existing.values():
        cat = obj.get("category", "factual")
        have[cat] = have.get(cat, 0) + 1
    todo = {c: max(0, targets[c] - have[c]) for c in targets}
    print(f"[targets] need: {todo} (existing: {have})")

    if sum(todo.values()) == 0:
        print("[done] target already reached")
        return

    rng = random.Random(args.seed)  # noqa: S311 — reproducibility seed, not crypto
    samples = _stratified_sample(df, todo, rng)
    print(f"[sample] {len(samples)} chunks to query")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    next_idx = max((int(qid[1:]) for qid in existing if qid.startswith("g")), default=0) + 1

    new_pairs: list[dict[str, Any]] = []
    with httpx.Client(base_url=args.ollama_url, timeout=300.0) as client:
        for i, (category, row) in enumerate(samples, start=1):
            t0 = time.perf_counter()
            try:
                raw = _call_ollama(
                    client,
                    args.model,
                    _system_prompt(),
                    _user_prompt(category, row["source"], row["title"], row["text"]),
                )
            except httpx.HTTPError as exc:
                print(f"  [{i:>2}/{len(samples)}] {category}: HTTP error: {exc}")
                continue
            qa = _parse_qa(raw)
            dt = time.perf_counter() - t0
            if not qa:
                print(f"  [{i:>2}/{len(samples)}] {category}: parse failed ({dt:.1f}s)")
                continue
            qid = f"g{next_idx:03d}"
            next_idx += 1
            pair = {
                "qid": qid,
                "question": qa["question"],
                "answer": qa["answer"],
                "category": category,
                "source": row["source"],
                "source_chunk_id": row["chunk_id"],
                "relevant_chunks": [row["chunk_id"]],
            }
            new_pairs.append(pair)
            print(
                f"  [{i:>2}/{len(samples)}] {category:<10} {qid} ({dt:.1f}s) "
                f"q={qa['question'][:60]!r}"
            )

    with args.output.open("a") as fh:
        for pair in new_pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[done] appended {len(new_pairs)} pairs to {args.output}")
    print(f"[total] {len(existing) + len(new_pairs)} pairs in golden set")


if __name__ == "__main__":
    main()
