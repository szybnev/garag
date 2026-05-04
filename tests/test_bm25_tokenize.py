"""Invariants for the BM25 tokenizer in `scripts.build_bm25`."""

from __future__ import annotations

import pandas as pd

from scripts.build_bm25 import searchable_text, tokenize

STOP = {"the", "a", "an", "is", "and", "of", "in", "for", "to"}


def test_lowercases_input() -> None:
    assert tokenize("MITRE ATT&CK", STOP) == ["mitre", "att", "ck"]


def test_strips_urls() -> None:
    text = "see https://attack.mitre.org/techniques/T1059/ for details"
    tokens = tokenize(text, STOP)
    assert "attack" not in tokens or "mitre" not in tokens or "techniques" not in tokens
    assert "see" in tokens
    assert "details" in tokens


def test_drops_stopwords() -> None:
    tokens = tokenize("the impact of the vulnerability is critical", STOP)
    assert "the" not in tokens
    assert "of" not in tokens
    assert "is" not in tokens
    assert "impact" in tokens
    assert "vulnerability" in tokens
    assert "critical" in tokens


def test_keeps_alphanumeric_words() -> None:
    tokens = tokenize("CVE-2023-1234 affects bge-m3 v1", STOP)
    assert "cve-2023-1234" in tokens
    assert "bge-m3" in tokens


def test_keeps_mitre_subtechnique_ids() -> None:
    tokens = tokenize("T1059.001 uses PowerShell", STOP)

    assert "t1059.001" in tokens
    assert "t1059" not in tokens
    assert "001" not in tokens


def test_searchable_text_includes_chunk_identifiers_and_title() -> None:
    row = pd.Series(
        {
            "chunk_id": "mitre_attack:T1134::0",
            "doc_id": "mitre_attack:T1134",
            "source": "mitre_attack",
            "title": "Access Token Manipulation",
            "text": "Adversaries may modify access tokens.",
        }
    )

    tokens = tokenize(searchable_text(row), STOP)

    assert "t1134" in tokens
    assert "mitre" in tokens
    assert "attack" in tokens
    assert "access" in tokens
    assert "token" in tokens
    assert "manipulation" in tokens


def test_searchable_text_includes_mitre_subtechnique_id() -> None:
    row = pd.Series(
        {
            "chunk_id": "mitre_attack:T1059.001::0",
            "doc_id": "mitre_attack:T1059.001",
            "source": "mitre_attack",
            "title": "PowerShell",
            "text": "Adversaries may abuse PowerShell commands.",
        }
    )

    tokens = tokenize(searchable_text(row), STOP)

    assert "t1059.001" in tokens
    assert "powershell" in tokens


def test_empty_input() -> None:
    assert tokenize("", STOP) == []


def test_only_punctuation() -> None:
    assert tokenize(",.;:!?", STOP) == []


def test_word_trailing_period_is_not_part_of_token() -> None:
    assert tokenize("foo. bar", STOP) == ["foo", "bar"]


def test_short_single_letter_dropped() -> None:
    tokens = tokenize("a b c d hello world", STOP)
    assert "a" not in tokens
    assert "b" not in tokens
    assert "hello" in tokens
    assert "world" in tokens
