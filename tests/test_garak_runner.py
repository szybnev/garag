"""Tests for the garak runner wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.run_garak import build_command, build_generator_options


def test_build_generator_options_targets_query_endpoint() -> None:
    options = build_generator_options("http://api.test/", top_k=3, timeout_s=120)

    assert options["rest"]["uri"] == "http://api.test/query"
    assert options["rest"]["method"] == "post"
    assert options["rest"]["req_template_json_object"] == {"query": "$INPUT", "top_k": 3}
    assert options["rest"]["response_json_field"] == "$.answer"
    assert options["rest"]["skip_codes"] == [400, 422, 502]


def test_build_command_uses_rest_generator_and_report_prefix() -> None:
    command = build_command(
        api_url="http://api.test/",
        generator_options_path=Path("/tmp/garag-rest.json"),
        probes="encoding.InjectBase64",
        generations=1,
        seed=42,
        garak_report_prefix="garag",
    )

    assert command[:3] == [sys.executable, "-m", "garak"]
    assert "--target_type" in command
    assert "rest" in command
    assert "--target_name" in command
    assert "http://api.test/query" in command
    assert "-G" in command
    assert "/tmp/garag-rest.json" in command
    assert "--report_prefix" in command
    assert "garag" in command
