"""Run a focused garak scan against the GaRAG `/query` endpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_PROBES = "encoding.InjectBase64,promptinject.HijackLongPrompt,dan.Ablation_Dan_11_0"
DEFAULT_REPORT_PREFIX = "security/garak/reports/garag"


def build_generator_options(api_url: str, *, top_k: int, timeout_s: int) -> dict[str, Any]:
    """Build garak REST generator options for FastAPI `POST /query`."""
    return {
        "rest": {
            "uri": f"{api_url.rstrip('/')}/query",
            "method": "post",
            "headers": {"Content-Type": "application/json"},
            "req_template_json_object": {"query": "$INPUT", "top_k": top_k},
            "response_json": True,
            "response_json_field": "$.answer",
            "request_timeout": timeout_s,
            "skip_codes": [400, 422, 502],
        }
    }


def build_command(
    *,
    api_url: str,
    generator_options_path: Path,
    probes: str,
    generations: int,
    seed: int,
    garak_report_prefix: str,
) -> list[str]:
    """Build the garak CLI command."""
    return [
        sys.executable,
        "-m",
        "garak",
        "--target_type",
        "rest",
        "--target_name",
        f"{api_url.rstrip('/')}/query",
        "-G",
        str(generator_options_path),
        "--probes",
        probes,
        "--generations",
        str(generations),
        "--seed",
        str(seed),
        "--report_prefix",
        garak_report_prefix,
        "--skip_unknown",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--probes", default=DEFAULT_PROBES)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-prefix", default=DEFAULT_REPORT_PREFIX)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_prefix_path = Path(args.report_prefix)
    report_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    garak_report_prefix = report_prefix_path.name
    generator_options = build_generator_options(
        args.api_url,
        top_k=args.top_k,
        timeout_s=args.timeout_s,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as tmp:
        json.dump(generator_options, tmp)
        tmp.flush()
        command = build_command(
            generator_options_path=Path(tmp.name),
            api_url=args.api_url,
            probes=args.probes,
            generations=args.generations,
            seed=args.seed,
            garak_report_prefix=garak_report_prefix,
        )
        completed = subprocess.run(command, check=False)  # noqa: S603
    _copy_garak_reports(garak_report_prefix, report_prefix_path)
    return completed.returncode


def _copy_garak_reports(garak_report_prefix: str, report_prefix_path: Path) -> None:
    garak_report_dir = Path.home() / ".local/share/garak/garak_runs"
    for source in garak_report_dir.glob(f"{garak_report_prefix}.*"):
        target = report_prefix_path.parent / source.name
        shutil.copy2(source, target)


if __name__ == "__main__":
    raise SystemExit(main())
