"""Run the Symphony coding-agent orchestration service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.symphony.config import load_service_config, validate_dispatch_config
from app.symphony.errors import SymphonyError
from app.symphony.orchestrator import SymphonyOrchestrator
from app.symphony.tracker import BdIssueTracker, LinearIssueTracker
from app.symphony.workflow import load_workflow

if TYPE_CHECKING:
    from app.symphony.models import TrackerConfig
    from app.symphony.tracker import IssueTracker


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run Symphony")
    parser.add_argument("workflow", nargs="?", help="Path to WORKFLOW.md")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s level=%(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_main(args.workflow))
    except SymphonyError as exc:
        print(f"Symphony startup failed [{exc.code}]: {exc.message}", file=sys.stderr)
        if exc.code == "missing_workflow_file":
            print(
                "Use ./WORKFLOW.md or pass a workflow path.",
                file=sys.stderr,
            )
        raise SystemExit(2) from exc


async def _main(workflow_path: str | None) -> None:
    workflow = load_workflow(Path(workflow_path) if workflow_path else None)
    config = load_service_config(workflow)
    validate_dispatch_config(config)
    tracker = _build_tracker(config.tracker, workflow.path.parent)
    orchestrator = SymphonyOrchestrator(workflow, tracker)
    await orchestrator.start()


def _build_tracker(config: TrackerConfig, root: Path) -> IssueTracker:
    if config.kind == "bd":
        return BdIssueTracker(config, root=root)
    return LinearIssueTracker(config)


if __name__ == "__main__":
    main()
