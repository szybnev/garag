"""Run the Symphony coding-agent orchestration service."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from app.symphony.config import load_service_config, validate_dispatch_config
from app.symphony.orchestrator import SymphonyOrchestrator
from app.symphony.tracker import LinearIssueTracker
from app.symphony.workflow import load_workflow


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
    asyncio.run(_main(args.workflow))


async def _main(workflow_path: str | None) -> None:
    workflow = load_workflow(Path(workflow_path) if workflow_path else None)
    config = load_service_config(workflow)
    validate_dispatch_config(config)
    tracker = LinearIssueTracker(config.tracker)
    orchestrator = SymphonyOrchestrator(workflow, tracker)
    await orchestrator.start()


if __name__ == "__main__":
    main()
