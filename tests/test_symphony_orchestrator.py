"""Scheduler dispatch, retry, reconciliation, and snapshot tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from app.symphony.models import (
    AgentEvent,
    BlockerRef,
    Issue,
    WorkflowDefinition,
    Workspace,
    utc_now,
)
from app.symphony.orchestrator import SymphonyOrchestrator, sort_for_dispatch

if TYPE_CHECKING:
    from pathlib import Path


class FakeTracker:
    def __init__(self, candidates: list[Issue]) -> None:
        self.candidates = candidates
        self.states: dict[str, Issue] = {issue.id: issue for issue in candidates}
        self.terminal: list[Issue] = []

    def fetch_candidate_issues(self) -> list[Issue]:
        return self.candidates

    def fetch_issues_by_states(self, _state_names: list[str]) -> list[Issue]:
        return self.terminal

    def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> list[Issue]:
        return [self.states[issue_id] for issue_id in issue_ids if issue_id in self.states]


class FakeRunner:
    def __init__(self, tmp_path: Path, event: AgentEvent | None = None) -> None:
        self.tmp_path = tmp_path
        self.event = event
        self.started: list[str] = []

    async def run_issue(self, issue: Issue, _attempt: int | None, on_event: Any) -> Workspace:
        self.started.append(issue.identifier)
        workspace_path = self.tmp_path / issue.identifier
        workspace_path.mkdir(exist_ok=True)
        if self.event:
            await on_event(self.event)
        return Workspace(workspace_path, issue.identifier, created_now=True)


def _workflow(tmp_path: Path) -> WorkflowDefinition:
    path = tmp_path / "WORKFLOW.md"
    path.write_text(
        """---
tracker:
  kind: bd
  active_states:
    - Todo
    - In Progress
  terminal_states:
    - Done
    - Canceled
polling:
  interval_ms: 1000
workspace:
  root: ws
agent:
  max_concurrent_agents: 1
  max_retry_backoff_ms: 20000
codex:
  command: codex app-server
---
Work on {{ issue.identifier }}.
""",
        encoding="utf-8",
    )
    return WorkflowDefinition(
        path=path,
        config={
            "tracker": {
                "kind": "bd",
                "active_states": ["Todo", "In Progress"],
                "terminal_states": ["Done", "Canceled"],
            },
            "polling": {"interval_ms": 1000},
            "workspace": {"root": "ws"},
            "agent": {"max_concurrent_agents": 1, "max_retry_backoff_ms": 20_000},
            "codex": {"command": "codex app-server"},
        },
        prompt_template="Work on {{ issue.identifier }}.",
        mtime_ns=path.stat().st_mtime_ns,
    )


def _issue(
    identifier: str,
    *,
    priority: int | None = 2,
    state: str = "Todo",
    blockers: list[BlockerRef] | None = None,
) -> Issue:
    return Issue(
        id=f"id-{identifier}",
        identifier=identifier,
        title=f"Title {identifier}",
        state=state,
        priority=priority,
        blocked_by=blockers or [],
    )


def test_sort_for_dispatch_priority_then_identifier() -> None:
    issues = [_issue("B", priority=None), _issue("A", priority=1), _issue("C", priority=1)]
    assert [issue.identifier for issue in sort_for_dispatch(issues)] == ["A", "C", "B"]


@pytest.mark.asyncio
async def test_tick_dispatches_one_issue_and_schedules_continuation_retry(tmp_path: Path) -> None:
    issue = _issue("SEC-1")
    tracker = FakeTracker([issue])
    runner = FakeRunner(tmp_path)
    orchestrator = SymphonyOrchestrator(_workflow(tmp_path), tracker, lambda _workflow: runner)

    await orchestrator.tick()
    await asyncio.sleep(0)

    assert runner.started == ["SEC-1"]
    assert issue.id in orchestrator.retry_attempts
    assert orchestrator.retry_attempts[issue.id].attempt == 1


@pytest.mark.asyncio
async def test_todo_with_open_blocker_is_not_dispatched(tmp_path: Path) -> None:
    issue = _issue("SEC-1", blockers=[BlockerRef(identifier="SEC-0", state="Todo")])
    runner = FakeRunner(tmp_path)
    orchestrator = SymphonyOrchestrator(_workflow(tmp_path), FakeTracker([issue]), lambda _: runner)

    await orchestrator.tick()
    await asyncio.sleep(0)

    assert runner.started == []


@pytest.mark.asyncio
async def test_terminal_reconciliation_cancels_and_releases_claim(tmp_path: Path) -> None:
    issue = _issue("SEC-1", state="In Progress")
    tracker = FakeTracker([issue])
    sleep_started = asyncio.Event()

    class SlowRunner(FakeRunner):
        async def run_issue(self, issue: Issue, _attempt: int | None, on_event: Any) -> Workspace:
            sleep_started.set()
            await asyncio.sleep(60)
            return await super().run_issue(issue, _attempt, on_event)

    runner = SlowRunner(tmp_path)
    orchestrator = SymphonyOrchestrator(_workflow(tmp_path), tracker, lambda _: runner)
    await orchestrator.tick()
    await sleep_started.wait()

    tracker.states[issue.id] = replace(issue, state="Done")
    tracker.candidates = []
    await orchestrator.tick()
    await asyncio.sleep(0)

    assert issue.id not in orchestrator.running
    assert issue.id not in orchestrator.claimed


@pytest.mark.asyncio
async def test_snapshot_tracks_token_usage(tmp_path: Path) -> None:
    issue = _issue("SEC-1")
    event = AgentEvent(
        event="thread/tokenUsage/updated",
        timestamp=utc_now(),
        thread_id="thr",
        turn_id="turn",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    runner = FakeRunner(tmp_path, event)
    orchestrator = SymphonyOrchestrator(_workflow(tmp_path), FakeTracker([issue]), lambda _: runner)

    await orchestrator.tick()
    await asyncio.sleep(0)
    snapshot = orchestrator.snapshot()

    assert snapshot["codex_totals"]["input_tokens"] == 10
    assert snapshot["codex_totals"]["total_tokens"] == 15
