"""Single-authority Symphony polling orchestrator."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.symphony.agent import AgentRunner
from app.symphony.config import load_service_config, validate_dispatch_config
from app.symphony.errors import ConfigError, SymphonyError, WorkflowError
from app.symphony.models import AgentEvent, CodexTotals, Issue, RetryEntry, RunningEntry, utc_now
from app.symphony.workflow import WorkflowDefinition, load_workflow
from app.symphony.workspace import WorkspaceManager

if TYPE_CHECKING:
    from app.symphony.tracker import IssueTracker

LOGGER = logging.getLogger(__name__)
CONTINUATION_RETRY_MS = 1_000
BASE_RETRY_MS = 10_000

RunnerFactory = Callable[[WorkflowDefinition], AgentRunner]


class SymphonyOrchestrator:
    """Owns polling, claims, retries, reconciliation, and runtime metrics."""

    def __init__(
        self,
        workflow: WorkflowDefinition,
        tracker: IssueTracker,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self.workflow = workflow
        self.config = load_service_config(workflow)
        self.tracker = tracker
        self.runner_factory = runner_factory or self._default_runner_factory
        self.running: dict[str, RunningEntry] = {}
        self.claimed: set[str] = set()
        self.retry_attempts: dict[str, RetryEntry] = {}
        self.completed: set[str] = set()
        self.codex_totals = CodexTotals()
        self.codex_rate_limits: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._immediate_tick = asyncio.Event()

    async def start(self) -> None:
        """Validate config, run startup cleanup, and start the polling loop."""

        validate_dispatch_config(self.config)
        await self._startup_terminal_workspace_cleanup()
        while not self._stop.is_set():
            await self.tick()
            timeout = self.config.polling.interval_ms / 1000
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._immediate_tick.wait(), timeout=timeout)
            self._immediate_tick.clear()

    async def stop(self) -> None:
        """Stop polling and cancel active worker tasks."""

        self._stop.set()
        for entry in list(self.running.values()):
            entry.stop_requested = True
            entry.worker.cancel()
        for retry in self.retry_attempts.values():
            if retry.timer_handle is not None:
                retry.timer_handle.cancel()

    async def refresh_now(self) -> None:
        """Request an immediate best-effort poll/reconcile cycle."""

        self._immediate_tick.set()

    async def tick(self) -> None:
        """Run one reconcile + dispatch cycle."""

        await self._reload_if_changed()
        await self._reconcile_running_issues()
        try:
            validate_dispatch_config(self.config)
        except ConfigError as exc:
            LOGGER.warning(
                "dispatch_validation failed code=%s reason=%s",
                exc.code,
                exc.message,
            )
            return

        try:
            issues = await asyncio.to_thread(self.tracker.fetch_candidate_issues)
        except SymphonyError as exc:
            LOGGER.warning("candidate_fetch failed code=%s reason=%s", exc.code, exc.message)
            return

        async with self._lock:
            for issue in sort_for_dispatch(issues):
                if self._available_slots() <= 0:
                    break
                if self._should_dispatch(issue):
                    self._dispatch_issue(issue, attempt=None)

    def snapshot(self) -> dict[str, Any]:
        """Return a synchronous runtime snapshot for observability surfaces."""

        now = utc_now()
        running = [self._running_snapshot(entry) for entry in self.running.values()]
        retrying = [self._retry_snapshot(entry) for entry in self.retry_attempts.values()]
        active_seconds = sum(
            (now - entry.started_at).total_seconds() for entry in self.running.values()
        )
        return {
            "generated_at": now.isoformat(),
            "counts": {"running": len(running), "retrying": len(retrying)},
            "running": running,
            "retrying": retrying,
            "codex_totals": {
                "input_tokens": self.codex_totals.input_tokens,
                "output_tokens": self.codex_totals.output_tokens,
                "total_tokens": self.codex_totals.total_tokens,
                "seconds_running": self.codex_totals.seconds_running + active_seconds,
            },
            "rate_limits": self.codex_rate_limits,
        }

    async def _reload_if_changed(self) -> None:
        try:
            mtime_ns = self.workflow.path.stat().st_mtime_ns
        except OSError as exc:
            LOGGER.warning("workflow_reload failed code=missing_workflow_file reason=%s", exc)
            return
        if mtime_ns == self.workflow.mtime_ns:
            return
        try:
            workflow = load_workflow(self.workflow.path)
            config = load_service_config(workflow)
        except (ConfigError, WorkflowError) as exc:
            LOGGER.warning("workflow_reload failed code=%s reason=%s", exc.code, exc.message)
            return
        self.workflow = workflow
        self.config = config
        LOGGER.info("workflow_reload completed path=%s", workflow.path)

    async def _reconcile_running_issues(self) -> None:
        await self._reconcile_stalled_runs()
        running_ids = list(self.running)
        if not running_ids:
            return
        try:
            refreshed = await asyncio.to_thread(self.tracker.fetch_issue_states_by_ids, running_ids)
        except SymphonyError as exc:
            LOGGER.warning("running_reconcile failed code=%s reason=%s", exc.code, exc.message)
            return
        by_id = {issue.id: issue for issue in refreshed}
        for issue_id, entry in list(self.running.items()):
            fresh = by_id.get(issue_id)
            if fresh is None:
                continue
            state = fresh.state.lower()
            if state in self.config.terminal_state_keys:
                await self._terminate_running_issue(issue_id, cleanup_workspace=True)
            elif state in self.config.active_state_keys:
                entry.issue = fresh
            else:
                await self._terminate_running_issue(issue_id, cleanup_workspace=False)

    async def _reconcile_stalled_runs(self) -> None:
        timeout_ms = self.config.codex.stall_timeout_ms
        if timeout_ms <= 0:
            return
        now = utc_now()
        for issue_id, entry in list(self.running.items()):
            since = entry.session.last_codex_timestamp or entry.started_at
            elapsed_ms = (now - since).total_seconds() * 1000
            if elapsed_ms > timeout_ms:
                await self._terminate_running_issue(issue_id, cleanup_workspace=False)

    async def _terminate_running_issue(self, issue_id: str, *, cleanup_workspace: bool) -> None:
        entry = self.running.get(issue_id)
        if entry is None:
            return
        entry.stop_requested = True
        entry.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await entry.worker
        if cleanup_workspace:
            manager = WorkspaceManager(self.config.workspace, self.config.hooks)
            await asyncio.to_thread(manager.remove_for_identifier, entry.issue.identifier)

    def _dispatch_issue(self, issue: Issue, attempt: int | None) -> None:
        runner = self.runner_factory(self.workflow)
        task = asyncio.create_task(self._run_worker(runner, issue, attempt))
        self.running[issue.id] = RunningEntry(
            issue=issue,
            worker=task,
            workspace_path=None,
            started_at=utc_now(),
            retry_attempt=attempt,
        )
        self.claimed.add(issue.id)
        retry = self.retry_attempts.pop(issue.id, None)
        if retry and retry.timer_handle is not None:
            retry.timer_handle.cancel()

    async def _run_worker(
        self,
        runner: AgentRunner,
        issue: Issue,
        attempt: int | None,
    ) -> None:
        reason = "normal"
        try:
            workspace = await runner.run_issue(
                issue,
                attempt,
                lambda event: self._on_agent_event(issue.id, event),
            )
            if issue.id in self.running:
                self.running[issue.id].workspace_path = workspace.path
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        except SymphonyError as exc:
            reason = f"{exc.code}: {exc.message}"
        except Exception as exc:
            reason = f"worker_error: {exc}"
        finally:
            await self._on_worker_exit(issue.id, reason)

    async def _on_worker_exit(self, issue_id: str, reason: str) -> None:
        async with self._lock:
            entry = self.running.pop(issue_id, None)
            if entry is None:
                return
            self._add_runtime_seconds(entry)
            if entry.stop_requested:
                self.claimed.discard(issue_id)
                return
            if reason == "normal":
                self.completed.add(issue_id)
                self._schedule_retry(entry.issue, 1, None, continuation=True)
            else:
                next_attempt = (entry.retry_attempt or 0) + 1
                self._schedule_retry(entry.issue, next_attempt, reason, continuation=False)

    async def _on_agent_event(self, issue_id: str, event: AgentEvent) -> None:
        entry = self.running.get(issue_id)
        if entry is None:
            return
        session = entry.session
        session.codex_app_server_pid = event.codex_app_server_pid
        session.thread_id = event.thread_id or session.thread_id
        session.turn_id = event.turn_id or session.turn_id
        if session.thread_id and session.turn_id:
            session.session_id = f"{session.thread_id}-{session.turn_id}"
        session.last_codex_event = event.event
        session.last_codex_timestamp = event.timestamp
        session.last_codex_message = event.message
        if event.event == "turn_started":
            session.turn_count += 1
        if event.usage:
            self._record_usage_delta(entry, event.usage)
        if event.rate_limits:
            self.codex_rate_limits = event.rate_limits

    def _schedule_retry(
        self,
        issue: Issue,
        attempt: int,
        error: str | None,
        *,
        continuation: bool,
    ) -> None:
        existing = self.retry_attempts.pop(issue.id, None)
        if existing and existing.timer_handle is not None:
            existing.timer_handle.cancel()
        delay_ms = (
            CONTINUATION_RETRY_MS
            if continuation
            else min(
                BASE_RETRY_MS * (2 ** max(attempt - 1, 0)),
                self.config.agent.max_retry_backoff_ms,
            )
        )
        loop = asyncio.get_running_loop()
        handle = loop.call_later(
            delay_ms / 1000,
            lambda: asyncio.create_task(self._handle_retry(issue.id)),
        )
        self.retry_attempts[issue.id] = RetryEntry(
            issue_id=issue.id,
            identifier=issue.identifier,
            attempt=attempt,
            due_at_ms=loop.time() * 1000 + delay_ms,
            error=error,
            timer_handle=handle,
        )

    async def _handle_retry(self, issue_id: str) -> None:
        async with self._lock:
            retry = self.retry_attempts.pop(issue_id, None)
        if retry is None:
            return
        try:
            candidates = await asyncio.to_thread(self.tracker.fetch_candidate_issues)
        except SymphonyError:
            async with self._lock:
                issue = Issue(id=issue_id, identifier=retry.identifier, title="", state="")
                self._schedule_retry(
                    issue,
                    retry.attempt + 1,
                    "retry poll failed",
                    continuation=False,
                )
            return
        issue = next((candidate for candidate in candidates if candidate.id == issue_id), None)
        async with self._lock:
            if issue is None:
                self.claimed.discard(issue_id)
                return
            if self._available_slots() <= 0:
                self._schedule_retry(
                    issue,
                    retry.attempt + 1,
                    "no available orchestrator slots",
                    continuation=False,
                )
                return
            if self._candidate_eligible(issue, ignore_claim=True):
                self._dispatch_issue(issue, attempt=retry.attempt)
            else:
                self.claimed.discard(issue_id)

    async def _startup_terminal_workspace_cleanup(self) -> None:
        try:
            issues = await asyncio.to_thread(
                self.tracker.fetch_issues_by_states,
                list(self.config.tracker.terminal_states),
            )
        except SymphonyError as exc:
            LOGGER.warning("startup_cleanup failed code=%s reason=%s", exc.code, exc.message)
            return
        manager = WorkspaceManager(self.config.workspace, self.config.hooks)
        for issue in issues:
            await asyncio.to_thread(manager.remove_for_identifier, issue.identifier)

    def _should_dispatch(self, issue: Issue) -> bool:
        eligible = self._candidate_eligible(issue, ignore_claim=False)
        return eligible and self._state_slot_available(issue)

    def _candidate_eligible(self, issue: Issue, *, ignore_claim: bool) -> bool:
        if not issue.id or not issue.identifier or not issue.title or not issue.state:
            return False
        state = issue.state.lower()
        if state not in self.config.active_state_keys or state in self.config.terminal_state_keys:
            return False
        if issue.id in self.running:
            return False
        if not ignore_claim and issue.id in self.claimed:
            return False
        return not _blocked_todo(issue, self.config.terminal_state_keys)

    def _state_slot_available(self, issue: Issue) -> bool:
        state = issue.state.lower()
        state_limit = self.config.agent.max_concurrent_agents_by_state.get(
            state,
            self.config.agent.max_concurrent_agents,
        )
        running_in_state = sum(
            1 for entry in self.running.values() if entry.issue.state.lower() == state
        )
        return running_in_state < state_limit

    def _available_slots(self) -> int:
        return max(self.config.agent.max_concurrent_agents - len(self.running), 0)

    def _default_runner_factory(self, workflow: WorkflowDefinition) -> AgentRunner:
        return AgentRunner(
            self.config,
            workflow,
            WorkspaceManager(self.config.workspace, self.config.hooks),
            self.tracker,
        )

    def _record_usage_delta(self, entry: RunningEntry, usage: dict[str, int]) -> None:
        session = entry.session
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        self.codex_totals.input_tokens += max(input_tokens - session.last_reported_input_tokens, 0)
        self.codex_totals.output_tokens += max(
            output_tokens - session.last_reported_output_tokens,
            0,
        )
        self.codex_totals.total_tokens += max(total_tokens - session.last_reported_total_tokens, 0)
        session.codex_input_tokens = input_tokens
        session.codex_output_tokens = output_tokens
        session.codex_total_tokens = total_tokens
        session.last_reported_input_tokens = input_tokens
        session.last_reported_output_tokens = output_tokens
        session.last_reported_total_tokens = total_tokens

    def _add_runtime_seconds(self, entry: RunningEntry) -> None:
        self.codex_totals.seconds_running += (utc_now() - entry.started_at).total_seconds()

    def _running_snapshot(self, entry: RunningEntry) -> dict[str, Any]:
        session = entry.session
        return {
            "issue_id": entry.issue.id,
            "issue_identifier": entry.issue.identifier,
            "state": entry.issue.state,
            "session_id": session.session_id,
            "turn_count": session.turn_count,
            "last_event": session.last_codex_event,
            "last_message": session.last_codex_message,
            "started_at": entry.started_at.isoformat(),
            "last_event_at": (
                session.last_codex_timestamp.isoformat() if session.last_codex_timestamp else None
            ),
            "tokens": {
                "input_tokens": session.codex_input_tokens,
                "output_tokens": session.codex_output_tokens,
                "total_tokens": session.codex_total_tokens,
            },
        }

    def _retry_snapshot(self, entry: RetryEntry) -> dict[str, Any]:
        due_at = datetime.fromtimestamp(entry.due_at_ms / 1000, tz=UTC)
        return {
            "issue_id": entry.issue_id,
            "issue_identifier": entry.identifier,
            "attempt": entry.attempt,
            "due_at": due_at.isoformat(),
            "error": entry.error,
        }


def sort_for_dispatch(issues: list[Issue]) -> list[Issue]:
    """Sort issues by priority, creation time, and identifier."""

    return sorted(
        issues,
        key=lambda issue: (
            issue.priority if issue.priority is not None else 999_999,
            issue.created_at or datetime.max.replace(tzinfo=UTC),
            issue.identifier,
        ),
    )


def _blocked_todo(issue: Issue, terminal_states: set[str]) -> bool:
    if issue.state.lower() != "todo":
        return False
    return any(
        blocker.state is not None and blocker.state.lower() not in terminal_states
        for blocker in issue.blocked_by
    )
