"""Domain models for the Symphony scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class BlockerRef:
    """A normalized issue blocker reference."""

    id: str | None = None
    identifier: str | None = None
    state: str | None = None


@dataclass(frozen=True, slots=True)
class Issue:
    """Normalized tracker issue used by scheduling, prompts, and logs."""

    id: str
    identifier: str
    title: str
    state: str
    description: str | None = None
    priority: int | None = None
    branch_name: str | None = None
    url: str | None = None
    labels: list[str] = field(default_factory=list)
    blocked_by: list[BlockerRef] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def template_context(self) -> dict[str, Any]:
        """Return a JSON-like issue object for strict template rendering."""

        return {
            "id": self.id,
            "identifier": self.identifier,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "state": self.state,
            "branch_name": self.branch_name,
            "url": self.url,
            "labels": self.labels,
            "blocked_by": [
                {
                    "id": blocker.id,
                    "identifier": blocker.identifier,
                    "state": blocker.state,
                }
                for blocker in self.blocked_by
            ],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """Parsed `WORKFLOW.md` payload."""

    path: Path
    config: dict[str, Any]
    prompt_template: str
    mtime_ns: int | None


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Typed tracker configuration."""

    kind: str
    endpoint: str
    api_key: str | None
    project_slug: str | None
    active_states: tuple[str, ...]
    terminal_states: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PollingConfig:
    """Polling cadence configuration."""

    interval_ms: int


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """Workspace root configuration."""

    root: Path


@dataclass(frozen=True, slots=True)
class HooksConfig:
    """Workspace hook scripts and timeout."""

    after_create: str | None
    before_run: str | None
    after_run: str | None
    before_remove: str | None
    timeout_ms: int


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Scheduler-level agent configuration."""

    max_concurrent_agents: int
    max_turns: int
    max_retry_backoff_ms: int
    max_concurrent_agents_by_state: dict[str, int]


@dataclass(frozen=True, slots=True)
class CodexConfig:
    """Codex app-server launch and protocol settings."""

    command: str
    approval_policy: str | None
    thread_sandbox: str | None
    turn_sandbox_policy: dict[str, Any] | str | None
    turn_timeout_ms: int
    read_timeout_ms: int
    stall_timeout_ms: int


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """Typed runtime view over a workflow definition."""

    workflow_path: Path
    tracker: TrackerConfig
    polling: PollingConfig
    workspace: WorkspaceConfig
    hooks: HooksConfig
    agent: AgentConfig
    codex: CodexConfig

    @property
    def active_state_keys(self) -> set[str]:
        """Lowercase active tracker state names."""

        return {state.lower() for state in self.tracker.active_states}

    @property
    def terminal_state_keys(self) -> set[str]:
        """Lowercase terminal tracker state names."""

        return {state.lower() for state in self.tracker.terminal_states}


@dataclass(frozen=True, slots=True)
class Workspace:
    """Filesystem workspace assigned to one issue."""

    path: Path
    workspace_key: str
    created_now: bool


@dataclass(slots=True)
class LiveSession:
    """Live Codex session metadata tracked while a worker runs."""

    session_id: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    codex_app_server_pid: int | None = None
    last_codex_event: str | None = None
    last_codex_timestamp: datetime | None = None
    last_codex_message: str | None = None
    codex_input_tokens: int = 0
    codex_output_tokens: int = 0
    codex_total_tokens: int = 0
    last_reported_input_tokens: int = 0
    last_reported_output_tokens: int = 0
    last_reported_total_tokens: int = 0
    turn_count: int = 0


@dataclass(slots=True)
class RunningEntry:
    """Issue currently owned by a worker task."""

    issue: Issue
    worker: Any
    workspace_path: Path | None
    started_at: datetime
    retry_attempt: int | None = None
    session: LiveSession = field(default_factory=LiveSession)
    stop_requested: bool = False


@dataclass(slots=True)
class RetryEntry:
    """Scheduled retry state for an issue."""

    issue_id: str
    identifier: str
    attempt: int
    due_at_ms: float
    error: str | None = None
    timer_handle: Any | None = None


@dataclass(slots=True)
class CodexTotals:
    """Aggregate Codex token and runtime counters."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds_running: float = 0.0


@dataclass(slots=True)
class AgentEvent:
    """Structured event emitted by the Codex client."""

    event: str
    timestamp: datetime
    codex_app_server_pid: int | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    message: str | None = None
    usage: dict[str, int] | None = None
    rate_limits: dict[str, Any] | None = None
