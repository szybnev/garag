"""Typed Symphony configuration resolution."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from app.symphony.errors import ConfigError
from app.symphony.models import (
    AgentConfig,
    CodexConfig,
    HooksConfig,
    PollingConfig,
    ServiceConfig,
    TrackerConfig,
    WorkflowDefinition,
    WorkspaceConfig,
)

DEFAULT_ACTIVE_STATES = ("Todo", "In Progress")
DEFAULT_TERMINAL_STATES = ("Closed", "Cancelled", "Canceled", "Duplicate", "Done")
DEFAULT_LINEAR_ENDPOINT = "https://api.linear.app/graphql"


def load_service_config(workflow: WorkflowDefinition) -> ServiceConfig:
    """Resolve defaults, env indirection, paths, and typed config values."""

    raw = workflow.config
    tracker_raw = _mapping(raw.get("tracker"))
    polling_raw = _mapping(raw.get("polling"))
    workspace_raw = _mapping(raw.get("workspace"))
    hooks_raw = _mapping(raw.get("hooks"))
    agent_raw = _mapping(raw.get("agent"))
    codex_raw = _mapping(raw.get("codex"))

    kind = _optional_string(tracker_raw.get("kind"))
    endpoint = _optional_string(tracker_raw.get("endpoint")) or DEFAULT_LINEAR_ENDPOINT
    api_key = _resolve_secret(_optional_string(tracker_raw.get("api_key")))
    if kind == "linear" and api_key is None:
        api_key = _empty_to_none(os.environ.get("LINEAR_API_KEY"))

    return ServiceConfig(
        workflow_path=workflow.path,
        tracker=TrackerConfig(
            kind=kind or "",
            endpoint=endpoint,
            api_key=api_key,
            project_slug=_optional_string(tracker_raw.get("project_slug")),
            active_states=_string_tuple(
                tracker_raw.get("active_states"),
                default=DEFAULT_ACTIVE_STATES,
            ),
            terminal_states=_string_tuple(
                tracker_raw.get("terminal_states"),
                default=DEFAULT_TERMINAL_STATES,
            ),
        ),
        polling=PollingConfig(interval_ms=_positive_int(polling_raw.get("interval_ms"), 30_000)),
        workspace=WorkspaceConfig(root=_workspace_root(workflow, workspace_raw.get("root"))),
        hooks=HooksConfig(
            after_create=_optional_string(hooks_raw.get("after_create")),
            before_run=_optional_string(hooks_raw.get("before_run")),
            after_run=_optional_string(hooks_raw.get("after_run")),
            before_remove=_optional_string(hooks_raw.get("before_remove")),
            timeout_ms=_positive_int(hooks_raw.get("timeout_ms"), 60_000),
        ),
        agent=AgentConfig(
            max_concurrent_agents=_positive_int(agent_raw.get("max_concurrent_agents"), 10),
            max_turns=_positive_int(agent_raw.get("max_turns"), 20),
            max_retry_backoff_ms=_positive_int(agent_raw.get("max_retry_backoff_ms"), 300_000),
            max_concurrent_agents_by_state=_state_limits(
                agent_raw.get("max_concurrent_agents_by_state")
            ),
        ),
        codex=CodexConfig(
            command=_optional_string(codex_raw.get("command")) or "codex app-server",
            approval_policy=_optional_string(codex_raw.get("approval_policy")),
            thread_sandbox=_optional_string(codex_raw.get("thread_sandbox")),
            turn_sandbox_policy=_turn_sandbox_policy(codex_raw.get("turn_sandbox_policy")),
            turn_timeout_ms=_positive_int(codex_raw.get("turn_timeout_ms"), 3_600_000),
            read_timeout_ms=_positive_int(codex_raw.get("read_timeout_ms"), 5_000),
            stall_timeout_ms=_int_value(codex_raw.get("stall_timeout_ms"), 300_000),
        ),
    )


def validate_dispatch_config(config: ServiceConfig) -> None:
    """Validate the fields required before dispatching new work."""

    if config.tracker.kind != "linear":
        raise ConfigError("unsupported_tracker_kind", "tracker.kind must be 'linear'")
    if not config.tracker.api_key:
        raise ConfigError("missing_tracker_api_key", "tracker.api_key is required")
    if not config.tracker.project_slug:
        raise ConfigError("missing_tracker_project_slug", "tracker.project_slug is required")
    if not config.codex.command.strip():
        raise ConfigError("missing_codex_command", "codex.command is required")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_secret(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("$") and len(value) > 1:
        return _empty_to_none(os.environ.get(value[1:]))
    return _empty_to_none(value)


def _string_tuple(value: Any, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list):
        return default
    strings = tuple(item for item in value if isinstance(item, str) and item.strip())
    return strings or default


def _positive_int(value: Any, default: int) -> int:
    number = _int_value(value, default)
    if number <= 0:
        raise ConfigError("invalid_config_value", "Expected a positive integer")
    return number


def _int_value(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigError("invalid_config_value", "Boolean is not a valid integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("invalid_config_value", f"Expected integer, got {value!r}") from exc


def _workspace_root(workflow: WorkflowDefinition, value: Any) -> Path:
    raw = _optional_string(value)
    if raw is None:
        raw = str(Path(tempfile.gettempdir()) / "symphony_workspaces")
    if raw.startswith("$") and len(raw) > 1:
        raw = os.environ.get(raw[1:], "")
        if not raw:
            raise ConfigError("invalid_workspace_root", "workspace.root env var is empty")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = workflow.path.parent / path
    return path.resolve()


def _state_limits(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}

    limits: dict[str, int] = {}
    for state, raw_limit in value.items():
        if not isinstance(state, str):
            continue
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            continue
        if limit > 0:
            limits[state.lower()] = limit
    return limits


def _turn_sandbox_policy(value: Any) -> dict[str, Any] | str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value
    return None
