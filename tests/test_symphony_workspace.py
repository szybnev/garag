"""Workspace safety and hook lifecycle tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.symphony.errors import WorkspaceError
from app.symphony.models import HooksConfig, WorkspaceConfig
from app.symphony.workspace import WorkspaceManager, sanitize_workspace_key

if TYPE_CHECKING:
    from pathlib import Path


def _hooks(**overrides: str | int | None) -> HooksConfig:
    data = {
        "after_create": None,
        "before_run": None,
        "after_run": None,
        "before_remove": None,
        "timeout_ms": 60_000,
    }
    data.update(overrides)
    return HooksConfig(**data)


def test_sanitize_workspace_key() -> None:
    assert sanitize_workspace_key("SEC-1 / bad:key") == "SEC-1___bad_key"


def test_create_reuses_workspace_and_runs_after_create_once(tmp_path: Path) -> None:
    manager = WorkspaceManager(
        WorkspaceConfig(tmp_path),
        _hooks(after_create="printf created >> marker"),
    )

    first = manager.create_for_issue("SEC-1")
    second = manager.create_for_issue("SEC-1")

    assert first.created_now is True
    assert second.created_now is False
    assert first.path == second.path
    assert (first.path / "marker").read_text(encoding="utf-8") == "created"


def test_hook_failure_is_fatal_when_requested(tmp_path: Path) -> None:
    manager = WorkspaceManager(WorkspaceConfig(tmp_path), _hooks(after_create="exit 7"))

    with pytest.raises(WorkspaceError) as exc_info:
        manager.create_for_issue("SEC-1")

    assert exc_info.value.code == "workspace_hook_failed"


def test_before_remove_failure_is_ignored_and_workspace_removed(tmp_path: Path) -> None:
    manager = WorkspaceManager(WorkspaceConfig(tmp_path), _hooks(before_remove="exit 7"))
    workspace = manager.create_for_issue("SEC-1")

    manager.remove_for_identifier("SEC-1")

    assert not workspace.path.exists()


def test_agent_cwd_must_equal_workspace_path(tmp_path: Path) -> None:
    manager = WorkspaceManager(WorkspaceConfig(tmp_path), _hooks())
    workspace = manager.create_for_issue("SEC-1")
    other = tmp_path / "other"
    other.mkdir()

    with pytest.raises(WorkspaceError) as exc_info:
        manager.validate_agent_cwd(other, workspace.path)

    assert exc_info.value.code == "invalid_workspace_cwd"
