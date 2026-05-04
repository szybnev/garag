"""Workspace lifecycle management and hook execution."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from app.symphony.errors import WorkspaceError
from app.symphony.models import HooksConfig, Workspace, WorkspaceConfig

if TYPE_CHECKING:
    from pathlib import Path

SAFE_KEY_PATTERN = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_workspace_key(identifier: str) -> str:
    """Convert an issue identifier into a safe workspace directory name."""

    return SAFE_KEY_PATTERN.sub("_", identifier)


class WorkspaceManager:
    """Creates, validates, and removes per-issue workspaces."""

    def __init__(self, workspace_config: WorkspaceConfig, hooks: HooksConfig) -> None:
        self.root = workspace_config.root.resolve()
        self.hooks = hooks

    def create_for_issue(self, identifier: str) -> Workspace:
        """Create or reuse the workspace for an issue identifier."""

        key = sanitize_workspace_key(identifier)
        path = (self.root / key).resolve()
        self.validate_inside_root(path)
        created_now = False

        if path.exists() and not path.is_dir():
            raise WorkspaceError("workspace_path_not_directory", f"{path} exists and is not a dir")
        if not path.exists():
            path.mkdir(parents=True)
            created_now = True

        workspace = Workspace(path=path, workspace_key=key, created_now=created_now)
        if created_now and self.hooks.after_create:
            self.run_hook("after_create", workspace.path, fatal=True)
        return workspace

    def remove_for_identifier(self, identifier: str) -> None:
        """Run cleanup hook and remove an issue workspace if it exists."""

        path = (self.root / sanitize_workspace_key(identifier)).resolve()
        self.validate_inside_root(path)
        if not path.exists():
            return
        if self.hooks.before_remove:
            self.run_hook("before_remove", path, fatal=False)
        shutil.rmtree(path)

    def validate_inside_root(self, path: Path) -> None:
        """Enforce the workspace-root containment invariant."""

        root = self.root.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise WorkspaceError(
                "workspace_outside_root",
                f"Workspace path {resolved} is outside workspace root {root}",
            )

    def validate_agent_cwd(self, cwd: Path, workspace_path: Path) -> None:
        """Validate the coding-agent cwd before launch."""

        resolved_cwd = cwd.resolve()
        resolved_workspace = workspace_path.resolve()
        self.validate_inside_root(resolved_workspace)
        if resolved_cwd != resolved_workspace:
            raise WorkspaceError("invalid_workspace_cwd", "Agent cwd must equal workspace path")

    def run_hook(self, name: str, cwd: Path, *, fatal: bool) -> None:
        """Run a configured hook script in a workspace directory."""

        script = getattr(self.hooks, name)
        if not script:
            return
        try:
            subprocess.run(  # noqa: S603
                ["/usr/bin/env", "bash", "-lc", script],
                cwd=cwd,
                timeout=self.hooks.timeout_ms / 1000,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if fatal:
                raise WorkspaceError("workspace_hook_failed", f"{name} hook failed") from exc
