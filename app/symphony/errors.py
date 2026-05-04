"""Typed Symphony errors surfaced to operators and tests."""

from __future__ import annotations


class SymphonyError(Exception):
    """Base exception with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class WorkflowError(SymphonyError):
    """Workflow file loading or rendering failed."""


class ConfigError(SymphonyError):
    """Resolved workflow configuration is invalid."""


class TrackerError(SymphonyError):
    """Issue tracker operation failed."""


class WorkspaceError(SymphonyError):
    """Workspace lifecycle operation failed."""


class AgentRunnerError(SymphonyError):
    """Coding-agent subprocess operation failed."""
