"""Issue tracker adapters used by Symphony orchestration."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from app.symphony.errors import TrackerError
from app.symphony.models import BlockerRef, Issue, TrackerConfig

PAGE_SIZE = 50
NETWORK_TIMEOUT_S = 30.0
BD_READY_COMMAND = ("bd", "ready", "--json")
BD_ISSUES_PATH = Path(".beads") / "issues.jsonl"
BdCommandRunner = Callable[[Sequence[str], Path], str]


class IssueTracker(Protocol):
    """Read-only issue tracker contract used by the scheduler and agent runner."""

    def fetch_candidate_issues(self) -> list[Issue]:
        """Return issues eligible for scheduler consideration."""
        ...

    def fetch_issues_by_states(self, state_names: list[str]) -> list[Issue]:
        """Return issues in the supplied tracker states."""
        ...

    def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> list[Issue]:
        """Return current issue state snapshots by tracker IDs."""
        ...


CANDIDATE_ISSUES_QUERY = """
query SymphonyCandidateIssues($projectSlug: String!, $states: [String!], $after: String) {
  issues(
    first: 50
    after: $after
    filter: {
      project: { slugId: { eq: $projectSlug } }
      state: { name: { in: $states } }
    }
  ) {
    nodes {
      id
      identifier
      title
      description
      priority
      branchName
      url
      createdAt
      updatedAt
      state { name }
      labels { nodes { name } }
      inverseRelations {
        nodes {
          type
          issue { id identifier state { name } }
        }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

ISSUES_BY_STATES_QUERY = """
query SymphonyIssuesByStates($projectSlug: String!, $states: [String!], $after: String) {
  issues(
    first: 50
    after: $after
    filter: {
      project: { slugId: { eq: $projectSlug } }
      state: { name: { in: $states } }
    }
  ) {
    nodes { id identifier title state { name } }
    pageInfo { hasNextPage endCursor }
  }
}
"""

ISSUE_STATES_BY_IDS_QUERY = """
query SymphonyIssueStates($ids: [ID!]) {
  issues(filter: { id: { in: $ids } }) {
    nodes {
      id
      identifier
      title
      description
      priority
      branchName
      url
      createdAt
      updatedAt
      state { name }
      labels { nodes { name } }
      inverseRelations {
        nodes {
          type
          issue { id identifier state { name } }
        }
      }
    }
  }
}
"""


class LinearIssueTracker:
    """Read-only Linear GraphQL adapter used by Symphony orchestration."""

    def __init__(self, config: TrackerConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client or httpx.Client(timeout=NETWORK_TIMEOUT_S)

    def fetch_candidate_issues(self) -> list[Issue]:
        """Return active project issues eligible for scheduler consideration."""

        return self._fetch_paginated(
            CANDIDATE_ISSUES_QUERY,
            {
                "projectSlug": self._project_slug(),
                "states": list(self.config.active_states),
            },
        )

    def fetch_issues_by_states(self, state_names: list[str]) -> list[Issue]:
        """Return project issues in the supplied states."""

        if not state_names:
            return []
        return self._fetch_paginated(
            ISSUES_BY_STATES_QUERY,
            {"projectSlug": self._project_slug(), "states": state_names},
        )

    def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> list[Issue]:
        """Return current normalized issue state snapshots by Linear IDs."""

        if not issue_ids:
            return []
        payload = self._graphql(ISSUE_STATES_BY_IDS_QUERY, {"ids": issue_ids})
        nodes = _nodes(payload, "issues")
        return [_normalize_issue(node) for node in nodes]

    def execute_graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one raw GraphQL operation for the optional agent tool extension."""

        if not query.strip():
            raise TrackerError("linear_graphql_invalid_input", "query must be non-empty")
        return self._graphql(query, variables or {})

    def _fetch_paginated(self, query: str, variables: dict[str, Any]) -> list[Issue]:
        issues: list[Issue] = []
        after: str | None = None
        while True:
            page_vars = variables | {"after": after}
            payload = self._graphql(query, page_vars)
            container = _container(payload, "issues")
            issues.extend(_normalize_issue(node) for node in container.get("nodes", []))
            page_info = container.get("pageInfo")
            if not isinstance(page_info, dict):
                raise TrackerError("linear_unknown_payload", "Missing pageInfo")
            if not page_info.get("hasNextPage"):
                return issues
            end_cursor = page_info.get("endCursor")
            if not isinstance(end_cursor, str) or not end_cursor:
                raise TrackerError("linear_missing_end_cursor", "Missing endCursor")
            after = end_cursor

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if not self.config.api_key:
            raise TrackerError("missing_tracker_api_key", "Linear API key is missing")
        try:
            response = self._client.post(
                self.config.endpoint,
                headers={"Authorization": self.config.api_key},
                json={"query": query, "variables": variables},
                timeout=NETWORK_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise TrackerError("linear_api_request", "Linear request failed") from exc

        if response.status_code != httpx.codes.OK:
            raise TrackerError("linear_api_status", f"Linear returned HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise TrackerError("linear_unknown_payload", "Linear returned non-object JSON")
        if payload.get("errors"):
            raise TrackerError("linear_graphql_errors", "Linear returned GraphQL errors")
        return payload

    def _project_slug(self) -> str:
        if not self.config.project_slug:
            raise TrackerError("missing_tracker_project_slug", "Linear project slug is missing")
        return self.config.project_slug


def _container(payload: dict[str, Any], key: str) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise TrackerError("linear_unknown_payload", "Missing data object")
    value = data.get(key)
    if not isinstance(value, dict):
        raise TrackerError("linear_unknown_payload", f"Missing {key} object")
    return value


def _nodes(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    nodes = _container(payload, key).get("nodes")
    if not isinstance(nodes, list):
        raise TrackerError("linear_unknown_payload", "Missing nodes list")
    return [node for node in nodes if isinstance(node, dict)]


def _normalize_issue(node: dict[str, Any]) -> Issue:
    state = node.get("state")
    labels = node.get("labels")
    return Issue(
        id=str(node.get("id") or ""),
        identifier=str(node.get("identifier") or ""),
        title=str(node.get("title") or ""),
        description=_optional_str(node.get("description")),
        priority=_optional_int(node.get("priority")),
        state=_state_name(state),
        branch_name=_optional_str(node.get("branchName")),
        url=_optional_str(node.get("url")),
        labels=_labels(labels),
        blocked_by=_blockers(node.get("inverseRelations")),
        created_at=_parse_dt(node.get("createdAt")),
        updated_at=_parse_dt(node.get("updatedAt")),
    )


def _state_name(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return value["name"]
    if isinstance(value, str):
        return value
    return ""


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _labels(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    nodes = value.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [
        item["name"].lower()
        for item in nodes
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def _blockers(value: Any) -> list[BlockerRef]:
    if not isinstance(value, dict):
        return []
    nodes = value.get("nodes")
    if not isinstance(nodes, list):
        return []

    blockers: list[BlockerRef] = []
    for relation in nodes:
        if not isinstance(relation, dict) or relation.get("type") != "blocks":
            continue
        issue = relation.get("issue")
        if not isinstance(issue, dict):
            continue
        blockers.append(
            BlockerRef(
                id=_optional_str(issue.get("id")),
                identifier=_optional_str(issue.get("identifier")),
                state=_state_name(issue.get("state")) or None,
            )
        )
    return blockers


class BdIssueTracker:
    """Read-only local bd adapter.

    Candidate dispatch uses `bd ready --json`; refresh and cleanup read the local JSONL database
    directly because some bd list/show commands are less stable in this repository clone.
    """

    def __init__(
        self,
        config: TrackerConfig,
        *,
        root: Path | None = None,
        command_runner: BdCommandRunner | None = None,
    ) -> None:
        self.config = config
        self.root = (root or Path.cwd()).resolve()
        self._command_runner = command_runner or _run_bd_command

    def fetch_candidate_issues(self) -> list[Issue]:
        """Return issues from `bd ready --json`."""

        payload = self._command_runner(BD_READY_COMMAND, self.root)
        raw_issues = _json_list(payload, "bd_ready_payload")
        return [_normalize_bd_issue(issue) for issue in raw_issues]

    def fetch_issues_by_states(self, state_names: list[str]) -> list[Issue]:
        """Return local bd issues whose status matches any supplied state."""

        if not state_names:
            return []
        state_keys = {state.lower() for state in state_names}
        return [issue for issue in self._read_local_issues() if issue.state.lower() in state_keys]

    def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> list[Issue]:
        """Return local bd issues matching the supplied IDs."""

        if not issue_ids:
            return []
        wanted = set(issue_ids)
        return [issue for issue in self._read_local_issues() if issue.id in wanted]

    def _read_local_issues(self) -> list[Issue]:
        path = self.root / BD_ISSUES_PATH
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise TrackerError("bd_issues_read_failed", f"Unable to read {path}") from exc

        issues: list[Issue] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrackerError(
                    "bd_issues_parse_failed",
                    f"Invalid JSON in {path}:{line_number}",
                ) from exc
            if isinstance(payload, dict):
                issues.append(_normalize_bd_issue(payload))
        return issues


def _run_bd_command(command: Sequence[str], cwd: Path) -> str:
    try:
        result = subprocess.run(  # noqa: S603
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise TrackerError("bd_command_failed", "Unable to execute bd") from exc

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "bd command failed").strip()
        raise TrackerError("bd_command_failed", message[:500])
    return result.stdout


def _json_list(raw: str, code: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrackerError(code, "bd returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise TrackerError(code, "bd returned non-list JSON")
    return [item for item in payload if isinstance(item, dict)]


def _normalize_bd_issue(node: dict[str, Any]) -> Issue:
    issue_id = str(node.get("id") or "")
    return Issue(
        id=issue_id,
        identifier=issue_id,
        title=str(node.get("title") or ""),
        description=_optional_str(node.get("description")),
        priority=_optional_int(node.get("priority")),
        state=str(node.get("status") or ""),
        labels=_bd_labels(node.get("labels")),
        created_at=_parse_dt(node.get("created_at")),
        updated_at=_parse_dt(node.get("updated_at")),
    )


def _bd_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.lower() for item in value if isinstance(item, str)]
