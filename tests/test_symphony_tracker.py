"""Tracker adapter tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.symphony.errors import TrackerError
from app.symphony.models import TrackerConfig
from app.symphony.tracker import ISSUE_STATES_BY_IDS_QUERY, BdIssueTracker, LinearIssueTracker


def _config() -> TrackerConfig:
    return TrackerConfig(
        kind="linear",
        endpoint="https://linear.test/graphql",
        api_key="lin-test",
        project_slug="sec",
        active_states=("Todo", "In Progress"),
        terminal_states=("Done",),
    )


def _bd_config() -> TrackerConfig:
    return TrackerConfig(
        kind="bd",
        endpoint="",
        api_key=None,
        project_slug=None,
        active_states=("open", "in_progress"),
        terminal_states=("closed",),
    )


def _issue(identifier: str, state: str = "Todo") -> dict[str, Any]:
    return {
        "id": f"id-{identifier}",
        "identifier": identifier,
        "title": f"Title {identifier}",
        "description": "desc",
        "priority": 2,
        "branchName": "branch",
        "url": "https://linear.test/SEC-1",
        "createdAt": "2026-05-01T10:00:00+00:00",
        "updatedAt": "2026-05-02T10:00:00+00:00",
        "state": {"name": state},
        "labels": {"nodes": [{"name": "Security"}]},
        "inverseRelations": {
            "nodes": [
                {
                    "type": "blocks",
                    "issue": {
                        "id": "blocker-1",
                        "identifier": "SEC-0",
                        "state": {"name": "Todo"},
                    },
                },
                {"type": "relates", "issue": {"identifier": "SEC-x"}},
            ]
        },
    }


def _bd_issue(identifier: str, status: str = "open") -> dict[str, Any]:
    return {
        "id": identifier,
        "title": f"Title {identifier}",
        "description": "desc",
        "status": status,
        "priority": 2,
        "labels": ["Security"],
        "created_at": "2026-05-04T07:48:23.900524611Z",
        "updated_at": "2026-05-04T07:49:23Z",
    }


def test_bd_fetch_candidate_issues_parses_ready_json(tmp_path) -> None:
    calls: list[tuple[list[str], str]] = []

    def runner(command, cwd) -> str:
        calls.append((list(command), str(cwd)))
        return json.dumps([_bd_issue("garag-123")])

    tracker = BdIssueTracker(_bd_config(), root=tmp_path, command_runner=runner)
    issues = tracker.fetch_candidate_issues()

    assert [issue.identifier for issue in issues] == ["garag-123"]
    assert issues[0].id == "garag-123"
    assert issues[0].state == "open"
    assert issues[0].labels == ["security"]
    assert issues[0].created_at is not None
    assert calls == [(["bd", "ready", "--json"], str(tmp_path))]


def test_bd_refresh_and_terminal_cleanup_read_jsonl(tmp_path) -> None:
    beads = tmp_path / ".beads"
    beads.mkdir()
    (beads / "issues.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_bd_issue("garag-open", "open")),
                json.dumps(_bd_issue("garag-progress", "in_progress")),
                json.dumps(_bd_issue("garag-closed", "closed")),
            ]
        ),
        encoding="utf-8",
    )
    tracker = BdIssueTracker(_bd_config(), root=tmp_path)

    assert [issue.id for issue in tracker.fetch_issues_by_states(["closed"])] == ["garag-closed"]
    refreshed = tracker.fetch_issue_states_by_ids(["garag-progress", "missing"])
    assert [(issue.id, issue.state) for issue in refreshed] == [("garag-progress", "in_progress")]


def test_fetch_candidate_issues_paginates_and_normalizes() -> None:
    payloads = [
        {
            "data": {
                "issues": {
                    "nodes": [_issue("SEC-1")],
                    "pageInfo": {"hasNextPage": True, "endCursor": "next"},
                }
            }
        },
        {
            "data": {
                "issues": {
                    "nodes": [_issue("SEC-2", "In Progress")],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        },
    ]
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payloads[len(requests) - 1])

    tracker = LinearIssueTracker(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    issues = tracker.fetch_candidate_issues()

    assert [issue.identifier for issue in issues] == ["SEC-1", "SEC-2"]
    assert issues[0].labels == ["security"]
    assert issues[0].blocked_by[0].identifier == "SEC-0"
    assert "slugId" in requests[0]["query"]
    assert requests[1]["variables"]["after"] == "next"


def test_fetch_issues_by_states_empty_skips_api_call() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    tracker = LinearIssueTracker(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    assert tracker.fetch_issues_by_states([]) == []
    assert called is False


def test_fetch_issue_states_query_uses_graphql_id_typing() -> None:
    assert "[ID!]" in ISSUE_STATES_BY_IDS_QUERY

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"issues": {"nodes": [_issue("SEC-1")]}}})

    tracker = LinearIssueTracker(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    assert tracker.fetch_issue_states_by_ids(["id-SEC-1"])[0].identifier == "SEC-1"


def test_graphql_errors_are_mapped() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errors": [{"message": "bad"}]})

    tracker = LinearIssueTracker(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(TrackerError) as exc_info:
        tracker.fetch_candidate_issues()
    assert exc_info.value.code == "linear_graphql_errors"
