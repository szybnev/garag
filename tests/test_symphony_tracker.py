"""Linear tracker adapter tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.symphony.errors import TrackerError
from app.symphony.models import TrackerConfig
from app.symphony.tracker import ISSUE_STATES_BY_IDS_QUERY, LinearIssueTracker


def _config() -> TrackerConfig:
    return TrackerConfig(
        kind="linear",
        endpoint="https://linear.test/graphql",
        api_key="lin-test",
        project_slug="sec",
        active_states=("Todo", "In Progress"),
        terminal_states=("Done",),
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
