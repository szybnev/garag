"""Workflow loading, config resolution, and strict prompt rendering tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.symphony.config import load_service_config, validate_dispatch_config
from app.symphony.errors import ConfigError, WorkflowError
from app.symphony.models import Issue
from app.symphony.workflow import load_workflow, render_prompt

if TYPE_CHECKING:
    from pathlib import Path


def _write_workflow(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "WORKFLOW.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_workflow_front_matter_and_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINEAR_TOKEN_TEST", "lin-test")
    monkeypatch.setenv("SYMPHONY_WS", str(tmp_path / "ws"))
    path = _write_workflow(
        tmp_path,
        """---
tracker:
  kind: linear
  api_key: $LINEAR_TOKEN_TEST
  project_slug: sec
polling:
  interval_ms: 1000
workspace:
  root: $SYMPHONY_WS
agent:
  max_concurrent_agents_by_state:
    Todo: 1
    broken: 0
codex:
  command: codex app-server
---
Handle {{ issue.identifier }} attempt={{ attempt }}.
""",
    )

    workflow = load_workflow(path)
    config = load_service_config(workflow)
    validate_dispatch_config(config)

    assert workflow.config["tracker"]["kind"] == "linear"
    assert config.tracker.api_key == "lin-test"
    assert config.workspace.root == tmp_path / "ws"
    assert config.agent.max_concurrent_agents_by_state == {"todo": 1}
    rendered = render_prompt(
        workflow,
        Issue(id="1", identifier="SEC-1", title="Fix", state="Todo"),
        attempt=2,
    )
    assert rendered == "Handle SEC-1 attempt=2."


def test_default_workflow_path_uses_cwd(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "Prompt")
    workflow = load_workflow(cwd=tmp_path)
    assert workflow.path == tmp_path / "WORKFLOW.md"
    assert workflow.prompt_template == "Prompt"


def test_missing_workflow_is_typed_error(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError) as exc_info:
        load_workflow(tmp_path / "missing.md")
    assert exc_info.value.code == "missing_workflow_file"


def test_non_map_front_matter_is_typed_error(tmp_path: Path) -> None:
    path = _write_workflow(tmp_path, "---\n- nope\n---\nPrompt")
    with pytest.raises(WorkflowError) as exc_info:
        load_workflow(path)
    assert exc_info.value.code == "workflow_front_matter_not_a_map"


@pytest.mark.parametrize(
    "template",
    [
        "Unknown {{ missing }}",
        "Unknown filter {{ issue.identifier | no_such_filter }}",
    ],
)
def test_prompt_rendering_is_strict(tmp_path: Path, template: str) -> None:
    workflow = load_workflow(_write_workflow(tmp_path, template))
    with pytest.raises(WorkflowError) as exc_info:
        render_prompt(workflow, Issue(id="1", identifier="SEC-1", title="Fix", state="Todo"))
    assert exc_info.value.code in {"template_parse_error", "template_render_error"}


def test_dispatch_validation_requires_linear_fields(tmp_path: Path) -> None:
    workflow = load_workflow(
        _write_workflow(
            tmp_path,
            """---
tracker:
  kind: linear
codex:
  command: codex app-server
---
Prompt
""",
        )
    )
    config = load_service_config(workflow)
    with pytest.raises(ConfigError) as exc_info:
        validate_dispatch_config(config)
    assert exc_info.value.code == "missing_tracker_api_key"
