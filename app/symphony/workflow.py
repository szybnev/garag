"""`WORKFLOW.md` loading and strict prompt rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError

from app.symphony.errors import WorkflowError
from app.symphony.models import Issue, WorkflowDefinition

DEFAULT_PROMPT = "You are working on a bd issue."


def select_workflow_path(path: str | Path | None = None, cwd: Path | None = None) -> Path:
    """Resolve an explicit workflow path or the cwd-local default."""

    base = cwd or Path.cwd()
    selected = Path(path) if path is not None else base / "WORKFLOW.md"
    return selected.expanduser().resolve()


def load_workflow(path: str | Path | None = None, cwd: Path | None = None) -> WorkflowDefinition:
    """Load YAML front matter and markdown body from a workflow file."""

    workflow_path = select_workflow_path(path, cwd)
    try:
        raw = workflow_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkflowError(
            "missing_workflow_file",
            f"Unable to read workflow file: {workflow_path}",
        ) from exc

    try:
        config, body = _split_front_matter(raw)
    except yaml.YAMLError as exc:
        raise WorkflowError("workflow_parse_error", str(exc)) from exc

    stat = workflow_path.stat()
    return WorkflowDefinition(
        path=workflow_path,
        config=config,
        prompt_template=body.strip(),
        mtime_ns=stat.st_mtime_ns,
    )


def render_prompt(
    workflow: WorkflowDefinition,
    issue: Issue,
    attempt: int | None = None,
) -> str:
    """Render a strict issue prompt from the workflow template."""

    source = workflow.prompt_template or DEFAULT_PROMPT
    env = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701
    try:
        template = env.from_string(source)
    except TemplateError as exc:
        raise WorkflowError("template_parse_error", str(exc)) from exc

    try:
        return template.render(issue=issue.template_context(), attempt=attempt)
    except TemplateError as exc:
        raise WorkflowError("template_render_error", str(exc)) from exc


def _split_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    if not raw.startswith("---"):
        return {}, raw

    lines = raw.splitlines()
    end_index = _front_matter_end_index(lines)
    if end_index is None:
        raise WorkflowError("workflow_parse_error", "YAML front matter is not closed")

    front_matter = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :])
    loaded = yaml.safe_load(front_matter) if front_matter.strip() else {}
    if loaded is None:
        return {}, body
    if not isinstance(loaded, dict):
        raise WorkflowError(
            "workflow_front_matter_not_a_map",
            "Workflow front matter must decode to a map/object",
        )
    return loaded, body


def _front_matter_end_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return index
    return None
