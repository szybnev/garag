"""Codex app-server client and Symphony agent runner."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from app.symphony.errors import AgentRunnerError
from app.symphony.models import AgentEvent, Issue, ServiceConfig, Workspace, utc_now
from app.symphony.workflow import WorkflowDefinition, render_prompt

if TYPE_CHECKING:
    from pathlib import Path

    from app.symphony.tracker import IssueTracker
    from app.symphony.workspace import WorkspaceManager

EventCallback = Callable[[AgentEvent], Awaitable[None] | None]

CONTINUATION_PROMPT = (
    "Continue working on the same issue. Do not repeat the original analysis unless needed. "
    "Check current tracker state and repository state, then proceed with the next useful step."
)


class CodexAppServerClient:
    """Minimal stdio JSON-RPC client for `codex app-server`."""

    def __init__(self, config: ServiceConfig, workspace: Path, on_event: EventCallback) -> None:
        self.config = config
        self.workspace = workspace
        self.on_event = on_event
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._thread_id: str | None = None

    @property
    def pid(self) -> int | None:
        """Return app-server process id when running."""

        return self.process.pid if self.process else None

    async def start(self) -> str:
        """Launch app-server, initialize JSON-RPC, and start a thread."""

        self.process = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            self.config.codex.command,
            cwd=self.workspace,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._emit("process_started")
        await self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "symphony",
                    "title": "Symphony",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await self._notify("initialized", {})
        result = await self._request("thread/start", self._thread_start_params())
        thread = _object_field(result, "thread")
        thread_id = _string_field(thread, "id")
        self._thread_id = thread_id
        await self._emit("thread_started", thread_id=thread_id)
        return thread_id

    async def run_turn(self, prompt: str) -> str:
        """Start one turn and stream notifications until it completes."""

        if self._thread_id is None:
            raise AgentRunnerError("response_error", "Codex thread has not been started")
        result = await self._request("turn/start", self._turn_start_params(prompt))
        turn = _object_field(result, "turn")
        turn_id = _string_field(turn, "id")
        await self._emit("turn_started", thread_id=self._thread_id, turn_id=turn_id)
        return await self._stream_until_turn_done(turn_id)

    async def close(self) -> None:
        """Terminate the app-server subprocess."""

        process = self.process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5)
        if process.returncode is None:
            process.kill()
            await process.wait()

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        await self._write({"method": method, "id": request_id, "params": params or {}})
        return await self._read_response(request_id)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"method": method, "params": params})

    async def _write(self, message: dict[str, Any]) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise AgentRunnerError("port_exit", "Codex stdin is unavailable")
        process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        await process.stdin.drain()

    async def _read_response(self, request_id: int) -> dict[str, Any]:
        deadline = self.config.codex.read_timeout_ms / 1000
        while True:
            message = await self._read_message(deadline)
            if message.get("id") == request_id:
                if "error" in message:
                    raise AgentRunnerError("response_error", _message_summary(message["error"]))
                result = message.get("result")
                return result if isinstance(result, dict) else {}
            await self._handle_incoming(message)

    async def _stream_until_turn_done(self, turn_id: str) -> str:
        deadline = self.config.codex.turn_timeout_ms / 1000
        while True:
            message = await self._read_message(deadline)
            method = message.get("method")
            params = message.get("params")
            await self._handle_incoming(message)
            if method == "turn/completed" and isinstance(params, dict):
                turn = params.get("turn")
                if isinstance(turn, dict) and turn.get("id") == turn_id:
                    status = str(turn.get("status") or "")
                    if status == "completed":
                        return status
                    if status == "interrupted":
                        raise AgentRunnerError("turn_cancelled", "Codex turn was interrupted")
                    raise AgentRunnerError("turn_failed", _message_summary(turn.get("error")))

    async def _read_message(self, timeout_s: float) -> dict[str, Any]:
        process = self._require_process()
        if process.stdout is None:
            raise AgentRunnerError("port_exit", "Codex stdout is unavailable")
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout_s)
        except TimeoutError as exc:
            raise AgentRunnerError("response_timeout", "Timed out waiting for Codex") from exc
        if not line:
            raise AgentRunnerError("port_exit", "Codex app-server exited")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            await self._emit("malformed", message=line[:500].decode("utf-8", errors="replace"))
            raise AgentRunnerError("response_error", "Malformed Codex JSONL message") from exc
        if not isinstance(message, dict):
            raise AgentRunnerError("response_error", "Codex message is not an object")
        return message

    async def _handle_incoming(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params")
        if isinstance(method, str):
            await self._emit(
                _event_name(method),
                thread_id=_extract_thread_id(params),
                turn_id=_extract_turn_id(params),
                message=_message_summary(params),
                usage=_extract_usage(method, params),
                rate_limits=_extract_rate_limits(params),
            )
        if "id" in message and isinstance(method, str):
            await self._resolve_server_request(message)

    async def _resolve_server_request(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        request_id = message["id"]
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            await self._write({"id": request_id, "result": {"decision": "acceptForSession"}})
        elif method == "tool/requestUserInput":
            await self._write(
                {
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": "Symphony does not support interactive user input.",
                    },
                }
            )
        elif method == "item/tool/call":
            await self._write(
                {
                    "id": request_id,
                    "result": {
                        "contentItems": [
                            {
                                "type": "inputText",
                                "text": "Unsupported Symphony dynamic tool call.",
                            }
                        ],
                        "success": False,
                    },
                }
            )

    def _thread_start_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "cwd": str(self.workspace),
            "serviceName": "symphony",
        }
        if self.config.codex.approval_policy:
            params["approvalPolicy"] = self.config.codex.approval_policy
        if self.config.codex.thread_sandbox:
            params["sandbox"] = self.config.codex.thread_sandbox
        return params

    def _turn_start_params(self, prompt: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text": prompt}],
            "cwd": str(self.workspace),
        }
        if self.config.codex.approval_policy:
            params["approvalPolicy"] = self.config.codex.approval_policy
        if self.config.codex.turn_sandbox_policy:
            params["sandboxPolicy"] = self.config.codex.turn_sandbox_policy
        return params

    def _require_process(self) -> asyncio.subprocess.Process:
        if self.process is None:
            raise AgentRunnerError("codex_not_found", "Codex app-server is not running")
        return self.process

    async def _emit(
        self,
        event: str,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
        message: str | None = None,
        usage: dict[str, int] | None = None,
        rate_limits: dict[str, Any] | None = None,
    ) -> None:
        emitted = self.on_event(
            AgentEvent(
                event=event,
                timestamp=utc_now(),
                codex_app_server_pid=self.pid,
                thread_id=thread_id,
                turn_id=turn_id,
                message=message,
                usage=usage,
                rate_limits=rate_limits,
            )
        )
        if emitted is not None:
            await emitted


class AgentRunner:
    """Wrap workspace preparation, prompt rendering, hooks, and Codex turns."""

    def __init__(
        self,
        config: ServiceConfig,
        workflow: WorkflowDefinition,
        workspace_manager: WorkspaceManager,
        tracker: IssueTracker,
        client_factory: Callable[
            [ServiceConfig, Path, EventCallback],
            CodexAppServerClient,
        ] = CodexAppServerClient,
    ) -> None:
        self.config = config
        self.workflow = workflow
        self.workspace_manager = workspace_manager
        self.tracker = tracker
        self.client_factory = client_factory

    async def run_issue(
        self,
        issue: Issue,
        attempt: int | None,
        on_event: EventCallback,
    ) -> Workspace:
        """Run one worker lifetime for an issue."""

        workspace = await asyncio.to_thread(
            self.workspace_manager.create_for_issue,
            issue.identifier,
        )
        self.workspace_manager.validate_agent_cwd(workspace.path, workspace.path)
        client = self.client_factory(self.config, workspace.path, on_event)
        try:
            await asyncio.to_thread(
                self.workspace_manager.run_hook,
                "before_run",
                workspace.path,
                fatal=True,
                issue_identifier=issue.identifier,
            )
            await client.start()
            active_issue = issue
            for turn_number in range(1, self.config.agent.max_turns + 1):
                prompt = (
                    render_prompt(self.workflow, active_issue, attempt)
                    if turn_number == 1
                    else CONTINUATION_PROMPT
                )
                await client.run_turn(prompt)
                refreshed = await asyncio.to_thread(
                    self.tracker.fetch_issue_states_by_ids,
                    [active_issue.id],
                )
                if refreshed:
                    active_issue = refreshed[0]
                if active_issue.state.lower() not in self.config.active_state_keys:
                    break
        finally:
            await client.close()
            await asyncio.to_thread(
                self.workspace_manager.run_hook,
                "after_run",
                workspace.path,
                fatal=False,
                issue_identifier=issue.identifier,
            )
        return workspace


def _object_field(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    field_value = value.get(field_name)
    if not isinstance(field_value, dict):
        raise AgentRunnerError("response_error", f"Missing {field_name} object")
    return field_value


def _string_field(value: dict[str, Any], field_name: str) -> str:
    field_value = value.get(field_name)
    if not isinstance(field_value, str) or not field_value:
        raise AgentRunnerError("response_error", f"Missing {field_name} string")
    return field_value


def _event_name(method: str) -> str:
    return method.replace("/", "_")


def _extract_thread_id(params: Any) -> str | None:
    if not isinstance(params, dict):
        return None
    value = params.get("threadId")
    if isinstance(value, str):
        return value
    thread = params.get("thread")
    if isinstance(thread, dict) and isinstance(thread.get("id"), str):
        return thread["id"]
    return None


def _extract_turn_id(params: Any) -> str | None:
    if not isinstance(params, dict):
        return None
    value = params.get("turnId")
    if isinstance(value, str):
        return value
    turn = params.get("turn")
    if isinstance(turn, dict) and isinstance(turn.get("id"), str):
        return turn["id"]
    item = params.get("item")
    if isinstance(item, dict) and isinstance(item.get("id"), str):
        return item["id"]
    return None


def _extract_usage(method: str, params: Any) -> dict[str, int] | None:
    if method != "thread/tokenUsage/updated" or not isinstance(params, dict):
        return None
    for candidate in (params.get("total_token_usage"), params.get("usage"), params):
        if isinstance(candidate, dict):
            usage = {
                "input_tokens": _token_int(candidate, "input_tokens", "inputTokens"),
                "output_tokens": _token_int(candidate, "output_tokens", "outputTokens"),
                "total_tokens": _token_int(candidate, "total_tokens", "totalTokens"),
            }
            if any(usage.values()):
                return usage
    return None


def _extract_rate_limits(params: Any) -> dict[str, Any] | None:
    if isinstance(params, dict) and isinstance(params.get("rateLimits"), dict):
        return params["rateLimits"]
    return None


def _token_int(value: dict[str, Any], *names: str) -> int:
    for name in names:
        raw = value.get(name)
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
    return 0


def _message_summary(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:500]
