# Symphony Implementation Notes

This repository includes a local Python implementation of the OpenAI Symphony draft specification.

Run it with:

```bash
uv run python -m scripts.symphony path/to/WORKFLOW.md
```

If the workflow path is omitted, Symphony reads `./WORKFLOW.md`.

For local manual runs, the convenience wrapper checks the workflow file, `LINEAR_API_KEY`, and
`codex` before starting:

```bash
scripts/run_symphony.sh path/to/WORKFLOW.md
```

Direct file execution is also supported:

```bash
uv run scripts/symphony.py path/to/WORKFLOW.md
```

## Implemented Scope

- `WORKFLOW.md` loader with YAML front matter and strict prompt rendering.
- Typed config defaults and `$VAR` resolution for configured secret/path fields.
- Dynamic workflow reload on poll ticks; invalid reloads keep the last known-good config.
- Linear-compatible read adapter for candidate fetch, terminal fetch, and state refresh.
- Per-issue workspace creation, safe key sanitization, root containment checks, and hooks.
- Async orchestrator with dispatch sorting, concurrency limits, claims, reconciliation, retries,
  startup terminal cleanup, and runtime snapshots.
- Codex app-server stdio JSON-RPC client using `initialize`, `thread/start`, and `turn/start`.

## Safety Posture

This implementation targets trusted local automation environments. It enforces Symphony's baseline
filesystem invariants: Codex runs from the per-issue workspace directory, workspace paths must stay
under `workspace.root`, and issue identifiers are sanitized before becoming directory names.

Approval and sandbox policy are implementation-defined and pass through from `WORKFLOW.md`:

- `codex.approval_policy` maps to Codex `approvalPolicy`.
- `codex.thread_sandbox` maps to thread startup `sandbox`.
- `codex.turn_sandbox_policy` maps to turn startup `sandboxPolicy`.

When Codex asks for command/file approvals, Symphony auto-approves for the session. Interactive
user input requests fail immediately so runs do not stall indefinitely. Unsupported dynamic tool
calls return a structured failure result.

## Real Integration Checks

Unit tests use fake clients and do not contact Linear or Codex. Before production use, run a real
smoke test with an isolated Linear project, a disposable workspace root, valid `LINEAR_API_KEY`,
and a Codex configuration appropriate for the host's trust boundary.
