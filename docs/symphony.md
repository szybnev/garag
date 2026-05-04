# Symphony Implementation Notes

This repository includes a local Python implementation of the OpenAI Symphony draft specification.

Run it with:

```bash
uv run scripts/symphony.py path/to/WORKFLOW.md
```

If the workflow path is omitted, Symphony reads `./WORKFLOW.md`.

The repository ships a default `WORKFLOW.md` that uses the local bd tracker:

- candidate dispatch comes from `bd ready --json`
- state refresh and terminal cleanup read `.beads/issues.jsonl`

For local manual runs, the convenience wrapper checks the workflow file and `codex` before starting:

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
- Local bd read adapter for candidate fetch, terminal fetch, and state refresh.
- Optional legacy Linear-compatible read adapter for external tracker experiments.
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

Unit tests use fake clients and do not contact external trackers or Codex. Before production use,
run a real smoke test with local bd issues, a disposable workspace root, and a Codex configuration
appropriate for the host's trust boundary.

Linear remains available only as a legacy adapter. To use it, set `tracker.kind: linear` and provide
`tracker.api_key`/`tracker.project_slug` in the workflow, for example via `$LINEAR_API_KEY` and
`$LINEAR_PROJECT_SLUG`.
