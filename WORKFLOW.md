---
tracker:
  kind: bd
  active_states:
    - open
    - in_progress
  terminal_states:
    - closed

polling:
  interval_ms: 30000

workspace:
  root: .symphony/workspaces

hooks:
  timeout_ms: 60000
  after_create: |
    git clone "$SYMPHONY_SERVICE_CWD" .
    git checkout -B "symphony/${SYMPHONY_ISSUE_IDENTIFIER}"
  before_run: |
    uv sync

agent:
  max_concurrent_agents: 1
  max_turns: 20
  max_retry_backoff_ms: 300000

codex:
  command: codex app-server
  approval_policy: on-request
  turn_timeout_ms: 3600000
  read_timeout_ms: 5000
  stall_timeout_ms: 300000
---

You are working on a bd issue for the GaRAG repository.

Issue:
- ID: {{ issue.id }}
- Identifier: {{ issue.identifier }}
- Title: {{ issue.title }}
- State: {{ issue.state }}
- URL: {{ issue.url }}
- Labels: {{ issue.labels }}
- Attempt: {{ attempt }}

Description:
{{ issue.description }}

Follow the repository AGENTS.md workflow:
- use bd for tracking
- keep changes scoped to the issue
- run relevant quality gates
- commit with a Russian message
- sync and push before handing off
