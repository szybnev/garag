#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOW_PATH="${1:-${SYMPHONY_WORKFLOW:-$ROOT_DIR/WORKFLOW.md}}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<EOF
Usage:
  scripts/run_symphony.sh [path/to/WORKFLOW.md]

Environment:
  SYMPHONY_WORKFLOW=/path/to/WORKFLOW.md
EOF
  exit 0
fi

if [[ ! -f "$WORKFLOW_PATH" ]]; then
  echo "WORKFLOW.md not found: $WORKFLOW_PATH" >&2
  echo "Run scripts/run_symphony.sh --help for usage." >&2
  exit 2
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex executable not found in PATH." >&2
  exit 2
fi

cd "$ROOT_DIR"
exec uv run python -m scripts.symphony "$WORKFLOW_PATH"
