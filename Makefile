.PHONY: help sync lint format typecheck test up down logs ingest eval garak clean

help:
	@echo "Targets:"
	@echo "  sync       - uv sync (install deps)"
	@echo "  lint       - ruff check"
	@echo "  format     - ruff format"
	@echo "  typecheck  - ty check app/"
	@echo "  test       - pytest tests/"
	@echo "  up         - docker compose up -d"
	@echo "  down       - docker compose down"
	@echo "  logs       - docker compose logs -f app"
	@echo "  ingest     - fetch + parse + chunk + build indices"
	@echo "  eval       - run retrieval + generation evaluation"
	@echo "  garak      - planned security scan placeholder"
	@echo "  clean      - remove caches and build artefacts"

sync:
	uv sync

lint:
	uv run ruff check app/ scripts/ tests/

format:
	uv run ruff format app/ scripts/ tests/

typecheck:
	uv run ty check app/

test:
	uv run pytest tests/ -v

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f app

ingest:
	uv run python -m scripts.fetch_mitre_attack
	uv run python -m scripts.fetch_mitre_atlas
	uv run python -m scripts.fetch_owasp_top10
	uv run python -m scripts.fetch_hackerone_reports --limit 500
	uv run python -m scripts.fetch_man_pages
	uv run python -m scripts.parse_sources
	uv run python -m scripts.chunk_corpus
	uv run python -m scripts.build_bm25
	uv run python -m scripts.build_qdrant

eval:
	uv run python -m scripts.eval_retrieval --golden data/golden/golden_set_v1.jsonl
	uv run python -m scripts.eval_generation --golden data/golden/golden_set_v1.jsonl

garak:
	@echo "garak runner is planned but not implemented in GaRAG runtime MVP"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ dist build htmlcov .coverage
