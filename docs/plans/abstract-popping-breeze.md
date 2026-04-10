# d10 — Generation evaluation (`garag-zqc.18` + `.19`)

## Context

d9 закончен: `app/rag/generator.py` + `scripts/validate_generator.py` дают
20/20 parsed, 20/20 grounded, p50=2.2s / p95=4.7s на первых 20 запросах
golden_set_v1 с `qwen3.5:35b` + полным retrieval + reranker (см.
`evaluation/reports/generator_smoke.md`).

Следующий шаг по GigaSchool-плану (`~/kurs/gigaschool/docs/plans/radiant-questing-sutton.md`
row d10): **LLM-as-judge faithfulness/correctness + FastAPI `/query` + per-stage
tracing**. По договорённости с пользователем скоуп d10 **сужается до evaluation**
(`.18` + `.19`); FastAPI `/query` (`.20`) переносится на d10.5 / d11. Причина:
две задачи eval уже составляют ~1-2 ч работы + ~1-1.5 ч прогон моделей, и
честная оценка генератора должна предшествовать веб-обёртке — иначе мы
рискуем коммитить API поверх неизмеренного качества.

Попутно добавляется `latency_ms` в `QueryResponse` (решение пользователя) —
это поле понадобится и для tuning-метрик `.18`, и для будущего FastAPI
`.20`, и для NFR benchmark `.26`. Instrumentation на уровне `HybridRetriever`
и нового `QueryPipeline` делается сейчас, чтобы `.20` свёлся к тонкой
ASGI-обёртке.

**Scope (bd issues):**
- `garag-zqc.18` — `scripts/tune_gen_params.py` + `experiments/04_generation_params.ipynb` + обновление `docs/design.md`
- `garag-zqc.19` — `scripts/eval_generation.py` + `evaluation/reports/generation_report.md` (LLM-as-judge + mechanical citation_acc + format_rate)

**Non-scope (переносится в d10.5 / d11):**
- `garag-zqc.20` — FastAPI `/query` + `/health` + `/metrics` + structlog
- `garag-zqc.21` — Gradio UI
- Prometheus + Grafana wiring (d11, `.22`)

## Key decisions

1. **`QueryResponse.latency_ms: dict[str, float] | None = None`** — новое опциональное поле, ключи свободные (`dense`, `sparse`, `fusion`, `rerank`, `gen`, `total`, + будущие `guardrails_*`). Значения в миллисекундах, округлены до 1 знака. Default `None` — обратная совместимость с d9 `validate_generator.py` и существующими contract-тестами.
2. **`HybridRetriever.retrieve(..., timings: dict | None = None)`** — новый kwarg, writable hook. Caller владеет dict (`retrieve` только пишет ключи, не чистит). Keys: `dense`, `sparse`, `fusion`, `rerank` (последний только если `self.reranker is not None`). Значения в секундах (`perf_counter()`), конверсия в мс — на границе `QueryPipeline`. Default `None` → zero intrusion для d8 тестов.
3. **`app/rag/query_pipeline.py::QueryPipeline`** — тонкая композиция `HybridRetriever + Generator`, замеряет 5 стадий + `total`, возвращает `QueryResponse` с заполненным `latency_ms`. Это же будущий entry point для FastAPI `.20`. DI — принимает готовые `retriever` / `generator` в `__init__`, без ленивой инициализации (иначе юнит-тесты подтянут Qdrant).
4. **`.18` tune метрики — без LLM-judge.** Per-config measure: `format_rate`, `grounded_rate`, `mean_confidence`, `mean_citations`, `mean/p95_latency_gen_ms`, `n_failures`. Score: `mean_confidence + 0.1*mean_citations - 0.02*mean_latency_gen_s`, tiebreak на `grounded_rate desc` потом `p95_latency asc`. Judge держим для `.19` — иначе grid выльется в лишние 1-2 часа без новой информации (judge-scores на одинаковых chunks почти не зависят от decode temperature).
5. **`.19` LLM-as-judge — 3 dimensions, integer 0..2 Likert.** Выносим отдельный класс `app/rag/judge.py::Judge` (зеркало `Generator`: httpx + `/api/chat` + `think=false` + structured output). Параметры: `temperature=0.0, top_p=1.0, num_predict=300, seed=42`. Model = `settings.ollama_judge_model` (= qwen3.5:35b). Три dimensions:
   - `faithfulness ∈ [0..2]` — поддерживает ли контекст claims в ответе
   - `correctness ∈ [0..2]` — совпадает ли по сути с golden answer
   - `citation_support ∈ [0..2]` — реально ли процитированные chunks подтверждают quote (дополнение к mechanical `citation_acc`, который проверяет только `chunk_id ∈ retrieved_ids`)
   - `rationale: str` (≤500 chars, debugging tool)

   Нормализация в отчёт: `score / 2 → [0, 1]`, чтобы сравнивать с NFR из `docs/design.md §3` (faithfulness ≥ 0.80, correctness ≥ 0.70, citation_acc ≥ 0.85).
6. **Self-bias caveat** — жирно в header `generation_report.md`. Генератор и судья — одна модель (qwen3.5:35b), по литературе это завышает faithfulness на 5-15% vs cross-model judge. Cross-model rerun отложен в PoxekBook increment 2.

## Files

### Edited
| Path | Change |
|---|---|
| `app/schemas.py` | `+latency_ms: dict[str, float] \| None = None` в `QueryResponse` |
| `app/rag/pipeline.py` | `+timings: dict[str, float] \| None = None` kwarg в `HybridRetriever.retrieve`; обёртки `perf_counter` вокруг dense/sparse/fusion/rerank |
| `app/config.py` | После `.18` — обновить `gen_temperature` / `gen_top_p` / `gen_num_predict` на выбранные значения |
| `tests/test_schemas.py` | +4 теста на `latency_ms` (default None, dict roundtrip, json_schema properties, forbid extra) |
| `tests/test_pipeline.py` | +3 теста на `timings` hook (keys populated, without reranker → нет `rerank`, backward compat без kwarg) |
| `docs/design.md` | Новая подсекция §4.x с выбранным gen config + оправдание по grid |

### New
| Path | Purpose |
|---|---|
| `app/rag/query_pipeline.py` | `QueryPipeline` — композиция retriever + generator с per-stage latency |
| `app/rag/judge.py` | `Judge` класс — `/api/chat` судья с `_JudgeVerdict` schema |
| `scripts/tune_gen_params.py` | Grid search `.18`, пишет `evaluation/results/gen_params_grid.json` |
| `scripts/eval_generation.py` | LLM-judge harness `.19`, пишет отчёт + raw json |
| `experiments/04_generation_params.ipynb` | Короткий ноутбук: загрузить grid JSON, table + heatmap, chosen config |
| `evaluation/reports/generation_report.md` | Отчёт `.19` (commit после первого прогона) |
| `evaluation/results/gen_params_grid.json` | Сырые результаты grid (commit) |
| `evaluation/results/generation_eval.json` | Сырые per-query verdicts судьи (commit, для reproducibility) |
| `tests/test_query_pipeline.py` | +6 mock-тестов через `_StubRetriever` + `httpx.MockTransport` |
| `tests/test_judge.py` | +3 mock-теста на судью |

## Signatures

### `app/rag/query_pipeline.py`

```python
class QueryPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        *,
        candidate_k: int = 20,
        top_k: int = 5,
    ) -> None: ...

    def query(
        self,
        question: str,
        *,
        candidate_k: int | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        """retrieve → generate → QueryResponse с populated latency_ms.

        latency_ms keys: dense, sparse, fusion, rerank (if reranker), gen, total.
        Все значения в миллисекундах, round(x, 1).
        """
```

Реализация: `timings = {}; retriever.retrieve(q, timings=timings); tg0=perf_counter(); resp=generator.generate(q, chunks); timings["gen"]=perf_counter()-tg0; timings["total"]=perf_counter()-t0; latency_ms={k: round(v*1000,1) for k,v in timings.items()}; return resp.model_copy(update={"latency_ms": latency_ms})`.

### `app/rag/judge.py`

```python
class _JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    faithfulness: int = Field(ge=0, le=2)
    correctness: int = Field(ge=0, le=2)
    citation_support: int = Field(ge=0, le=2)
    rationale: str = Field(min_length=1, max_length=500)


class Judge:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,  # default → settings.ollama_judge_model
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_predict: int = 300,
        seed: int = 42,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None: ...

    def judge(
        self,
        *,
        question: str,
        golden: str,
        candidate: QueryResponse,
        chunks: list[ScoredChunk],
    ) -> _JudgeVerdict: ...
```

Внутри — `/api/chat`, `think=false`, `format=_JudgeVerdict.model_json_schema()`, `keep_alive`, strict JSON decode, `model_validate`. Structural prompts (см. §"Judge prompts").

### `scripts/tune_gen_params.py` — grid

```
temperature ∈ {0.0, 0.1, 0.2, 0.4}                 # 4 значения
top_p       ∈ {0.8, 0.9, 1.0}                      # 3 значения
num_predict ∈ {400, 800, 1200}                     # 3 значения
# итого 4 × 3 × 3 = 36 configs
```

Bound: 36 configs × 20 queries × ~5 s gen latency ≈ **60 min** на RTX 5090.
Retrieval кешируется один раз на запрос перед grid loop — те же chunks для
всех configs, меняется только generation. Если время оценочно > 75 мин (по
первым 2-3 configs) — drop `num_predict=1200` → 24 configs ≈ 40 min.

Per-config output в JSON:
```json
{"config": {"temperature": 0.2, "top_p": 0.9, "num_predict": 800},
 "format_rate": 1.0, "grounded_rate": 1.0,
 "mean_confidence": 0.87, "mean_citations": 1.35,
 "mean_latency_gen_ms": 4820.1, "p95_latency_gen_ms": 6120.0,
 "heuristic_score": 0.909, "n_queries": 20, "n_failures": 0}
```

Ranking: `heuristic_score desc → grounded_rate desc → p95_latency asc`. Top-10 в stdout.

### Judge prompts (English, см. generator stylistic baseline)

**System:**
```
You are a strict evaluator of cybersecurity Q&A systems. Given a question,
a reference answer (golden), a candidate answer, and the retrieval context,
you assign three integer scores in [0, 2] and a brief rationale.

Scoring rubric:

faithfulness  (is the candidate answer supported by the provided context?)
  0 — claims not in context or contradict it
  1 — partially supported; at least one claim unsupported or hedged
  2 — every factual claim directly supported by context

correctness  (does the candidate answer match the reference answer on substance?)
  0 — wrong or off-topic
  1 — partially correct (right topic, wrong details, or missing key fact)
  2 — semantically equivalent to the reference (paraphrase is fine)

citation_support  (do cited chunks actually back the quoted claims?)
  0 — citations missing, wrong, or do not support the quoted text
  1 — citations exist but at least one is loosely related / off-topic
  2 — every citation is on-point and directly supports its quoted claim

Output a single JSON object: {faithfulness, correctness, citation_support, rationale}.
Keep rationale under 400 characters. No markdown fences, no extra keys. Be strict:
when in doubt, score lower.
```

**User template:** question / golden / candidate_answer / citations_block /
context_block (context_block переиспользует `_format_context` из
`app/rag/generator.py` — либо лифтим в `app/rag/_context.py` для чистого
shared use, либо импорт underscored helper как in-package private).

## Work order

**Phase A — schema + instrumentation** (~30 мин, unblocks everything)
1. `app/schemas.py`: `+latency_ms` в `QueryResponse`.
2. `app/rag/pipeline.py`: `+timings` kwarg + `perf_counter()` wrappers.
3. Обновить `tests/test_schemas.py` (+4 теста) и `tests/test_pipeline.py` (+3 теста).
4. Lint + ty + pytest.
5. Commit: `feat(schemas,pipeline): latency_ms и timings hook (d10 prep)`.

**Phase B — `QueryPipeline` wrapper** (~30 мин)
6. `app/rag/query_pipeline.py` — composition class, ~60 LOC.
7. `tests/test_query_pipeline.py` — 6 mock-тестов через stub retrievers + httpx.MockTransport.
8. Lint + pytest.
9. Commit: `feat(rag): QueryPipeline — композиция retriever+generator с per-stage latency`.

**Phase C — `.18` tune_gen_params** (~60-90 мин walltime)
10. `scripts/tune_gen_params.py` — grid + per-config metrics + JSON output.
11. Прогон: `OLLAMA_URL=http://localhost:11434 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m scripts.tune_gen_params --limit 20`.
12. Перенести top-1 config в `app/config.py`.
13. `experiments/04_generation_params.ipynb` — таблица top-10, heatmap temperature × top_p по score.
14. `docs/design.md` §4.x — короткий абзац + итоговая таблица.
15. `bd close garag-zqc.18`.
16. Commit: `feat(eval): tune_gen_params + ноутбук 04 (best: temp=X top_p=Y num_predict=Z)`.

**Phase D — `.19` eval_generation** (~15-20 мин walltime)
17. `app/rag/judge.py` — Judge class + `_JudgeVerdict`.
18. `tests/test_judge.py` — 3 mock-теста.
19. `scripts/eval_generation.py` — полный pipeline + judge loop + report renderer.
20. Прогон: `OLLAMA_URL=http://localhost:11434 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m scripts.eval_generation`.
21. Ручной review 10 sample rows в отчёте — sanity check.
22. `bd close garag-zqc.19`.
23. Commit: `feat(eval): eval_generation LLM-as-judge + отчёт (faith=X corr=Y cite=Z)`.

**Phase E — landing the plane**
24. `bd sync` + `git pull --rebase` + `git push`.

## Verification

- `uv run ruff check app tests scripts` → clean
- `uv run ty check` → clean (особенно `dict[str, float] | None` narrowing в `retrieve`)
- `uv run pytest --cov=app --cov-report=term-missing` → 66 → 82 tests (+16), coverage ≥ 60%
- `uv run python -m scripts.validate_generator --limit 5` — регрессия не допустима, 5/5 parsed (схема `QueryResponse` обратно-совместима по `latency_ms=None`)
- `evaluation/results/gen_params_grid.json` создан, top-3 напечатаны
- `evaluation/reports/generation_report.md` создан, содержит overall + per-category + 10 manual samples + self-bias caveat
- Выбранный gen config закоммичен в `app/config.py`, и `validate_generator` на нём всё ещё даёт 20/20 parsed
- `bd ready` показывает `.20` как разблокированный (старый статус `.17 → .20` остаётся, `.19 → .27`)

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Grid (36 × 20 × ~5s) > 60 мин | Cache retrieve() один раз на запрос; если проекция > 75 мин — drop `num_predict=1200` → 24 configs |
| Stochasticity qwen3.5:35b на `temp=0.4` | `seed=42` pinned; report `n_failures`; при близких top configs (< 1% delta) — prefer более низкую temperature |
| Self-bias судьи (одна модель) | Caveat в header отчёта; полагаться на per-category deltas и ручной 10-sample review, не на абсолют |
| Judge structured output falls back на MoE (как было в d9) | `_JudgeVerdict` нарочно плоский (3 int + 1 str). При flapping → retry с `num_predict=500`. Считаем `judge_format_failures` в отчёте |
| `num_predict=400` ломает tool_usage walkthroughs | Видим по `format_rate < 1.0` в grid → выбираем следующий more conservative config |
| `latency_ms` ломает существующие JSON consumers | Field optional, default None → старые клиенты парсят без изменений |

## Out of scope

Следующие вещи **не делаются** в d10 и уезжают в d10.5 / d11 / PoxekBook:

- FastAPI `/query` + `/health` + `/metrics` → `garag-zqc.20` (d10.5)
- Gradio UI → `garag-zqc.21` (d11)
- Prometheus `prometheus_client` metrics wiring → `garag-zqc.20` scope
- structlog configuration → `garag-zqc.20` scope
- Dockerfile / docker-compose `app` service → `garag-zqc.22` (d11)
- Cross-model judge rerun (GPT-4o / Claude) → PoxekBook increment 2
- Ground truth citations в golden_set (currently `relevant_chunks` есть, но нет gold `quote` per citation) → PoxekBook
