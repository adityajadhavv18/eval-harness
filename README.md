# eval-harness

A reusable evaluation harness for RAG pipelines. Every architecture in [rag-sprints](https://github.com/adityajadhavv18/rag-sprints) is scored against the same golden set, so results are directly comparable.

## Usage

```python
from eval_harness import evaluate, set_baseline

evaluate(my_pipeline, arch="hybrid")          # run, score, save results/hybrid_<time>.json, print report
set_baseline("naive_20261003-1430.json")      # promote a run to results/baseline_v1.json (explicit step)
```

```bash
uv run compare-runs hybrid_....json           # report vs the baseline
uv run compare-runs hybrid_....json naive_....json
```

## The contract

A pipeline is any function `question: str -> PipelineOutput`:

```python
PipelineOutput(
    answer="...",
    contexts=["retrieved chunk text", ...],   # rank order
    context_ids=["aapl_10k_2025", ...],       # source doc of each chunk
    tokens_in=850, tokens_out=20,
)
```

The runner, not the pipeline, measures latency, so timing is identical across architectures.

## Metrics

| Family | Metrics | Cost |
|---|---|---|
| Retrieval | doc hit / MRR, evidence recall / MRR (did the exact answer passage come back?) | free |
| Generation | RAGAS faithfulness, answer relevancy, context precision, context recall | LLM judge |
| No-answer | abstention (declined instead of inventing an answer) | LLM judge |
| Engineering | latency mean / p50 / p95, cost per query, error rate | free |

## Golden set

`golden_sets/golden_v1.jsonl`: 30 questions over Apple, Microsoft and Tesla 10-Ks (15 simple, 8 multi-part, 4 exact-term, 3 unanswerable). Each item has a verbatim `evidence` quote from its source document. Changing questions means creating a new version file and re-running the baseline.

## Setup

Uses [uv](https://docs.astral.sh/uv/). Configuration is read from the `.env` of the project that imports the harness (see `rag-sprints/.env.example`).
