"""Score a run: free retrieval metrics, RAGAS generation metrics, abstention, latency and cost.

Two metric families, kept separate on purpose:
  retrieval  (string matching, free)  -> did the right text come back?
  generation (LLM judge, costs money) -> is the answer grounded and correct?
If retrieval is low, no prompt change will fix generation.
"""

import math
import re
import statistics

from eval_harness.config import settings
from eval_harness.schema import GoldenItem, QuestionResult

# USD per 1M tokens (input, output). Update when prices or models change.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

RAGAS_KEYS = {  # RAGAS column name -> our short name
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


# ---------- Retrieval (free) ----------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _evidence_fragments(item: GoldenItem) -> list[str]:
    # Evidence joins several verbatim quotes with " ... " or " / " (same convention as check_golden.py)
    return [_norm(f) for f in re.split(r" \.\.\. | / ", item.evidence or "") if f.strip()]


def retrieval_scores(item: GoldenItem, result: QuestionResult) -> dict[str, float]:
    """doc_hit/doc_mrr: right document retrieved. evidence_recall/evidence_mrr: right *passage* retrieved."""
    if item.question_type == "no_answer":
        return {}  # nothing correct to retrieve
    if result.output is None:  # pipeline crashed: count as a miss, don't silently drop it
        return {"doc_hit": 0.0, "doc_mrr": 0.0, "evidence_recall": 0.0, "evidence_mrr": 0.0}

    ids = result.output.context_ids
    doc_rank = next((i for i, d in enumerate(ids, start=1) if d == item.source_doc), None)
    scores = {"doc_hit": float(doc_rank is not None), "doc_mrr": 1 / doc_rank if doc_rank else 0.0}

    fragments = _evidence_fragments(item)
    if fragments:
        contexts = [_norm(c) for c in result.output.contexts]
        found = [f for f in fragments if any(f in c for c in contexts)]
        ev_rank = next((i for i, c in enumerate(contexts, start=1) if any(f in c for f in fragments)), None)
        scores["evidence_recall"] = len(found) / len(fragments)
        scores["evidence_mrr"] = 1 / ev_rank if ev_rank else 0.0
    return scores


# ---------- Generation (LLM judge) ----------

def ragas_scores(golden: list[GoldenItem], results: list[QuestionResult]) -> dict[str, dict[str, float]]:
    """RAGAS on answerable questions that ran successfully. Returns {golden_id: {metric: score}}."""
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    by_id = {g.id: g for g in golden}
    scored = [r for r in results if r.output is not None and by_id[r.golden_id].question_type != "no_answer"]
    if not scored:
        return {}

    samples = [
        SingleTurnSample(
            user_input=by_id[r.golden_id].question,
            response=r.output.answer,
            retrieved_contexts=r.output.contexts,
            reference=by_id[r.golden_id].ground_truth,
        )
        for r in scored
    ]
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(ChatOpenAI(model=settings.judge_model, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name=settings.embed_model)),
    )
    df = result.to_pandas()
    out = {}
    for r, (_, row) in zip(scored, df.iterrows()):
        # RAGAS returns NaN when a judge call fails; drop it rather than count it as 0
        out[r.golden_id] = {ours: float(row[col]) for col, ours in RAGAS_KEYS.items()
                            if col in row and not math.isnan(row[col])}
    return out


def abstention_score(question: str, answer: str) -> float:
    """For no_answer questions: 1.0 if the pipeline declined instead of inventing an answer."""
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=settings.judge_model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": "The question below cannot be answered from the available documents.\n"
                       f"Question: {question}\nAnswer given: {answer}\n\n"
                       "Did the answer decline, or say the information is not available, rather than "
                       "giving a specific figure or fact? Reply with exactly one word: yes or no.",
        }],
    )
    return 1.0 if resp.choices[0].message.content.strip().lower().startswith("yes") else 0.0


# ---------- Cost ----------

def cost_usd(tokens_in: int, tokens_out: int, model: str) -> float | None:
    """Generation cost of one pipeline call. None if the model isn't in PRICES."""
    price = next((p for name, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0])) if model.startswith(name)), None)
    if price is None:
        return None
    return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000


# ---------- Put it together ----------

def score_run(golden: list[GoldenItem], results: list[QuestionResult], use_llm_judge: bool = True) -> dict[str, float]:
    """Fill each result.scores in place and return the run summary."""
    by_id = {g.id: g for g in golden}

    for r in results:
        r.scores = retrieval_scores(by_id[r.golden_id], r)
        if r.output is not None:
            cost = cost_usd(r.output.tokens_in, r.output.tokens_out, settings.llm_model)
            if cost is not None:
                r.scores["cost_usd"] = cost

    if use_llm_judge:
        for gid, scores in ragas_scores(golden, results).items():
            next(r for r in results if r.golden_id == gid).scores.update(scores)
        for r in results:
            item = by_id[r.golden_id]
            if item.question_type == "no_answer":
                r.scores["abstention"] = abstention_score(item.question, r.output.answer) if r.output else 0.0

    return summarize(results)


def summarize(results: list[QuestionResult]) -> dict[str, float]:
    """Average every per-question metric over the questions that have it, plus latency, cost and errors."""
    summary: dict[str, float] = {"n_questions": len(results),
                                 "error_rate": sum(r.error is not None for r in results) / len(results)}

    metric_names = sorted({k for r in results for k in r.scores if k != "cost_usd"})
    for name in metric_names:
        values = [r.scores[name] for r in results if name in r.scores]
        summary[name] = statistics.mean(values)

    latencies = sorted(r.latency_ms for r in results if r.error is None)
    if latencies:
        summary["latency_mean_ms"] = statistics.mean(latencies)
        summary["latency_p50_ms"] = statistics.median(latencies)
        summary["latency_p95_ms"] = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)]

    costs = [r.scores["cost_usd"] for r in results if "cost_usd" in r.scores]
    if costs:
        summary["cost_total_usd"] = sum(costs)
        summary["cost_per_query_usd"] = statistics.mean(costs)
    return summary
