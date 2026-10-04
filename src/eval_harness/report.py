"""The single entry point (evaluate), saving results, baselines, and printed reports.

    from eval_harness import evaluate
    evaluate(my_pipeline, arch="naive")

Compare any two saved runs from the terminal:
    uv run compare-runs results/naive_....json results/hybrid_....json
"""

import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

from eval_harness.config import settings
from eval_harness.golden import load_golden
from eval_harness.metrics import score_run
from eval_harness.runner import run_pipeline
from eval_harness.schema import PipelineFn, RunResult
from eval_harness.tracing import flush

# (key, label) per section; order here is the order printed
SECTIONS = {
    "Retrieval (free)": [("evidence_recall", "evidence recall"), ("evidence_mrr", "evidence MRR"),
                         ("doc_hit", "doc hit"), ("doc_mrr", "doc MRR")],
    "Generation (judge)": [("faithfulness", "faithfulness"), ("answer_relevancy", "answer relevancy"),
                           ("context_precision", "context precision"), ("context_recall", "context recall")],
    "No-answer (judge)": [("abstention", "abstention")],
    "Engineering": [("latency_p50_ms", "latency p50 (ms)"), ("latency_p95_ms", "latency p95 (ms)"),
                    ("cost_per_query_usd", "cost / query ($)"), ("judge_cost_usd", "judge cost ($)"),
                    ("error_rate", "error rate")],
}
LOWER_IS_BETTER = {"latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "cost_per_query_usd",
                   "cost_total_usd", "judge_cost_usd", "error_rate"}
BY_TYPE_KEYS = ["evidence_recall", "faithfulness", "context_recall", "abstention"]


def evaluate(
    pipeline_fn: PipelineFn,
    arch: str,
    golden_set: str = "golden_v1.jsonl",
    use_llm_judge: bool = True,
    compare_to: str | Path | None = "baseline",
    save: bool = True,
) -> RunResult:
    """Run, score, save, print. compare_to: "baseline" (this golden set's baseline), a results path, or None."""
    golden = load_golden(golden_set)
    started_at = datetime.now()
    print(f"▶ evaluating '{arch}' on {golden_set} ({len(golden)} questions)")

    questions = run_pipeline(pipeline_fn, golden, arch=arch, run_label=f"{arch}-{started_at:%Y%m%d-%H%M%S}")
    print("▶ scoring" + (" (LLM judge on)" if use_llm_judge else " (free metrics only)"))
    summary = score_run(golden, questions, use_llm_judge=use_llm_judge)

    run = RunResult(arch=arch, started_at=started_at, llm_model=settings.llm_model,
                    judge_model=settings.judge_model, embed_model=settings.embed_model,
                    golden_set=golden_set, questions=questions, summary=summary)
    if save:
        path = save_run(run)
        print(f"▶ saved {path.relative_to(settings.results_dir.parent)}")
    flush()

    print_report(run, _resolve_reference(compare_to, golden_set))
    return run


# ---------- Files ----------

def save_run(run: RunResult) -> Path:
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.results_dir / f"{run.arch}_{run.started_at:%Y%m%d-%H%M%S}.json"
    path.write_text(run.model_dump_json(indent=2))
    return path


def load_run(path: str | Path) -> RunResult:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = settings.results_dir / path
    return RunResult.model_validate_json(path.read_text())


def baseline_path(golden_set: str) -> Path:
    """golden_v1.jsonl -> results/baseline_v1.json, so each golden version has its own baseline."""
    return settings.results_dir / (Path(golden_set).stem.replace("golden", "baseline") + ".json")


def set_baseline(result_path: str | Path) -> Path:
    """Promote a saved run to the baseline for its golden set. Deliberately a separate, explicit step."""
    run = load_run(result_path)
    target = baseline_path(run.golden_set)
    shutil.copyfile(Path(result_path) if Path(result_path).exists() else settings.results_dir / result_path, target)
    print(f"✅ '{run.arch}' run from {run.started_at:%Y-%m-%d %H:%M} is now the baseline for {run.golden_set}")
    return target


def _resolve_reference(compare_to, golden_set: str) -> RunResult | None:
    if compare_to is None:
        return None
    path = baseline_path(golden_set) if compare_to == "baseline" else Path(compare_to)
    if not path.exists():
        if compare_to == "baseline":
            print(f"ℹ️  no baseline yet for {golden_set}; promote a run with set_baseline(<results file>)")
        return None
    return load_run(path)


# ---------- Printing ----------

def print_report(run: RunResult, reference: RunResult | None = None):
    if reference is not None and reference.golden_set != run.golden_set:
        print(f"⚠️  not comparing: reference used {reference.golden_set}, this run used {run.golden_set}")
        reference = None

    title = f"{run.arch}  ·  {run.started_at:%Y-%m-%d %H:%M}  ·  {run.golden_set}  ·  llm={run.llm_model}"
    print(f"\n{'═' * 72}\n {title}")
    if reference:
        print(f" vs {reference.arch} ({reference.started_at:%Y-%m-%d %H:%M})")
    print("═" * 72)

    for section, metrics in SECTIONS.items():
        rows = [(k, label) for k, label in metrics if k in run.summary]
        if not rows:
            continue
        print(f"\n {section}")
        for key, label in rows:
            line = f"   {label:<22}{_fmt(key, run.summary[key]):>10}"
            if reference and key in reference.summary:
                line += f"   {_delta(key, run.summary[key], reference.summary[key])}"
            print(line)

    _print_by_type(run)
    _print_worst(run)
    errors = [q for q in run.questions if q.error]
    if errors:
        print(f"\n Errors ({len(errors)})")
        for q in errors:
            print(f"   {q.golden_id}: {q.error[:90]}")
    print()


def _print_by_type(run: RunResult):
    types = {g.id: g.question_type for g in load_golden(run.golden_set)}
    keys = [k for k in BY_TYPE_KEYS if any(k in q.scores for q in run.questions)]
    print(f"\n By question type\n   {'type':<12}{'n':>4}" + "".join(f"{k.replace('_', ' '):>18}" for k in keys))
    for qtype in ["simple", "multi_part", "exact_term", "no_answer"]:
        qs = [q for q in run.questions if types.get(q.golden_id) == qtype]
        if not qs:
            continue
        cells = []
        for k in keys:
            vals = [q.scores[k] for q in qs if k in q.scores]
            cells.append(f"{statistics.mean(vals):>18.2f}" if vals else f"{'-':>18}")
        print(f"   {qtype:<12}{len(qs):>4}" + "".join(cells))


def _print_worst(run: RunResult, n: int = 3):
    """Lowest-scoring answerable questions: the failure cases worth reading in LangSmith."""
    key = "faithfulness" if any("faithfulness" in q.scores for q in run.questions) else "evidence_recall"
    scored = sorted((q for q in run.questions if key in q.scores), key=lambda q: q.scores[key])[:n]
    if scored:
        print(f"\n Worst {len(scored)} by {key.replace('_', ' ')} (open these traces in LangSmith)")
        for q in scored:
            print(f"   {q.golden_id}  {q.scores[key]:.2f}  {(q.output.answer if q.output else '')[:70]!r}")


def _fmt(key: str, value: float) -> str:
    if key.endswith("_usd"):
        return f"{value:.5f}"
    if key.endswith("_ms"):
        return f"{value:.0f}"
    return f"{value:.2f}"


def _delta(key: str, new: float, old: float) -> str:
    diff = new - old
    if _fmt(key, abs(diff)) == _fmt(key, 0.0):  # too small to show at this precision
        return "  ="
    better = diff < 0 if key in LOWER_IS_BETTER else diff > 0
    return f"{'+' if diff > 0 else '-'}{_fmt(key, abs(diff))} {'✅' if better else '🔻'}"


# ---------- CLI ----------

def main():
    """uv run compare-runs <new.json> [<reference.json>]   (reference defaults to the baseline)"""
    if len(sys.argv) < 2:
        print(main.__doc__)
        sys.exit(1)
    run = load_run(sys.argv[1])
    reference = load_run(sys.argv[2]) if len(sys.argv) > 2 else _resolve_reference("baseline", run.golden_set)
    print_report(run, reference)
