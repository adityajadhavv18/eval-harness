"""Run a pipeline over every golden question, timing each call and capturing failures."""

import time
from datetime import datetime

from pydantic import ValidationError

from eval_harness.schema import GoldenItem, PipelineFn, PipelineOutput, QuestionResult
from eval_harness.tracing import question_trace


def run_pipeline(
    pipeline_fn: PipelineFn,
    golden: list[GoldenItem],
    arch: str = "unnamed",
    run_label: str | None = None,
    verbose: bool = True,
) -> list[QuestionResult]:
    """Call pipeline_fn once per question, sequentially, with one LangSmith trace per question.

    Sequential on purpose: parallel calls would compete for the API and inflate each other's latency,
    making latency numbers incomparable between architectures.
    """
    run_label = run_label or f"{arch}-{datetime.now():%Y%m%d-%H%M%S}"  # groups this run's traces
    results = []
    for i, item in enumerate(golden, start=1):
        raw, output, error, latency_ms = None, None, None, 0.0

        try:
            # The exception is allowed to escape the trace so LangSmith marks the question as failed
            with question_trace(arch, item, run_label) as run:
                start = time.perf_counter()
                try:
                    raw = pipeline_fn(item.question)
                finally:
                    latency_ms = (time.perf_counter() - start) * 1000
                if run is not None:
                    run.add_outputs(_trace_outputs(raw))
        except Exception as e:  # one broken question must not kill the whole run
            raw, error = None, f"{type(e).__name__}: {e}"

        # Validate outside the timer so schema checking isn't billed to the pipeline's latency
        if error is None:
            try:
                output = raw if isinstance(raw, PipelineOutput) else PipelineOutput.model_validate(raw)
            except ValidationError as e:
                error = f"ContractViolation: pipeline returned an invalid PipelineOutput: {e.errors()[0]['msg']}"

        results.append(QuestionResult(golden_id=item.id, output=output, error=error, latency_ms=latency_ms))
        if verbose:
            status = "✅" if error is None else f"❌ {error[:80]}"
            print(f"  [{i:>2}/{len(golden)}] {item.id}  {latency_ms:7.0f} ms  {status}")

    return results


def _trace_outputs(raw) -> dict:
    if isinstance(raw, PipelineOutput):
        return raw.model_dump()
    return raw if isinstance(raw, dict) else {"output": repr(raw)}
