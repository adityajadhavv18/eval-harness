"""Run a pipeline over every golden question, timing each call and capturing failures."""

import time

from pydantic import ValidationError

from eval_harness.schema import GoldenItem, PipelineFn, PipelineOutput, QuestionResult


def run_pipeline(pipeline_fn: PipelineFn, golden: list[GoldenItem], verbose: bool = True) -> list[QuestionResult]:
    """Call pipeline_fn once per question, sequentially.

    Sequential on purpose: parallel calls would compete for the API and inflate each other's latency,
    making latency numbers incomparable between architectures.
    """
    results = []
    for i, item in enumerate(golden, start=1):
        output, error = None, None

        start = time.perf_counter()
        try:
            raw = pipeline_fn(item.question)
        except Exception as e:  # one broken question must not kill the whole run
            raw, error = None, f"{type(e).__name__}: {e}"
        latency_ms = (time.perf_counter() - start) * 1000

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
