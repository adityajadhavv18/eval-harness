"""The contract between the harness and every RAG sprint.

These shapes are frozen for the whole quarter: if they change, Day 1 scores
stop being comparable to Day 12 scores.

Flow:   GoldenItem --(question)--> pipeline_fn --> PipelineOutput
        runner wraps each PipelineOutput with timing/errors -> QuestionResult
        all QuestionResults + aggregate scores -> RunResult (saved to results/)
"""

from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["simple", "multi_part", "exact_term", "no_answer"]


class _Strict(BaseModel):
    # extra="forbid" turns a typo like "contexs" into an error instead of a silently ignored field
    model_config = ConfigDict(extra="forbid")


class GoldenItem(_Strict):
    """One line of a golden set file, e.g. golden_sets/golden_v1.jsonl."""

    id: str
    question: str = Field(min_length=5)
    ground_truth: str  # the reference answer; for no_answer items, what a correct refusal says
    source_doc: str | None = None  # which doc the answer lives in; None only for no_answer
    question_type: QuestionType = "simple"
    evidence: str | None = None  # verbatim quote from source_doc backing ground_truth, for human review

    @model_validator(mode="after")
    def _source_matches_type(self):
        if self.question_type == "no_answer" and self.source_doc is not None:
            raise ValueError(f"{self.id}: no_answer questions must not have a source_doc")
        if self.question_type != "no_answer" and self.source_doc is None:
            raise ValueError(f"{self.id}: answerable questions need a source_doc")
        return self


class PipelineOutput(_Strict):
    """What every sprint's pipeline must return for one question."""

    answer: str
    contexts: list[str]  # retrieved chunk texts, in rank order (RAGAS reads these)
    context_ids: list[str]  # source doc of each chunk, same order (hit-rate/MRR read these)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _contexts_aligned(self):
        if len(self.contexts) != len(self.context_ids):
            raise ValueError(
                f"contexts ({len(self.contexts)}) and context_ids ({len(self.context_ids)}) must be the same length"
            )
        return self


# The one function signature every sprint implements: question in, PipelineOutput out.
PipelineFn = Callable[[str], PipelineOutput]


class QuestionResult(_Strict):
    """One golden question after the runner has executed and scored it."""

    golden_id: str
    output: PipelineOutput | None = None  # None if the pipeline raised
    error: str | None = None
    latency_ms: float = Field(ge=0)  # measured by the runner, never by the pipeline
    scores: dict[str, float] = {}  # per-question metrics, filled in Phase 5


class RunResult(_Strict):
    """One full eval run; serialised to results/<arch>_<timestamp>.json."""

    arch: str  # e.g. "naive", "hybrid"
    started_at: datetime
    llm_model: str
    judge_model: str
    embed_model: str
    golden_set: str  # filename (e.g. golden_v1.jsonl), so you know which ruler produced these numbers
    questions: list[QuestionResult]
    summary: dict[str, float] = {}  # aggregate metrics, filled in Phase 5
