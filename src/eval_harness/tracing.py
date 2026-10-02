"""LangSmith tracing helpers shared by the harness and every sprint.

Sprints use two things from here:
  client = traced_openai()        # every LLM call logged with prompt, response, tokens
  @traceable(run_type="retriever") # mark your own steps so they nest under each question
"""

from contextlib import contextmanager

from langsmith import Client, trace, traceable, tracing_context
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from eval_harness.config import settings

__all__ = ["traceable", "traced_openai", "question_trace", "judge_tracing", "flush"]


def tracing_on() -> bool:
    return settings.langsmith_tracing and "LANGSMITH_API_KEY" not in settings.missing_keys()


def traced_openai() -> OpenAI:
    """An OpenAI client whose calls appear in LangSmith (plain client when tracing is off)."""
    return wrap_openai(OpenAI()) if tracing_on() else OpenAI()


@contextmanager
def question_trace(arch: str, item, run_label: str):
    """One root trace per golden question; anything @traceable inside nests under it."""
    if not tracing_on():
        yield None
        return
    with trace(
        name=f"{arch} · {item.id}",
        run_type="chain",
        inputs={"question": item.question},
        tags=[f"arch:{arch}", f"type:{item.question_type}", run_label],
        metadata={"arch": arch, "golden_id": item.id, "question_type": item.question_type,
                  "source_doc": item.source_doc, "llm_model": settings.llm_model, "run": run_label},
        project_name=settings.langsmith_project,
    ) as run:
        yield run


@contextmanager
def judge_tracing():
    """Send RAGAS / judge calls to a separate '<project>-judge' project so they don't bury pipeline traces."""
    with tracing_context(enabled=tracing_on(), project_name=f"{settings.langsmith_project}-judge"):
        yield


def flush():
    """Traces upload in the background; call before the process exits so none are lost."""
    if tracing_on():
        Client().flush()
