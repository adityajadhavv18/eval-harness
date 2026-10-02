"""Single source of settings for the eval harness.

Every other module imports `settings` from here instead of calling os.getenv.
Values come from the nearest .env (searched upward from where you run `uv run`).
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings

# Load .env into os.environ (not just into Settings) because LangSmith and the
# OpenAI SDK read their env vars directly.
load_dotenv(find_dotenv(usecwd=True))

# src/eval_harness/config.py -> parents[2] is the eval_harness/ project root
HARNESS_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Models
    llm_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Secrets: SecretStr hides the value when printed or logged
    openai_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None

    # Tracing
    langsmith_tracing: bool = False
    langsmith_project: str = "rag-sprints"

    # Paths
    golden_dir: Path = HARNESS_ROOT / "golden_sets"
    results_dir: Path = HARNESS_ROOT / "results"

    def missing_keys(self) -> list[str]:
        """Names of required keys that are unset or still the .env.example placeholder."""
        missing = []
        if _is_placeholder(self.openai_api_key):
            missing.append("OPENAI_API_KEY")
        if self.langsmith_tracing and _is_placeholder(self.langsmith_api_key):
            missing.append("LANGSMITH_API_KEY")
        return missing


def _is_placeholder(key: SecretStr | None) -> bool:
    return key is None or key.get_secret_value() in ("", "sk-...", "lsv2_...")


settings = Settings()

# Tracing on with a placeholder key makes LangChain spam 403 errors on every call: turn it off instead
if settings.langsmith_tracing and "LANGSMITH_API_KEY" in settings.missing_keys():
    print("⚠️  LANGSMITH_TRACING=true but LANGSMITH_API_KEY is not set: tracing disabled for this run")
    settings.langsmith_tracing = False
    os.environ["LANGSMITH_TRACING"] = "false"
