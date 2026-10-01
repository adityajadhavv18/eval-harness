"""Load and validate a golden set (one GoldenItem per line of a .jsonl file)."""

import json
from pathlib import Path

from pydantic import ValidationError

from eval_harness.config import settings
from eval_harness.schema import GoldenItem


class GoldenSetError(ValueError):
    """Raised with every problem in the file at once, so you can fix them in one pass."""


def load_golden(name: str = "golden_v1.jsonl", known_docs: set[str] | None = None) -> list[GoldenItem]:
    """Load golden_sets/<name>. Pass known_docs to also check every source_doc exists in the corpus."""
    path = Path(name) if Path(name).is_absolute() else settings.golden_dir / name
    items: list[GoldenItem] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = GoldenItem.model_validate(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"line {line_no}: invalid JSON ({e.msg})")
            continue
        except ValidationError as e:
            errors.extend(f"line {line_no}: {err['msg']} ({'.'.join(map(str, err['loc'])) or 'item'})"
                          for err in e.errors())
            continue

        if item.id in seen_ids:
            errors.append(f"line {line_no}: duplicate id '{item.id}'")
        seen_ids.add(item.id)
        if known_docs is not None and item.source_doc is not None and item.source_doc not in known_docs:
            errors.append(f"line {line_no}: source_doc '{item.source_doc}' not in corpus {sorted(known_docs)}")
        items.append(item)

    if errors:
        raise GoldenSetError(f"{path.name} has {len(errors)} problem(s):\n  " + "\n  ".join(errors))
    if not items:
        raise GoldenSetError(f"{path.name} is empty")
    return items
