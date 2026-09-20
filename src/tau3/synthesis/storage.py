"""Atomic checkpoints and a fingerprint of the fixed benchmark environment."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    """Hash canonical JSON independently of insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: str | Path) -> Any:
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any) -> None:
    """Replace one artifact atomically; callers own distinct paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


@lru_cache(maxsize=1)
def environment_fingerprint() -> str:
    """Fingerprint KB, baseline DB, tools, retrieval, user and scoring code."""
    from tau3.domains.banking_knowledge.utils import KNOWLEDGE_DATA_DIR

    root = Path(__file__).resolve().parents[1]
    files = list((KNOWLEDGE_DATA_DIR / "documents").glob("*.json"))
    files += list((KNOWLEDGE_DATA_DIR / "prompts").rglob("*.md"))
    files += [KNOWLEDGE_DATA_DIR / "db.json"]
    for directory in (
        "domains/banking_knowledge",
        "knowledge",
        "environment",
        "evaluator",
        "user",
        "orchestrator",
        "data_model",
        "agent",
    ):
        files += list((root / directory).rglob("*.py"))
    sources = {}
    for path in sorted(files):
        key = (
            "data/" + str(path.relative_to(KNOWLEDGE_DATA_DIR))
            if path.is_relative_to(KNOWLEDGE_DATA_DIR)
            else "src/" + str(path.relative_to(root))
        )
        sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(sources)


def retrieval_settings() -> dict:
    """Record actual resolved retrieval specifications and prompt checksum."""
    from dataclasses import asdict

    from tau3.domains.banking_knowledge.retrieval import resolve_variant

    variant = resolve_variant("bm25_grep", top_k=10)
    return {
        "name": variant.name,
        "kwargs": {"top_k": 10},
        "kb_search": asdict(variant.kb_search),
        "grep": asdict(variant.grep),
        "prompt_sha256": hashlib.sha256(
            variant.prompt_template.read_bytes()
        ).hexdigest(),
    }


def generator_fingerprint() -> str:
    """Record the generator implementation, including its narrative/review prompts."""
    directory = Path(__file__).parent
    return digest(
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.py"))
        }
    )
