"""Small IO helpers shared across the perturbation pipeline."""

import hashlib
import json
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    """Hash canonical JSON independently of key insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: str | Path) -> Any:
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any) -> None:
    """Write a JSON artifact atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
