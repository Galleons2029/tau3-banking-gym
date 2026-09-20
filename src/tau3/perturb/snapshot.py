"""Freeze the canonical corpus into an immutable, content-addressed archive.

Everything downstream -- scanning, planning, materializing, verifying -- reads
a snapshot rather than the live domain directory.  Three reasons:

1. **The originals stay untouched.**  Nothing in this package opens a canonical
   file for writing, and reading through a snapshot makes that structural
   rather than a promise.
2. **Consistency under concurrent use.**  The canonical corpus is shared with
   other running jobs; a snapshot is one stable view instead of 800-odd reads
   that could straddle someone else's write.
3. **Reproducibility.**  A plan records the ``corpus_digest`` it was built
   against, so a variant can be regenerated exactly, or detected as stale.

The snapshot is a single gzipped JSON archive, not a directory tree.  The
corpus is 3.7 MB of text in 824 files; on this filesystem that tree costs
~400 MB of blocks and ~2.5 minutes of per-file round trips, while the archive
is about a megabyte and loads in well under a second.
"""

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: Relative globs copied out of the domain's data directory.
DATA_INCLUDES: tuple[tuple[str, str], ...] = (
    ("documents", "*.json"),
    ("tasks", "task_*.json"),
    ("prompts", "**/*.md"),
)

#: Individual data files copied verbatim.
DATA_FILES: tuple[str, ...] = ("db.json", "tasks.json")

#: Globs copied out of the domain's Python package.
CODE_INCLUDES: tuple[str, ...] = ("*.py", "*.md")

#: Archive filename inside a snapshot directory.
ARCHIVE = "corpus.json.gz"


@dataclass(frozen=True)
class Snapshot:
    """A frozen copy of the canonical corpus, held in memory."""

    root: Path
    digest: str
    files: dict[str, str] = field(repr=False, default_factory=dict)
    hashes: dict[str, str] = field(repr=False, default_factory=dict)

    @property
    def file_count(self) -> int:
        """Number of captured files."""
        return len(self.files)


def _sources() -> list[tuple[str, Path]]:
    """Enumerate ``(key, path)`` for every canonical file, sorted by key."""
    from tau3.perturb.scan import domain_code_dir, domain_data_dir

    data, code = domain_data_dir(), domain_code_dir()
    found: list[tuple[str, Path]] = []
    for subdir, pattern in DATA_INCLUDES:
        base = data / subdir
        for path in base.glob(pattern):
            found.append((f"data/{subdir}/{path.relative_to(base).as_posix()}", path))
    for name in DATA_FILES:
        if (data / name).exists():
            found.append((f"data/{name}", data / name))
    for pattern in CODE_INCLUDES:
        for path in code.glob(pattern):
            found.append((f"code/{path.name}", path))
    return sorted(found)


def _digest_of(hashes: dict[str, str]) -> str:
    return hashlib.sha256(
        "\n".join(f"{key}:{hashes[key]}" for key in sorted(hashes)).encode()
    ).hexdigest()


def create(destination: Path, *, force: bool = False) -> Snapshot:
    """Capture the canonical corpus into ``destination``.

    Args:
        destination: Directory to hold the snapshot archive.
        force: Recapture even if a snapshot is already present.

    Returns:
        The captured :class:`Snapshot`.
    """
    destination = Path(destination)
    archive = destination / ARCHIVE
    if archive.exists() and not force:
        return load(destination)

    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for key, path in _sources():
        raw = path.read_bytes()
        files[key] = raw.decode()
        hashes[key] = hashlib.sha256(raw).hexdigest()

    digest = _digest_of(hashes)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "corpus_digest": digest,
        "hashes": hashes,
        "files": files,
    }
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    temporary.replace(archive)
    return Snapshot(root=destination, digest=digest, files=files, hashes=hashes)


def load(root: Path) -> Snapshot:
    """Load a snapshot archive.

    Raises:
        ValueError: If the archive schema is unsupported or its contents no
            longer hash to the recorded digest.
    """
    root = Path(root)
    with gzip.open(root / ARCHIVE, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported snapshot schema: {payload.get('schema_version')}")

    files, hashes = payload["files"], payload["hashes"]
    for key, expected in hashes.items():
        actual = hashlib.sha256(files[key].encode()).hexdigest()
        if actual != expected:
            raise ValueError(f"Snapshot content does not match its hash: {key}")
    digest = _digest_of(hashes)
    if digest != payload["corpus_digest"]:
        raise ValueError("Snapshot digest does not match its file hashes")
    return Snapshot(root=root, digest=digest, files=files, hashes=hashes)


def default_root(base: Optional[Path] = None) -> Path:
    """Where snapshots live by default: alongside variants, never in the domain."""
    from tau3.perturb.workspace import variants_root

    return (Path(base) if base else variants_root()) / "_snapshots"
