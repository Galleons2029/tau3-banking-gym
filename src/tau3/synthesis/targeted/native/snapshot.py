"""Freeze source data and actual runtime bytes, including uncommitted changes."""

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from tau3.synthesis.storage import digest, read_json, retrieval_settings, write_json


@lru_cache(maxsize=1)
def sources():
    """Keep official task/gold files and credentials outside the synthesis snapshot."""
    from tau3.domains.banking_knowledge.utils import KNOWLEDGE_DATA_DIR

    tau = Path(__file__).resolve().parents[3]
    data = Path(KNOWLEDGE_DATA_DIR)
    paths = [("data/" + str(p.relative_to(data)), p) for p in sorted((data / "documents").glob("*.json"))]
    paths += [("data/" + str(p.relative_to(data)), p) for p in sorted((data / "prompts").rglob("*.md"))]
    paths.append(("data/db.json", data / "db.json"))
    paths += [("runtime/" + str(p.relative_to(tau)), p) for p in sorted(tau.rglob("*.py"))]
    return {key: path.read_text() for key, path in paths}


def snapshot(root):
    """Refuse code or corpus drift; a new implementation requires a new round."""
    files = sources()
    hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in files.items()}
    from tau3.domains.banking_knowledge.environment import get_environment

    env = get_environment(retrieval_variant="bm25_grep", retrieval_kwargs={"top_k": 10})
    manifest = {"schema_version": 2, "hashes": hashes,
                "db_snapshot_hash": hashes["data/db.json"],
                "public_tool_schema_hash": digest([t.openai_schema for t in env.get_tools()]),
                "policy_hash": digest(env.get_policy()), "retrieval": retrieval_settings()}
    manifest["snapshot_hash"] = digest(manifest)
    path = root / "snapshot" / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Native runtime or public corpus changed; use a new round")
        with gzip.open(path.parent / "sources.json.gz", "rt", encoding="utf-8") as handle:
            if json.load(handle) != files:
                raise ValueError("Native source archive changed")
        return manifest
    archive = path.parent / "sources.json.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive.with_suffix(".tmp"), "wt", encoding="utf-8") as handle:
        json.dump(files, handle, ensure_ascii=False)
    archive.with_suffix(".tmp").replace(archive)
    write_json(path, manifest)
    return manifest
