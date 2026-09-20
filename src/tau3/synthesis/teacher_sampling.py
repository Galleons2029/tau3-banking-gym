"""Independent teacher selection with immutable historical trajectory reuse."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from tau3.synthesis.stream_resume import check_trial
from tau3.synthesis.world_sft import trial_sample
from tau3.worldgen.v2.audit import AuditIncomplete
from tau3.worldgen.v2.pipeline import write_json
from tau3.worldgen.v2.readiness import evidence_files
from tau3.worldgen.v2.runtime import digest


class TeacherPolicy(BaseModel):
    """Select one teacher and two distinct trial seeds, independently of verifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    seeds: tuple[int, int] = (42, 43)

    @model_validator(mode="after")
    def distinct_seeds(self):
        """Reject empty model names and duplicate sampling identities."""
        if not self.model.strip() or len(set(self.seeds)) != 2:
            raise ValueError("A teacher and two distinct seeds are required")
        return self


def read(path):
    """Read an immutable capture or provenance record."""
    return json.loads(path.read_text())


def compatible_settings(settings):
    """Recognize the previously replayed schema with identical default actor args."""
    current = settings.model_dump()
    hashes = {digest(current)}
    old = dict(current)
    if all(
        old.get(k) == v
        for k, v in (
            ("agent_llm_args", {}),
            ("user_llm_args", {}),
            ("capture_retrieval", "bm25"),
        )
    ):
        for key in ("agent_llm_args", "user_llm_args", "capture_retrieval"):
            old.pop(key, None)
        hashes.add(digest(old))
    return hashes


def resolve_trial(directory):
    """Resolve one pinned reuse pointer; changes to historical evidence fail closed."""
    pointer = directory / "reuse.json"
    if not pointer.exists():
        return directory
    receipt = read(pointer)
    source = Path(receipt["source"])
    if evidence_files(source) != receipt["files"]:
        raise AuditIncomplete("Reused teacher evidence changed")
    return source


def run_teacher_trial(
    task, settings, model, directory, seed, binding, world, sources, capture
):
    """Reuse the same model/seed or capture once; never resample a failed GLM trial."""
    request = {
        "task": digest(task.model_dump(mode="json")),
        "model": model,
        "seed": seed,
        "settings": digest(settings.model_dump()),
        "binding": binding,
    }
    pointer = directory / "reuse.json"
    if pointer.exists():
        if read(pointer)["request"] != request:
            raise AuditIncomplete("Teacher reuse identity changed")
        source = resolve_trial(directory)
        return check_trial(source, task, world, settings)
    if (directory / "started.json").exists():
        return capture(task, settings, model, directory, seed, binding)
    for candidate in sources:
        source = resolve_trial(candidate)
        marker = source / "started.json"
        if not marker.exists():
            continue
        identity = read(marker)
        if any(identity.get(k) != request[k] for k in ("task", "model", "seed")):
            continue
        if identity.get("settings") not in compatible_settings(settings):
            raise AuditIncomplete("Historical teacher settings are incompatible")
        if not (source / "result.json").exists():
            raise AuditIncomplete("Previous interrupted GLM trial retained")
        # Strict replay verifies tool outputs and retains negative model grades.
        outcome = check_trial(source, task, world, settings)
        write_json(
            pointer,
            {
                "source": str(source.resolve()),
                "files": evidence_files(source),
                "request": request,
            },
        )
        return outcome
    return capture(task, settings, model, directory, seed, binding)


def teacher_sample(directory, model, seed):
    """Export only correctly bound samples from the requested teacher and seed."""
    source = resolve_trial(directory)
    marker = read(source / "started.json")
    sample = trial_sample(source)
    if (
        marker.get("model") != model
        or marker.get("seed") != seed
        or sample.get("teacher_model") != model
        or sample.get("seed") != seed
    ):
        raise AuditIncomplete("SFT teacher/model/seed binding changed")
    return sample


def historical_rejection(directories):
    """Retain a real rejection from any prior generation of the same attempt."""
    for directory in directories:
        path = directory / "rejected.json"
        if not path.exists():
            continue
        record = read(path)
        if record["reason"] != "Stale validation certificate":
            return {**record, "origin": str(directory)}
    return None
