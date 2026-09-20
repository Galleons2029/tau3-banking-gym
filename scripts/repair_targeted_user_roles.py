"""Budgeted prompt A/B experiment and immutable, whole-row SFT role filtering."""

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.budget import Budget
from tau3.synthesis.targeted.native.models import NativeConfig
from tau3.synthesis.targeted.native.user_roles import (
    review_customer_roles,
    role_context,
    role_disposition,
)
from tau3.user.role_guard import CUSTOMER_ROLE_INSTRUCTION, guard_customer_prompt
from tau3.utils import llm_utils
from tau3.utils.llm_concurrency import file_lock


def file_hash(path):
    """Hash an artifact without interpreting or rewriting its contents."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_bound(path, identity, work):
    """Resume completed evidence, retaining unresolved calls instead of hiding them."""
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            saved = read_json(path)
            if saved["identity"] != identity:
                raise ValueError("Role repair evidence identity changed")
            return saved
        try:
            result = {"status": "COMPLETE", **work()}
        except Exception as exc:
            result = {"status": "INCONCLUSIVE", "error": type(exc).__name__,
                      "reason": str(exc)[:300]}
        saved = {"identity": identity, **result}
        write_json(path, saved)
        return saved


def parallel(jobs, work, progress, workers=256):
    """Use the round's shared physical pool; publish logical progress periodically."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0 or len(results) == len(jobs):
                write_json(progress, {"completed": len(results), "total": len(jobs),
                                     "updated_at": time.time()})
    return results


def prepare_probes(root, source, output):
    """Freeze cases before new responses: suspicious, ordinary and user-tool turns."""
    path = output / "probe-plan.json"
    if path.exists():
        return read_json(path)["cases"]
    audit = read_json(source / "verification/user-request-audit/formal-request-boundaries.json")
    candidates = read_json(source / "verification/formal-user-role-screen.json")["candidates"]
    suspect = {c["line"] for c in candidates}
    cases = []
    def append(entry, group):
        cases.append({"physical_path": entry["physical_path"], "group": group,
                      "physical_sha256": file_hash(Path(entry["physical_path"]))})
    for row in [r for r in audit["rows"] if r["sft_line"] in suspect][:28]:
        append(row["requests"][0], "historical_screen_candidate")
    for identifier in ("748a89fddf3d4281ba0da39dec4b5455", "bc9287b8f7cd4d448722f7fd3d80dbbe",
                       "3fa40462f22f4df387b9eb4fbc102dfc", "7f36b2cfc0da4c2c907772539c3d5213"):
        append({"physical_path": str(root / "physical_calls" / f"{identifier}.json")}, "new_confirmed_case")
    healthy = [r for r in audit["rows"] if r["sft_line"] not in suspect]
    for row in healthy[:8]:
        append(row["requests"][0], "ordinary_first_turn")
    selected = 0
    for row in healthy:
        entry = next((r for r in row["requests"] if r["api_roles"][-1] == "tool"), None)
        if entry:
            append(entry, "customer_tool_result")
            selected += 1
            if selected == 8:
                break
    write_json(path, {"cases": cases, "instruction": CUSTOMER_ROLE_INSTRUCTION,
                     "single_changed_variable": "Append instruction to system content only",
                     "temperature": 0, "fixed_source_seed": True,
                     "budget_root": str(root), "no_teacher_slots": True})
    return cases


def probes(root, source, output, config):
    """Run all frozen conditions and independently classify their customer output."""
    cases = prepare_probes(root, source, output)
    jobs = [(i, case, variant) for i, case in enumerate(cases) for variant in ("baseline", "guarded")]
    def generate(job):
        i, case, variant = job
        path = Path(case["physical_path"])
        if file_hash(path) != case["physical_sha256"]:
            raise ValueError("Original request changed")
        request = copy.deepcopy(read_json(path)["request"])
        if variant == "guarded":
            request["messages"][0]["content"] = guard_customer_prompt(request["messages"][0]["content"])
        identity = digest(request)
        def call():
            response = Budget(root, config).call("customer_role_ab_" + variant,
                llm_utils.completion, **{**request, "api_key": config.settings.llm_args.get("api_key", "not-required")})
            return {"response": response.model_dump(mode="json")}
        result = save_bound(output / "probes" / f"{i:03d}-{variant}.json", identity, call)
        return {"case": i, "group": case["group"], "variant": variant, **result}
    generated = parallel(jobs, generate, output / "probe-progress.json", 128)
    review_jobs = [(item, model) for item in generated
                   for model in (config.teacher_model, config.reviewer_model)]
    def review(job):
        item, model = job
        response = item.get("response") or {}
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        context = [{"speaker": "CUSTOMER", "customer_index": 0, "content": message.get("content") or ""}]
        original = read_json(Path(cases[item["case"]]["physical_path"]))["request"]
        scope = {"kind": "prompt_ab", "original_customer_system": original["messages"][0]["content"],
                 "prior_api_conversation": original["messages"][1:], "output_tool_calls": message.get("tool_calls")}
        def judge():
            if item["status"] != "COMPLETE" or choice.get("finish_reason") != "stop" or not context[0]["content"].strip():
                return {"status": "INCONCLUSIVE", "reason": "No complete nonempty customer text; tool-only responses recorded separately"}
            return review_customer_roles(root, config, context, model, scope=scope)
        saved = save_bound(output / "probe-reviews" / f"{item['case']:03d}-{item['variant']}-{digest(model)[:8]}.json",
                           digest([item, model]), judge)
        return {"case": item["case"], "variant": item["variant"], "model": model, **saved}
    reviews = parallel(review_jobs, review, output / "probe-review-progress.json")
    counts, rows = {}, []
    for item in sorted(generated, key=lambda r: (r["case"], r["variant"])):
        decisions = [r for r in reviews if r["case"] == item["case"] and r["variant"] == item["variant"]]
        status = role_disposition(decisions)
        counts.setdefault(item["variant"], Counter())[status] += 1
        rows.append({**item, "disposition": status, "reviews": decisions})
    write_json(output / "probe-results.json", {"counts": counts, "cases": len(cases), "results": rows,
               "limitations": "Enriched diagnostic cases, not population rate; repeated seed at temperature zero need not reproduce a prior response. No teacher replacement or SFT output from probes."})


def reaudit(root, source, output, config):
    """Review every source trajectory, preserving originals and exact retained rows."""
    source_file = source / "small/sft.jsonl"
    source_hash = file_hash(source_file)
    rows = [json.loads(line) for line in source_file.read_text().splitlines()]
    jobs = []
    for line, row in enumerate(rows, 1):
        directory = Path(row["metadata"]["evidence"])
        capture = read_json(directory / "capture.json")
        context = role_context(capture["simulation"]["messages"])
        task = read_json(directory.parent.parent / "task.json")
        for model in (config.teacher_model, config.reviewer_model):
            jobs.append((line, row, context, task["user_scenario"], model))
    def review(job):
        line, row, context, scenario, model = job
        identity = digest([source_hash, line, context, scenario, model])
        def call():
            return review_customer_roles(root, config, context, model,
                scope={"source_sha256": source_hash, "line": line, "customer_scenario": scenario})
        saved = save_bound(output / "row-reviews" / f"{line:04d}-{digest(model)[:8]}.json", identity, call)
        return {"line": line, "model": model, **saved}
    reviews = parallel(jobs, review, output / "reaudit-progress.json")
    by_line = {}
    for review in reviews:
        by_line.setdefault(review["line"], []).append(review)
    human_confirmed = {5, 30, 43}
    mapping, totals = [], Counter()
    package = output / "filtered"
    package.mkdir(parents=True, exist_ok=True)
    paths = {key: package / name for key, name in
             (("KEEP", "sft.jsonl"), ("DISCARD", "discarded.jsonl"), ("QUARANTINE", "quarantine.jsonl"))}
    handles = {k: p.with_suffix(".tmp").open("wb") for k, p in paths.items()}
    try:
        with source_file.open("rb") as original:
            for line, raw in enumerate(original, 1):
                verdict = role_disposition(by_line.get(line, []))
                if line in human_confirmed:
                    verdict = "DISCARD"
                totals[verdict] += 1
                handles[verdict].write(raw)
                mapping.append({"source_line": line, "disposition": verdict,
                    "source_line_sha256": hashlib.sha256(raw).hexdigest(),
                    "output_line": totals[verdict], "task_id": rows[line-1]["task_id"],
                    "seed": rows[line-1]["seed"], "human_confirmed": line in human_confirmed,
                    "reviews": by_line.get(line, [])})
    finally:
        for handle in handles.values():
            handle.close()
    if file_hash(source_file) != source_hash:
        raise ValueError("Source changed during re-audit")
    for path in paths.values():
        path.with_suffix(".tmp").replace(path)
    write_json(package / "sampling_index.json", mapping)
    manifest = {"status": "COMPLETE", "scope": "Customer role filtering; other source quality claims unchanged",
        "source": str(source_file.resolve()), "source_sha256": source_hash,
        "source_rows": len(rows), "counts": totals, "retained_rows_are_byte_identical": True,
        "rows_rewritten": 0, "no_seed_replacement": True, "budget_root": str(root),
        "files": {k: {"path": str(p.resolve()), "sha256": file_hash(p)} for k, p in paths.items()},
        "mapping_sha256": file_hash(package / "sampling_index.json"),
        "upstream_status": "Original main export and its upsampled derivatives are superseded for training by this filtered package; originals retained for evidence only"}
    write_json(package / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def main():
    """Run diagnostics or re-audit under an existing cumulative round budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probes", "reaudit"])
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--source-round", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = NativeConfig.model_validate(read_json(args.round_dir / "config.json"))
    with file_lock(args.output / (args.mode + ".lock")):
        (probes if args.mode == "probes" else reaudit)(args.round_dir.resolve(), args.source_round.resolve(), args.output.resolve(), config)


if __name__ == "__main__":
    main()
