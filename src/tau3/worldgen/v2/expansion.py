"""Bounded automatic category proposal, admission and immutable world expansion."""

import json
from pathlib import Path

from tau3.worldgen.v2.diversity import diversity_report
from tau3.worldgen.v2.pipeline import build, check_certificate, write_json
from tau3.worldgen.v2.runtime import digest
from tau3.worldgen.v2.specs import CategorySpec, WorldSpec, load_categories


def expand(
    source: Path,
    destination: Path,
    packages: list[Path],
    require_novel: bool = True,
    readiness_path: Path | None = None,
    foundation_only: bool = False,
    verification_settings: Path | None = None,
    **build_kwargs,
) -> dict:
    """Admit supplied packages through the same full checks, leaving source intact."""
    if source.resolve() == destination.resolve():
        raise ValueError("Expansion requires a new immutable destination")
    check_certificate(source)
    if not foundation_only:
        from tau3.worldgen.v2.readiness import check_readiness

        if readiness_path is None:
            raise ValueError(
                "Expansion requires readiness evidence; foundation_only is explicit development scope"
            )
        check_readiness(source, readiness_path, verification_settings)
    baseline = WorldSpec.model_validate_json((source / "spec.json").read_text())
    spec = WorldSpec.model_validate(
        {
            **baseline.model_dump(),
            "categories": [*baseline.categories, *load_categories(packages)],
        }
    )
    before = diversity_report(baseline)
    after = diversity_report(spec)
    novel = set(after["families"]) - set(before["families"])
    if require_novel and not novel:
        raise ValueError(
            "No new workflow structure; renaming and parameter expansion do not establish novelty"
        )
    report = build(destination, spec, **build_kwargs)
    write_json(
        destination / "expansion.json",
        {
            "source_spec_hash": digest(baseline.model_dump()),
            "added_categories": [
                c.id for c in spec.categories[len(baseline.categories) :]
            ],
            "new_structures": sorted(novel),
            "validation": report["status"],
            "scope": "foundation_only" if foundation_only else "controlled_expansion",
        },
    )
    return report


def propose(
    source: Path,
    output: Path,
    concept: str,
    model: str,
    llm_args: dict | None = None,
    max_attempts: int = 3,
    review_models: list[str] | None = None,
    readiness_path: Path | None = None,
    foundation_only: bool = False,
    verification_settings: Path | None = None,
) -> dict:
    """Generate data-only candidates; quarantine invalid or structurally duplicate ones."""
    from tau3.data_model.message import SystemMessage, UserMessage
    from tau3.utils.llm_utils import extract_json_from_llm_response, generate

    if not 1 <= max_attempts <= 5:
        raise ValueError("Proposal attempts must be between one and five")
    check_certificate(source)
    if not foundation_only:
        from tau3.worldgen.v2.readiness import check_readiness

        if readiness_path is None:
            raise ValueError("Category proposal requires readiness evidence")
        check_readiness(source, readiness_path, verification_settings)
    baseline = WorldSpec.model_validate_json((source / "spec.json").read_text())
    review_models = review_models or []
    if len(set(review_models)) < 2 or not any(m != model for m in review_models):
        raise ValueError(
            "Novel proposals require two reviewer models including one other than the generator"
        )
    identity = digest(
        [digest(baseline.model_dump()), concept, model, llm_args or {}, review_models]
    )
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "proposal.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"identity": identity, "attempts": 0, "status": "INCONCLUSIVE"}
    )
    if progress["identity"] != identity:
        raise ValueError("Proposal resume input differs")
    if progress["status"] == "PASS":
        check_certificate(output / progress["world"])
        return progress
    while progress["attempts"] < max_attempts:
        progress["attempts"] += 1
        write_json(progress_path, progress)
        attempt = progress["attempts"]
        stage = "generation"
        try:
            response = generate(
                model=model,
                messages=[
                    SystemMessage(
                        role="system",
                        content=(
                            "Propose a coherent fictional banking category as JSON matching the schema. "
                            "Introduce new workflow/state structure, not just product renaming. "
                            "Every fact and rule must be explicit, financially plausible and executable. "
                            "Use only expressions with literals, dict field access, + - * /, comparisons, "
                            "and/or/not, conditional expressions, days(date,date), money(number), min/max. "
                            "Runtime namespaces: facts, args, record, user.joined_on, tables, clock. "
                            "Every table row and operation requires user_id and product_id. Effect values "
                            "are expressions. Scenarios need independent literal goals, exact inputs, "
                            "complete evidence and user requests with all required information. "
                            "Denial/grounding scenarios need explicit communication assertions. "
                            "Start with one product and one genuinely new scenario to stay within the output budget. "
                            "Never import code. Output only the category object."
                        ),
                    ),
                    UserMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "concept": concept,
                                "clock": baseline.clock,
                                "existing_categories": [
                                    {"id": c.id, "description": c.description}
                                    for c in baseline.categories
                                ],
                                "example": baseline.categories[0].model_dump(),
                                "schema": CategorySpec.model_json_schema(),
                                "previous_failure": progress.get("reason"),
                            }
                        ),
                    ),
                ],
                call_name="worldgen_v2_category_proposal",
                **(llm_args or {}),
            )
            write_json(
                output / f"response_{attempt}.json",
                {"model": model, "content": response.content},
            )
            raw = json.loads(extract_json_from_llm_response(response.content or ""))
            stage = "admission"
            package = CategorySpec.model_validate(raw)
            path = output / f"candidate_{attempt}.json"
            write_json(path, package.model_dump(mode="json"))
            world = f"candidate_world_{attempt}"
            report = expand(
                source,
                output / world,
                [path],
                readiness_path=readiness_path,
                foundation_only=foundation_only,
                verification_settings=verification_settings,
                require_novel=True,
                review_models=review_models,
                llm_args=llm_args,
                max_model_calls=2 * len(review_models),
            )
            if report["status"] != "PASS":
                details = [report.get("reason", ""), *report.get("errors", [])]
                details.extend(
                    error
                    for scenario in report.get("scenarios", [])
                    for error in scenario["errors"]
                )
                raise ValueError("; ".join(details)[:4000])
            progress.update(
                status="PASS",
                package=path.name,
                world=world,
                assurance="Executable synthetic contract; external real-bank fidelity is not established",
            )
            write_json(progress_path, progress)
            return progress
        except Exception as exc:
            # Do not dump transport exceptions, which may contain credentials.
            reason = (
                str(exc)[:4000]
                if stage == "admission" and isinstance(exc, ValueError)
                else type(exc).__name__
            )
            progress.update(status="INCONCLUSIVE", reason=reason)
            write_json(progress_path, progress)
    return progress
