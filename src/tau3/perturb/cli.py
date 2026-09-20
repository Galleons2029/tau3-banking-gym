"""``tau3 perturb`` -- build and check variant domains."""

import argparse
from pathlib import Path

from tau3.perturb.storage import write_json


def add_perturb_parser(subparsers) -> None:
    """Register the ``perturb`` command.

    Imports stay lazy so registering the parser costs nothing.
    """
    parser = subparsers.add_parser(
        "perturb",
        help="Build re-skinned variants of the banking domain",
        description=(
            "Generate variant domains with different company, product, tool and "
            "document names so training and evaluation can use disjoint symbol "
            "sets. The canonical domain is never modified."
        ),
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    snapshot = stages.add_parser("snapshot", help="Freeze the canonical corpus")
    snapshot.add_argument("--output", type=Path, default=None)
    snapshot.add_argument("--force", action="store_true")

    plan = stages.add_parser("plan", help="Assign variant names and write a plan")
    plan.add_argument("--seed", type=int, required=True)
    plan.add_argument("--snapshot", type=Path, default=None)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument(
        "--exclude-plan",
        type=Path,
        action="append",
        default=[],
        help="Plan whose names this variant must not reuse; repeatable",
    )

    build = stages.add_parser("materialize", help="Build a variant from a plan")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--snapshot", type=Path, default=None)
    build.add_argument("--output", type=Path, default=None)
    build.add_argument("--force", action="store_true")

    check = stages.add_parser("verify", help="Check a materialized variant")
    check.add_argument("--variant", type=Path, required=True)
    check.add_argument("--snapshot", type=Path, default=None)
    check.add_argument("--skip-slow", action="store_true")
    check.add_argument(
        "--other",
        type=Path,
        action="append",
        default=[],
        help="Sibling variant to check symbol disjointness against; repeatable",
    )

    listing = stages.add_parser("list", help="List materialized variants")
    listing.add_argument("--root", type=Path, default=None)

    parser.set_defaults(func=run_perturb)


def _snapshot_root(explicit) -> Path:
    from tau3.perturb.workspace import variants_root

    return Path(explicit) if explicit else variants_root() / "_snapshots" / "canonical"


def run_perturb(args) -> int:
    """Dispatch a ``tau3 perturb`` stage."""
    from tau3.perturb import materialize as materializer
    from tau3.perturb import plan as planner
    from tau3.perturb import snapshot as snapshotter
    from tau3.perturb import verify as verifier
    from tau3.perturb.models import PerturbationPlan
    from tau3.perturb.storage import read_json
    from tau3.perturb.workspace import assert_outside_domain, variants_root

    if args.stage == "snapshot":
        root = _snapshot_root(args.output)
        assert_outside_domain(root)
        snap = snapshotter.create(root, force=args.force)
        print(f"snapshot {snap.digest[:12]} -> {root} ({snap.file_count} files)")
        return 0

    if args.stage == "plan":
        excluded: list[str] = []
        for path in args.exclude_plan:
            other = PerturbationPlan.model_validate(read_json(path))
            excluded.extend(
                form.target for concept in other.concepts for form in concept.forms
            )
            excluded.extend(other.token_map.values())
        built = planner.build_plan(
            args.seed, snapshot_root=_snapshot_root(args.snapshot), exclude=excluded
        )
        write_json(args.output, built.model_dump())
        print(
            f"plan {built.variant_id} -> {args.output} "
            f"({len(built.concepts)} concepts, "
            f"{built.expected['total_replacements']} replacements)"
        )
        return 0

    if args.stage == "materialize":
        built = PerturbationPlan.model_validate(read_json(args.plan))
        destination = Path(args.output) if args.output else variants_root() / built.variant_id
        result = materializer.materialize(
            built,
            snapshot_root=_snapshot_root(args.snapshot),
            destination=destination,
            force=args.force,
        )
        print(
            f"variant {built.variant_id} -> {result.root} "
            f"({result.file_count} files, {result.replacements} replacements)"
        )
        return 0

    if args.stage == "verify":
        others = [
            verifier.load_plan(path) for path in args.other
        ]
        report = verifier.verify(
            args.variant,
            snapshot_root=_snapshot_root(args.snapshot),
            others=others,
            skip_slow=args.skip_slow,
        )
        write_json(Path(args.variant) / "report.json", report.to_dict())
        for stage, outcome in report.stages.items():
            print(f"  {stage:22s} {outcome}")
        for finding in report.findings[:40]:
            print(f"    - {finding}")
        if not report.ok:
            # main() calls args.func(args) and ignores its return value, so a
            # failed verification has to raise to reach the shell as non-zero.
            raise SystemExit(f"FAILED: {len(report.findings)} finding(s)")
        print("OK")
        return 0

    if args.stage == "list":
        root = Path(args.root) if args.root else variants_root()
        for path in sorted(root.glob("*/plan.json")):
            built = PerturbationPlan.model_validate(read_json(path))
            print(f"{built.variant_id}  {built.domain_name}  seed={built.seed}")
        return 0

    raise ValueError(f"Unknown stage: {args.stage}")
