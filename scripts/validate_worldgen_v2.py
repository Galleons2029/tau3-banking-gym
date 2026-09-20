"""Build repeatable scale fixtures and report correctness separately from novelty."""

import argparse
import time
from pathlib import Path

from tau3.worldgen.v2.catalog import PROFILES, scaffold
from tau3.worldgen.v2.pipeline import build, publish, write_json
from tau3.worldgen.v2.specs import WorldSpec


def main() -> None:
    """Run isolated 4/12/24-category checks without model or network dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/synthetic/worldgen-v2-final-scale")
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 12, 24])
    args = parser.parse_args()
    if any(size < 1 or size > 100 for size in args.sizes):
        parser.error("sizes must be between one and 100")
    profiles = list(PROFILES)
    reports = []
    for size in args.sizes:
        started = time.monotonic()
        spec = WorldSpec(
            categories=[
                scaffold(f"category_{i:03d}", profiles[i % len(profiles)], i)
                for i in range(size)
            ]
        )
        root = args.output / f"categories_{size:03d}"
        report = build(root, spec)
        row = {
            "size": size,
            "status": report["status"],
            "seconds": round(time.monotonic() - started, 3),
            "counts": report.get("counts"),
            "diversity": report.get("diversity"),
            "counterexamples": sum(
                len(s["counterexamples"]) for s in report.get("scenarios", [])
            ),
            "assurance": "Offline contract checks; these sizes do not establish novel category semantics or paper equivalence",
        }
        reports.append(row)
        write_json(args.output / "summary.json", reports)
        print({k: v for k, v in row.items() if k != "diversity"}, flush=True)
        if report["status"] != "PASS":
            raise SystemExit(2)
        publish(root)


if __name__ == "__main__":
    main()
