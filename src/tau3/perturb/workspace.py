"""Where generated artifacts live.

Variants are build outputs, never checked in and never written anywhere near
the canonical domain -- ``environment_fingerprint()`` rglobs the domain's
prompts directory, so a variant nested under it would silently change the
canonical hash and invalidate every published bundle.
"""

import os
from pathlib import Path

#: Environment override for the variant workspace.
VARIANTS_DIR_ENV = "TAU3_VARIANTS_DIR"


def variants_root() -> Path:
    """Root directory for snapshots and materialized variants."""
    override = os.getenv(VARIANTS_DIR_ENV)
    if override:
        return Path(override)
    from tau3.utils.utils import DATA_DIR

    return Path(DATA_DIR) / "variants"


def assert_outside_domain(path: Path) -> None:
    """Fail if ``path`` would place generated files inside the canonical domain.

    Raises:
        ValueError: If the path is inside the canonical data directory.
    """
    from tau3.perturb.scan import domain_code_dir, domain_data_dir

    resolved = Path(path).resolve()
    for forbidden in (domain_data_dir().resolve(), domain_code_dir().resolve()):
        if resolved == forbidden or forbidden in resolved.parents:
            raise ValueError(
                f"Refusing to write generated files inside the canonical domain: {resolved}"
            )
