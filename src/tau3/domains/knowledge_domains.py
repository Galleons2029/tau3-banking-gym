"""Domains built on the banking knowledge environment stack.

`banking_synth` runs synthesized worlds through the same tools, retrieval and
policy as `banking_knowledge`, so every knowledge-specific branch in the runner,
the batch runner and the run config must treat the two identically — while still
reading each domain's own corpus.
"""

import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from tau3.domains.banking_knowledge.data_model import KnowledgeBase
    from tau3.environment.environment import Environment

KNOWLEDGE_DOMAINS = frozenset({"banking_knowledge", "banking_synth"})


def knowledge_base_for(domain: str) -> "KnowledgeBase":
    """Load the corpus a knowledge domain serves.

    Callers that warm caches or inline documents into a policy must go through
    this rather than importing `banking_knowledge` directly: the document cache
    is process-global, so warming it from the wrong corpus silently indexes the
    wrong documents.
    """
    if domain == "banking_synth":
        from tau3.domains.banking_synth.environment import get_knowledge_base
    elif domain == "banking_knowledge":
        from tau3.domains.banking_knowledge.environment import get_knowledge_base
    else:
        raise ValueError(f"Not a knowledge domain: {domain}")
    return get_knowledge_base()


def environment_factory_for(domain: str) -> Callable[..., "Environment"]:
    """Return a knowledge domain's environment constructor."""
    if domain == "banking_synth":
        from tau3.domains.banking_synth.environment import get_environment
    elif domain == "banking_knowledge":
        from tau3.domains.banking_knowledge.environment import get_environment
    else:
        raise ValueError(f"Not a knowledge domain: {domain}")
    return get_environment


def corpus_identity(domain: str) -> str:
    """Distinguish corpora a domain can serve across one process.

    `banking_knowledge` always serves the installed corpus, while `banking_synth`
    serves whichever bundle is configured, so anything caching an environment has
    to key on the world as well as the domain.
    """
    if domain == "banking_synth":
        return os.environ.get("TAU3_SYNTH_WORLD", "")
    return ""
