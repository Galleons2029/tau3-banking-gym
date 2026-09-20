"""The plan manifest: the auditable artifact a variant is built from.

A plan is pure data.  ``materialize(plan, snapshot)`` is a deterministic
function of it, so the plan -- not the generated tree -- is what gets committed
and reviewed.  Every ``count`` is an assertion: if the materializer replaces a
different number of occurrences than the plan predicted, the build fails rather
than producing a variant with a half-applied rename.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

#: Bumped when the plan layout changes incompatibly.
SCHEMA_VERSION = 1


class FormMap(BaseModel):
    """One surface form and what it becomes, with its observed frequency."""

    model_config = {"extra": "forbid"}

    source: str = Field(description="The canonical spelling")
    target: str = Field(description="The variant spelling")
    style: Optional[str] = Field(default=None, description="Naming style that renders it")
    count: int = Field(description="Occurrences in the canonical corpus")


class ConceptMap(BaseModel):
    """A renamed concept and every spelling of it."""

    model_config = {"extra": "forbid"}

    concept_id: str
    kind: Literal["brand", "product", "tool", "identifier", "document"]
    root: str
    new_root: str
    slug: Optional[str] = Field(
        default=None, description="Document-id slug this concept owns, if any"
    )
    base: Optional[str] = Field(
        default=None,
        description="Concept this composes from; its new name is derived, not drawn",
    )
    detail: Optional[str] = None
    forms: list[FormMap] = Field(default_factory=list)

    @property
    def total(self) -> int:
        """Total occurrences across all spellings."""
        return sum(form.count for form in self.forms)


class Recipe(BaseModel):
    """Which perturbation axes are switched on."""

    model_config = {"extra": "forbid"}

    name: str = "names_only"
    brand: bool = True
    products: bool = True
    tools: bool = True
    identifiers: bool = True
    numerics: bool = False
    entity_ids: bool = False
    distractors: int = 0


class PerturbationPlan(BaseModel):
    """Everything needed to build one variant, and to audit it afterwards."""

    model_config = {"extra": "forbid"}

    schema_version: int = SCHEMA_VERSION
    variant_id: str
    domain_name: str
    package_name: str
    base_domain: str = "banking_knowledge"
    corpus_digest: str = Field(description="Snapshot the plan was built against")
    seed: int
    recipe: Recipe
    token_map: dict[str, str] = Field(
        default_factory=dict,
        description="Identity tokens renamed consistently across every product name",
    )
    concepts: list[ConceptMap] = Field(default_factory=list)
    frozen_tools: list[str] = Field(default_factory=list)
    kept_tokens: list[str] = Field(
        default_factory=list,
        description="Structural and generic words deliberately left alone",
    )
    expected: dict = Field(default_factory=dict)
    plan_digest: str = ""

    def concept(self, concept_id: str) -> ConceptMap:
        """Look up one concept by id."""
        for candidate in self.concepts:
            if candidate.concept_id == concept_id:
                return candidate
        raise KeyError(concept_id)

    def replacement_pairs(self) -> list[tuple[str, str]]:
        """Every ``(source, target)`` spelling pair, for the rewriter."""
        return [
            (form.source, form.target)
            for concept in self.concepts
            for form in concept.forms
        ]

    def expected_counts(self) -> dict[str, int]:
        """Planned occurrence count per source form."""
        return {
            form.source: form.count
            for concept in self.concepts
            for form in concept.forms
        }
