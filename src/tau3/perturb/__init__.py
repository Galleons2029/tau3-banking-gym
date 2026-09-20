"""Deterministic re-skinning of the banking domain into disjoint variants.

Task synthesis runs against a single official corpus, so a trained model can
score well by memorising surface forms -- the brand, the product names, the
tool names, the document ids -- instead of retrieving and reasoning.  This
package materialises *variant domains* from the canonical one: same business
structure, different symbols.  Training on one set of variants and evaluating
on a held-out variant turns that memorisation into a measurable score gap.

Nothing here mutates the canonical domain.  Variants are build artifacts,
reproducible from a committed plan manifest.
"""
