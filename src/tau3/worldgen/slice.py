"""The slice definition: which categories, products and scenario families exist.

Product names are drawn from a palette deliberately disjoint from the official
corpus. Sharing a name would let an agent answer from memory of the benchmark it
was trained on, and would also make the two corpora interfere if they are ever
indexed together. L10 enforces the disjointness rather than trusting this list.
"""

from dataclasses import dataclass, field
from typing import Literal

BANK = "Rho-Bank"

# The tiers a catalogue is layered into, most modest first. A tier is something
# a customer can say out loud without naming a product -- "I am looking at your
# everyday accounts" -- which is what makes it usable as the scope of a task.
# The official corpus layers its own catalogue the same way, in metals and
# colours; these labels deliberately share none of its words, for the same
# reason the product palette does not.
SEGMENTS = ["Everyday", "Signature", "Keystone", "Landmark"]
# Fewest products a tier may hold: an answer and two near misses, plus one more
# so a tier is a range rather than a shortlist.
MIN_SEGMENT = 4

# Variables every product in a category must define, because the scenario
# families turn on them. A generator may add more; it may not omit these, and it
# does not get to decide their type, unit or visibility -- the solver and the
# families depend on those, and a product's own class name being marked internal
# would keep it out of the customer-facing documents entirely.
REQUIRED_VARIABLES: dict[str, dict[str, tuple[str, str | None]]] = {
    "checking": {
        "account_class": ("string", None),
        "service_tier": ("string", None),
        "monthly_fee": ("currency", "USD"),
        "mobile_deposit_limit": ("currency", "USD"),
        "early_direct_deposit_days": ("int_days", "days"),
        "minimum_balance": ("currency", "USD"),
        "overdraft_fee": ("currency", "USD"),
    },
    "savings": {
        "account_class": ("string", None),
        "service_tier": ("string", None),
        "base_apy": ("percent", "percent"),
        "promotional_apy_boost": ("percent", "percent"),
        "minimum_balance": ("currency", "USD"),
        "minimum_opening_deposit": ("currency", "USD"),
        "monthly_fee": ("currency", "USD"),
        "withdrawal_allowance": ("int_count", "withdrawals"),
    },
    "business_checking": {
        "account_class": ("string", None),
        "service_tier": ("string", None),
        "monthly_fee": ("currency", "USD"),
        "minimum_balance": ("currency", "USD"),
        "transaction_allowance": ("int_count", "transactions"),
        "cash_deposit_allowance": ("currency", "USD"),
        "wire_transfer_fee": ("currency", "USD"),
    },
    "card": {
        "card_class": ("string", None),
        "service_tier": ("string", None),
        "base_cashback_rate": ("percent", "percent"),
        "promotional_cashback_rate": ("percent", "percent"),
        "annual_fee": ("currency", "USD"),
        "purchase_apr": ("percent", "percent"),
        "foreign_transaction_fee": ("percent", "percent"),
        "late_payment_fee": ("currency", "USD"),
    },
}


@dataclass(frozen=True)
class ProductSpec:
    """One product in the slice."""

    feature_id: str
    entity_name: str
    category_id: str
    kind: Literal["checking", "savings", "card", "business_checking"]
    # Which tier of the catalogue this product sits in. A customer can say which
    # range they are shopping in without naming a product, which is what lets
    # several scenario families of the same kind coexist.
    segment: str = SEGMENTS[0]

    @property
    def class_name(self) -> str:
        """The exact string the opening tools expect for this product."""
        suffix = "Card" if self.kind == "card" else "Account"
        return f"{self.entity_name} {suffix}"


@dataclass(frozen=True)
class CategorySpec:
    """One knowledge category."""

    category_id: str
    display_name: str
    kind: Literal["product", "protocol", "program", "support"]
    audience: Literal["customer", "internal", "both"]


@dataclass(frozen=True)
class FamilySpec:
    """A scenario family and the trap it carries.

    The solver assigns values so that each family has exactly one answer and
    every rejected candidate fails exactly one stated constraint.
    """

    family_id: str
    kind: Literal["selection", "promotional", "temporal"]
    # Everything the customer's requirements range over. A customer who does not
    # name a shortlist is comparing the whole catalogue, so the answer has to be
    # unique across all of it, not just against two chosen rivals.
    candidates: list[str]
    answer: str
    # The two products deliberately placed one constraint away from qualifying.
    # Everything else in `candidates` is pushed clearly out of contention.
    near_misses: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    description: str = ""
    # The tier the customer states. Families of the same product kind are
    # disjoint across tiers, so each has its own answer, its own near misses and
    # its own evidence instead of overwriting one another's values.
    segment: str = SEGMENTS[0]


# Distinctive names, none of them official corpus terms. A product is one of
# these paired with its category's noun, which is how the official catalogue
# reads and what keeps a seventy-product world from running out of names. L10
# enforces the disjointness rather than trusting this list.
PALETTE = [
    "Copper",
    "Zinc",
    "Slate",
    "Amber",
    "Onyx",
    "Pewter",
    "Marigold",
    "Basalt",
    "Quarry",
    "Cinder",
    "Umber",
    "Garnet",
    "Sable",
    "Flint",
    "Ochre",
    "Juniper",
    "Cobble",
    "Thistle",
    "Harrow",
    "Kestrel",
    "Mallow",
    "Nettle",
    "Osprey",
    "Quill",
    "Rowan",
    "Sorrel",
    "Tamarisk",
    "Vetch",
    "Willow",
    "Yarrow",
    "Alder",
    "Bracken",
    "Clover",
    "Dogwood",
    "Elder",
    "Fennel",
    "Gorse",
    "Hazel",
    "Ivy",
    "Kelp",
    "Larch",
    "Myrtle",
    "Nutmeg",
    "Poplar",
]

# The kinds of product a bank documents. Scale comes from spreading more
# products across these, not from inventing new kinds: the kinds are bounded by
# what the official tools can actually execute.
CATEGORY_TEMPLATES = [
    ("cat_checking", "Personal Checking Accounts", "checking"),
    ("cat_savings", "Personal Savings Accounts", "savings"),
    ("cat_cards", "Personal Credit Cards", "card"),
    ("cat_business", "Business Checking Accounts", "business_checking"),
]

PROTOCOLS = [
    ("feat_account_opening", "Account Opening"),
    ("feat_account_closure", "Account Closure"),
]

SUFFIXES = {"savings": " Saver", "card": " Rewards", "business_checking": " Business"}

CONSTRAINTS = {
    "checking": [
        "monthly_fee == 0",
        "early_direct_deposit_days >= 2",
        "mobile_deposit_limit >= 2500",
        "minimum_balance <= 500",
        "overdraft_fee <= 30",
    ],
    "savings": [
        "minimum_balance <= 1000",
        "base_apy >= 3.0",
        "minimum_opening_deposit <= 500",
        "monthly_fee == 0",
        "withdrawal_allowance >= 6",
    ],
    # Every kind states a comparable number of requirements. How much evidence a
    # task needs is bounded by how much it asks about, so a kind with two
    # requirements produces a task a couple of documents can settle while a kind
    # with five needs the catalogue. These all use variables the kind already
    # defines, so widening them costs no regeneration.
    "card": [
        "annual_fee == 0",
        "purchase_apr <= 26.0",
        "base_cashback_rate >= 1.0",
        "foreign_transaction_fee == 0",
        "late_payment_fee <= 35",
    ],
    "business_checking": [
        "monthly_fee == 0",
        "minimum_balance <= 2500",
        "transaction_allowance >= 100",
        "cash_deposit_allowance >= 5000",
        "wire_transfer_fee <= 25",
    ],
}
TRAP_FOR = {"checking": "selection", "savings": "promotional", "card": "temporal"}


def build_categories() -> list[CategorySpec]:
    """The slice's categories; fixed, because the tool set bounds them."""
    return [
        CategorySpec(
            category_id=identifier,
            display_name=name,
            kind="product",
            audience="both",
        )
        for identifier, name, _kind in CATEGORY_TEMPLATES
    ]


# Modifiers pair with the palette the way the official catalogue names products
# -- "Sky Blue", "Light Green", "Silver Plus" -- which is what lets a large world
# keep naming products without ever putting a digit in a name. A digit in a
# product name is not cosmetic: documents state it, and the renderer's guard
# against invented figures cannot tell it from an invented one.
MODIFIERS = [
    "Harbor",
    "Meridian",
    "Northfield",
    "Old",
    "Bright",
    "Deep",
    "First",
    "Grand",
    "Hillside",
    "Ironwood",
    "Lakeside",
    "Summit",
]


def _entity_name(index: int) -> str:
    """A distinct, digit-free product name for any index.

    Modifiers compose as the index grows, so the supply of names is the palette
    times the modifiers squared rather than running out and wrapping.
    """
    base = PALETTE[index % len(PALETTE)]
    tier = index // len(PALETTE)
    if tier == 0:
        return base
    first = MODIFIERS[(tier - 1) % len(MODIFIERS)]
    second = (tier - 1) // len(MODIFIERS)
    if second == 0:
        return f"{first} {base}"
    return f"{MODIFIERS[second % len(MODIFIERS)]} {first} {base}"


def segments_for(per_kind: int) -> list[str]:
    """How many tiers a catalogue of this size is worth layering into.

    A tier has to hold enough products to be a range rather than a shortlist, so
    a small slice keeps a single tier and behaves exactly as it did before
    layering existed. Scale buys diversity: more tiers mean more scenario
    families, each with its own answer and its own evidence.
    """
    count = max(1, min(len(SEGMENTS), per_kind // MIN_SEGMENT))
    return SEGMENTS[:count]


def build_products(topics: int) -> list[ProductSpec]:
    """Spread a topic budget across the categories, round robin.

    Every category keeps at least three products, so a selection family always
    has an answer and two near misses to choose between. Within a category the
    products are dealt round robin into the tiers, so every tier holds a mix of
    the name space rather than one contiguous run of it.
    """
    budget = max(len(CATEGORY_TEMPLATES) * 3, topics - len(PROTOCOLS))
    tiers = segments_for(budget // len(CATEGORY_TEMPLATES))
    products: list[ProductSpec] = []
    for index in range(budget):
        identifier, _name, kind = CATEGORY_TEMPLATES[index % len(CATEGORY_TEMPLATES)]
        entity = _entity_name(index)
        entity = f"{entity}{SUFFIXES.get(kind, '')}"
        rank = index // len(CATEGORY_TEMPLATES)
        products.append(
            ProductSpec(
                feature_id=f"feat_{entity.lower().replace(' ', '_')}",
                entity_name=entity,
                category_id=identifier,
                kind=kind,
                segment=tiers[rank % len(tiers)],
            )
        )
    names = [p.entity_name for p in products]
    if len(set(names)) != len(names):
        # Two products sharing a name would share documents and quietly ruin
        # every task that has to tell them apart.
        raise ValueError(
            f"Name supply exhausted at {len(products)} products; extend PALETTE "
            "or MODIFIERS"
        )
    return products


def build_families(products: list[ProductSpec]) -> list[FamilySpec]:
    """One scenario family per tier of each product kind.

    A customer states requirements rather than a shortlist, so a family ranges
    over a whole tier and its answer must be unique across that tier. Two
    families over the *same* products would be setting the same variables
    against each other, each overwriting the other's answer -- which is why
    tiers, not extra families, are what buys variety: the tiers partition the
    catalogue, so their families never touch the same values.

    The kind's trap alternates across tiers, so the same kind yields both the
    comparison a careless agent gets wrong and the plain one it does not.
    """
    grouped: dict[tuple[str, str], list[ProductSpec]] = {}
    for spec in products:
        grouped.setdefault((spec.kind, spec.segment), []).append(spec)

    families: list[FamilySpec] = []
    for kind, segment in sorted(grouped):
        group = grouped[(kind, segment)]
        if len(group) < 3:
            # Too few to hold an answer and two near misses; the products stay
            # in the catalogue as scenery rather than carrying a task.
            continue
        position = SEGMENTS.index(segment) if segment in SEGMENTS else 0
        trap = TRAP_FOR.get(kind, "selection") if position % 2 == 0 else "selection"
        families.append(
            FamilySpec(
                family_id=f"fam_{kind}_{segment.lower()}",
                kind=trap,
                candidates=[spec.feature_id for spec in group],
                answer=group[0].feature_id,
                near_misses=[spec.feature_id for spec in group[1:3]],
                constraints=CONSTRAINTS[kind],
                description=(
                    f"Choose the only {segment} {kind} product that qualifies."
                ),
                segment=segment,
            )
        )
    return families


CATEGORIES = build_categories()
PRODUCTS = build_products(topics=11)
FAMILIES = build_families(PRODUCTS)


def configure(topics: int) -> None:
    """Resize the slice in place; scale is a setting, not a rewrite."""
    global PRODUCTS, FAMILIES
    PRODUCTS = build_products(topics)
    FAMILIES = build_families(PRODUCTS)


def product(feature_id: str) -> ProductSpec:
    """Look up a product by feature id."""
    for spec in PRODUCTS:
        if spec.feature_id == feature_id:
            return spec
    raise KeyError(feature_id)


def business_feature_id() -> str:
    """The slice's business checking product.

    Looked up by kind rather than by a substring of the identifier: product ids
    carry the product's own name, so "quarry" says nothing about the kind of
    account it is.
    """
    for spec in PRODUCTS:
        if spec.kind == "business_checking":
            return spec.feature_id
    raise KeyError("This slice defines no business checking product")
