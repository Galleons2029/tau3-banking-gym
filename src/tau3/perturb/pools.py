"""Word banks that new names are drawn from.

Two deliberate properties:

**Family-coherent.**  Each product family draws from one lexical family, so a
variant still reads like a bank's product line rather than a random word salad.

**Order-neutral.**  The canonical tiers (Bronze < Silver < Gold < Platinum <
Diamond) let a model rank products from world knowledge alone, without reading
a single document.  Replacement families carry no such implied ordering, so the
documented numbers become the only way to rank them -- which is the behaviour
we are trying to train.  The numbers still define the true order; only the
free shortcut is removed.

Pools are large enough that several variants can draw disjoint names; the plan
builder enforces disjointness explicitly rather than trusting pool size.
"""

#: Two-word bank names.  Deliberately not near-misses of "Rho-Bank".
BRAND_ROOTS: tuple[str, ...] = (
    "Vela Trust",
    "Kestrel Union",
    "Aster Financial",
    "Halcyon Trust",
    "Meridian Union",
    "Sable Financial",
    "Lumen Trust",
    "Corvus Union",
    "Ardent Financial",
    "Zephyr Trust",
    "Cinder Union",
    "Verity Financial",
)

#: Lexical families for product names.  Each word is a single token so that
#: composed names ("Business <X> Rewards Card") stay readable.
FAMILIES: dict[str, tuple[str, ...]] = {
    "mineral": (
        "Quartz", "Onyx", "Slate", "Basalt", "Marble", "Granite", "Obsidian",
        "Jasper", "Flint", "Shale", "Gypsum", "Feldspar", "Agate", "Opal",
        "Topaz", "Beryl", "Garnet", "Zircon", "Pumice", "Mica",
    ),
    "botanical": (
        "Alder", "Birch", "Cedar", "Willow", "Hazel", "Rowan", "Aspen",
        "Maple", "Juniper", "Laurel", "Myrtle", "Olive", "Poplar", "Sorrel",
        "Thistle", "Bracken", "Clover", "Fennel", "Bramble", "Nettle",
    ),
    "avian": (
        "Kestrel", "Merlin", "Osprey", "Petrel", "Plover", "Curlew", "Godwit",
        "Dunlin", "Redwing", "Fieldfare", "Wheatear", "Linnet", "Siskin",
        "Twite", "Brambling", "Avocet", "Turnstone", "Sanderling",
    ),
    "celestial": (
        "Vega", "Altair", "Rigel", "Mizar", "Alcor", "Deneb", "Antares",
        "Capella", "Procyon", "Bellatrix", "Elnath", "Alphard", "Markab",
        "Sadr", "Izar", "Talitha", "Alkaid", "Phecda",
    ),
    "textile": (
        "Cambric", "Damask", "Poplin", "Sateen", "Chenille", "Organza",
        "Taffeta", "Grosgrain", "Batiste", "Jacquard", "Brocade", "Velour",
        "Chambray", "Seersucker", "Gabardine", "Twill",
    ),
}

#: Which lexical family each product category draws from.
CATEGORY_FAMILY: dict[str, str] = {
    "checking_accounts": "mineral",
    "savings_accounts": "botanical",
    "credit_cards": "avian",
    "business_checking_accounts": "celestial",
    "business_savings_accounts": "textile",
    "business_credit_cards": "avian",
    "buy_now_pay_later": "botanical",
    "everyone_pay": "celestial",
    "personal_subscriptions": "mineral",
}

#: Verb phrases for renamed tools, paired by the action they describe so a
#: renamed tool still reads like what it does.
TOOL_VERBS: dict[str, tuple[str, ...]] = {
    "get": ("fetch", "lookup", "read", "retrieve", "list", "inspect"),
    "list": ("enumerate", "list", "index", "survey"),
    "submit": ("file", "lodge", "register", "raise", "post"),
    "open": ("create", "establish", "start", "initiate"),
    "close": ("terminate", "shutter", "wind_down", "retire"),
    "update": ("revise", "amend", "adjust", "modify"),
    "approve": ("authorize", "grant", "clear", "endorse"),
    "deny": ("decline", "reject", "refuse", "block"),
    "transfer": ("move", "shift", "route", "remit"),
    "activate": ("enable", "switch_on", "arm", "commission"),
    "apply": ("request", "claim", "seek", "petition"),
    "cancel": ("void", "revoke", "rescind", "annul"),
    "request": ("ask", "solicit", "order", "demand"),
    "verify": ("confirm", "validate", "attest", "certify"),
    "send": ("dispatch", "deliver", "transmit", "issue"),
    "add": ("attach", "append", "register", "enroll"),
    "remove": ("detach", "withdraw", "strike", "purge"),
    "change": ("alter", "switch", "reassign", "swap"),
    "set": ("assign", "configure", "define", "fix"),
    "check": ("test", "probe", "assess", "review"),
    "create": ("mint", "spawn", "draft", "generate"),
    "reset": ("clear", "restore", "rewind", "reinitialize"),
    "freeze": ("suspend", "hold", "lock", "quarantine"),
    "unfreeze": ("resume", "release", "unlock", "reinstate"),
    "report": ("flag", "notify", "escalate", "record"),
    "schedule": ("plan", "queue", "book", "slot"),
    "process": ("handle", "settle", "execute", "fulfil"),
    "calculate": ("compute", "derive", "total", "tally"),
    "search": ("find", "seek", "scan", "locate"),
    "link": ("bind", "couple", "associate", "pair"),
    "unlink": ("unbind", "decouple", "dissociate", "unpair"),
    "enable": ("turn_on", "permit", "allow", "admit"),
    "disable": ("turn_off", "forbid", "bar", "deactivate"),
    "give": ("grant", "hand", "provide", "supply"),
    "file": ("submit", "lodge", "record", "enter"),
    "order": ("request", "commission", "place", "arrange"),
    "replace": ("swap", "substitute", "renew", "supersede"),
    "dispute": ("contest", "challenge", "query", "appeal"),
    "redeem": ("cash_in", "claim", "convert", "exchange"),
    "waive": ("forgive", "excuse", "remit", "drop"),
    "log": ("record", "note", "journal", "register"),
    "clear": ("resolve", "dismiss", "lift", "discharge"),
    "deposit": ("lodge", "remit", "credit", "pay_in"),
    "emergency": ("urgent", "expedited", "priority", "rush"),
    "example": ("sample", "demo", "specimen", "illustrative"),
    "initial": ("first", "opening", "starting", "inaugural"),
    "pay": ("settle", "remit", "discharge", "clear"),
}


def family_for(category: str) -> str:
    """Lexical family a product category draws from.

    Raises:
        KeyError: If the category has no configured family.
    """
    return CATEGORY_FAMILY[category]
