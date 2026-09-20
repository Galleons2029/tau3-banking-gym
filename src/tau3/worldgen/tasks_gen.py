"""Stage F: build tasks and the account state that makes their situation true.

Tasks are assembled from what the earlier stages already established: a scenario
family supplies the products and the one constraint set that singles one out, a
verified rule supplies an eligibility refusal, and the allocation supplies the
documents that hold the evidence. Nothing here decides what is true -- it
decides who is asking and what they have.

Reference actions are written in this world's tool names and routed the way an
agent must reach them, through unlock and call, so the official audit records
match what a real trajectory would produce.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import Any

from tau3.domains.banking_knowledge.data_model import TransactionalDB
from tau3.worldgen.models import DocPlan, TaskSpec, WorldSchema
from tau3.worldgen.slice import BANK

VERIFIED_AT = "2025-11-14 03:40:00 EST"
TODAY = "11/14/2025"
LIST_TOOL = "get_all_user_accounts_by_user_id_3847"
OPEN_TOOL = "open_bank_account_4821"
CLOSE_TOOL = "close_bank_account_7392"

FIRST_NAMES = [
    "Dana",
    "Marcus",
    "Priya",
    "Elena",
    "Theo",
    "Nadia",
    "Owen",
    "Ines",
    "Rafael",
    "Yuki",
    "Celia",
    "Hugo",
    "Amara",
    "Lars",
    "Noor",
    "Ivan",
]
LAST_NAMES = [
    "Whitfield",
    "Ilori",
    "Raghunathan",
    "Sokolova",
    "Brennan",
    "Haddad",
    "Kirby",
    "Moreau",
    "Santos",
    "Tanaka",
    "Okonkwo",
    "Lindqvist",
    "Delacroix",
    "Bergman",
    "Farrow",
    "Nakamura",
]
CITIES = [
    ("Boise", "ID", "83702"),
    ("Akron", "OH", "44305"),
    ("Providence", "RI", "02903"),
    ("Tucson", "AZ", "85719"),
    ("Burlington", "VT", "05401"),
    ("Fresno", "CA", "93704"),
    ("Mobile", "AL", "36604"),
    ("Duluth", "MN", "55802"),
]
STREETS = [
    "Copper Lane",
    "Fernwood Court",
    "Harborview Way",
    "Larkspur Avenue",
    "Millbrook Road",
    "Sycamore Terrace",
    "Ashgrove Street",
    "Quarry Bend",
]


@dataclass
class GeneratedTask:
    """One benchmark task plus the specification that explains it."""

    task: dict[str, Any]
    spec: TaskSpec


def customer(rng: random.Random, index: int) -> dict[str, str]:
    """A customer record; identity is the only thing a task needs to verify."""
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[(index * 7 + 3) % len(LAST_NAMES)]
    city, state, postcode = CITIES[index % len(CITIES)]
    street = STREETS[(index * 5 + 1) % len(STREETS)]
    initials = f"{first[0]}{last[0]}".lower()
    user_id = f"{initials}{rng.getrandbits(32):08x}"
    return {
        "name": f"{first} {last}",
        "user_id": user_id,
        "address": f"{rng.randint(10, 4999)} {street}, {city}, {state} {postcode}",
        "email": f"{first.lower()}.{last.lower()}.{user_id}@example.{rng.choice(('net', 'org', 'com'))}",
        "phone_number": f"{rng.randint(200, 989)}-555-{rng.randint(100, 9999):04d}",
        "date_of_birth": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1960, 2000)}",
    }


def verification_action(person: dict[str, str], action_id: str) -> dict:
    """The identity check every servicing task opens with."""
    return {
        "name": "log_verification",
        "arguments": {**person, "time_verified": VERIFIED_AT},
        "requestor": "assistant",
        "action_id": action_id,
    }


def routed(alias: str, arguments: dict | None, action_id: str) -> list[dict]:
    """Unlock then call, the way an agent must reach a discoverable tool."""
    calls = [
        {
            "name": "unlock_discoverable_agent_tool",
            "arguments": {"agent_tool_name": alias},
            "requestor": "assistant",
            "action_id": f"{action_id}_unlock",
        }
    ]
    if arguments is not None:
        calls.append(
            {
                "name": "call_discoverable_agent_tool",
                "arguments": {
                    "agent_tool_name": alias,
                    "arguments": json.dumps(arguments),
                },
                "requestor": "assistant",
                "action_id": f"{action_id}_call",
            }
        )
    return calls


def _account(account_id, user_id, kind, level, opened, status="OPEN", holdings="2500"):
    return {
        "account_id": account_id,
        "user_id": user_id,
        "class": kind,
        "account_type": kind,
        "level": level,
        "date_opened": opened,
        "status": status,
        "current_holdings": holdings,
    }


def evidence_for(
    required: set[str], documents: list[DocPlan], rotation: int = 0
) -> list[str]:
    """A minimal set of documents covering everything a task needs.

    Greedy set cover: at each step take the document that adds the most missing
    knowledge. Minimality is proved separately by Stage G -- this only has to
    produce a candidate small enough to be minimal.

    `rotation` only changes which of several equally good documents is taken
    first. Two customers asking the same thing are answerable from different
    documents whenever the corpus holds a fact in more than one place, and a
    greedy pass that always starts at the same end of the corpus would hand them
    the same gold set and make the evidence look far more concentrated than the
    corpus actually is.
    """
    if documents and rotation:
        offset = rotation % len(documents)
        documents = documents[offset:] + documents[:offset]
    remaining = set(required)
    chosen: list[str] = []
    while remaining:
        best, gain = None, 0
        for document in documents:
            covered = len(remaining & document.reveals)
            if covered > gain:
                best, gain = document, covered
        if best is None:
            break
        chosen.append(best.doc_id)
        remaining -= best.reveals
    # Drop anything the rest already covers, so Stage G's minimality check has a
    # candidate rather than a pile.
    for doc_id in list(chosen):
        others = set()
        for other in chosen:
            if other != doc_id:
                others |= next(d for d in documents if d.doc_id == other).reveals
        if required <= others:
            chosen.remove(doc_id)
    return chosen


def selection_requirements(family: dict, schema: WorldSchema) -> set[str]:
    """The minimal evidence that the chosen product is the right one.

    Not every dimension of every product in the catalogue: that is what a
    solver would read in the worst case, and requiring it inflates the gold set
    well past what the corpus being imitated carries. What a reviewer needs is
    the answer's own figures on every stated requirement, the exact class string
    the opening tool expects, and the single disqualifying figure for each
    product placed deliberately close to qualifying. Products that are clearly
    out are rejected on sight and are not evidence.
    """
    answer = family["answer"]
    # The exact class string the opening tool expects, and the tier that puts the
    # answer inside the range the customer asked for. Without the tier the
    # evidence proves a product qualifies but not that it is one of the ones
    # under consideration.
    required = {
        v.id
        for v in schema.variables
        if v.feature_id == answer
        and v.name in {"account_class", "card_class", "service_tier"}
    }

    def figure(feature_id: str, name: str):
        return next(
            (
                v
                for v in schema.variables
                if v.feature_id == feature_id and v.name == name and not v.derived
            ),
            None,
        )

    names = [constraint.split()[0] for constraint in family["constraints"]]
    for name in names:
        variable = figure(answer, name)
        if variable is not None:
            required.add(variable.id)

    from tau3.worldgen.linters import satisfies

    for feature_id in family["candidates"]:
        if feature_id == answer:
            continue
        # One figure per rejected product: the reason it is out. Taking every
        # dimension of every product instead is what a worst-case solver would
        # read, and it inflates the evidence set well past the corpus being
        # imitated; taking only the two near misses leaves most of the
        # catalogue unaccounted for.
        for constraint, name in zip(family["constraints"], names, strict=True):
            if satisfies(schema, feature_id, constraint):
                continue
            variable = figure(feature_id, name)
            if variable is not None:
                required.add(variable.id)
            break

    # Products outside the stated tier that meet every number the customer gave.
    # The tier is the only thing that rules them out, so the agent has to be able
    # to read their tier -- claiming a gold set without it would be claiming the
    # task is settled by evidence that does not settle it.
    for feature in schema.features:
        if feature.id in family["candidates"]:
            continue
        if not all(satisfies(schema, feature.id, c) for c in family["constraints"]):
            continue
        tier = figure(feature.id, "service_tier")
        if tier is not None:
            required.add(tier.id)
    return required | {OPEN_TOOL}


def prospective_requirements(
    schema: WorldSchema, families: list[dict]
) -> dict[str, set[str]]:
    """What every task this world will contain needs to know.

    Stage C has to protect these before Stage F exists, or one document ends up
    holding most of a task and retrieval stops mattering. Computing them here
    rather than in the allocator keeps the two definitions from drifting: the
    same function decides what a task requires and what may not be concentrated.
    """
    requirements = {
        family["family_id"]: selection_requirements(family, schema)
        for family in families
    }
    requirements["prospective_denial"] = {LIST_TOOL, "rule_savings_tenure"}
    requirements["prospective_ordering"] = {
        LIST_TOOL,
        OPEN_TOOL,
        CLOSE_TOOL,
        "rule_business_no_closed",
    } | {business_class_variable(schema).id}
    return requirements


def selection_task(
    index: int,
    family: dict,
    schema: WorldSchema,
    documents: list[DocPlan],
    alias: dict[str, str],
    person: dict[str, str],
) -> GeneratedTask:
    """Open the only product in the catalogue that meets the stated needs.

    The customer describes requirements and never names a product. That is how
    the official tasks read, and it is what forces the agent to find and compare
    the catalogue rather than look up three names it was handed.
    """
    answer = family["answer"]
    class_variable = next(
        v
        for v in schema.variables
        if v.feature_id == answer and v.name in {"account_class", "card_class"}
    )
    kind = _product_kind(answer) or "checking"
    if kind == "card":
        raise ValueError(
            "Legacy credit-card tools reject synthetic product names. "
            "Use a V2 credit-card category instead of opening a checking account."
        )
    product_noun = (
        "credit card"
        if class_variable.name == "card_class"
        else "business checking account"
        if _product_kind(answer) == "business_checking"
        else "personal bank account"
    )
    segment = family.get("segment", "")

    # The variables a solver must read: the constraint dimensions on every
    # candidate, plus the exact class string the opening tool expects.
    required_variables = selection_requirements(family, schema) - {OPEN_TOOL}

    spec = TaskSpec(
        task_id=f"task_{index:03d}",
        archetype="selection",
        requires_variables=sorted(required_variables),
        requires_tools=[OPEN_TOOL],
        candidate_features=family["candidates"],
        near_misses=family.get("near_misses", []),
        constraints=family["constraints"],
        unique_answer=answer,
        scope=segment,
    )
    wants = "; ".join(
        _phrase(schema, family["answer"], c) for c in family["constraints"]
    )
    task = {
        "id": spec.task_id,
        "description": {
            "purpose": f"Open the only product that meets the stated needs"
        },
        "user_scenario": {
            "instructions": (
                f"You are {person['name']}, a customer of {BANK} "
                f"(email {person['email']}, date of birth {person['date_of_birth']}, "
                f"phone {person['phone_number']}, address {person['address']}). "
                f"You are shopping for a new {product_noun}. "
                f"{_scope_sentence(segment)}"
                "You do not know what this bank's products are called and you "
                "must never name one yourself. What you know is what you need: "
                f"{wants}. "
                "State one requirement per message, in that order, describing "
                "each in your own words as a customer would, and keep going "
                "until you have stated every one of them. Only one product meets "
                "all of them, so if the agent recommends something before you "
                "have finished, say you have more requirements and carry on. If "
                "the agent offers you more than one option, say you do not want "
                "to compare and ask which single one fits. When every "
                "requirement is on the table, ask the agent to open the one "
                "product that meets them all. Do not thank the agent or end the "
                "conversation until they have told you the account is open. "
                "Never respond as the assistant."
            )
        },
        "evaluation_criteria": {
            "actions": [
                verification_action(person, f"{index:03d}_0"),
                *routed(
                    alias[OPEN_TOOL],
                    {
                        "user_id": person["user_id"],
                        "account_type": kind,
                        "account_class": str(class_variable.value),
                    },
                    f"{index:03d}_1",
                ),
            ],
            "reward_basis": ["DB"],
        },
        "required_documents": evidence_for(spec.requires, documents, index),
    }
    return GeneratedTask(task=task, spec=spec)


def _product_kind(feature_id: str) -> str:
    """This world's kind for a product, or an empty string if it defines none."""
    from tau3.worldgen.slice import product

    try:
        return product(feature_id).kind
    except KeyError:
        return ""


def _scope_sentence(segment: str) -> str:
    """How the customer says which tier of the catalogue they are shopping in.

    A tier is sayable without naming a product, which is the whole reason the
    catalogue is layered: it lets several scenario families of one kind coexist,
    each with its own answer, instead of every customer of that kind being
    handed the same one.
    """
    if not segment:
        return ""
    return (
        f"You have been told this bank groups its products into tiers and that "
        f"the {segment} tier is the right range for you, so say early on that "
        f"you want something from the {segment} tier. "
    )


def _phrase(schema: WorldSchema, answer: str, constraint: str) -> str:
    """State a constraint the way a customer would."""
    name, operator, value = constraint.split()
    label = name.replace("_", " ")
    if operator == "==":
        return f"the {label} must be exactly {value}"
    if operator in {">=", ">"}:
        return f"the {label} must be at least {value}"
    return f"the {label} must be no more than {value}"


def denial_task(
    index: int,
    rule_id: str,
    schema: WorldSchema,
    documents: list[DocPlan],
    alias: dict[str, str],
    person: dict[str, str],
) -> GeneratedTask:
    """The customer asks for something the environment will refuse."""
    spec = TaskSpec(
        task_id=f"task_{index:03d}",
        archetype="denial",
        requires_tools=[LIST_TOOL],
        requires_rules=[rule_id],
    )
    task = {
        "id": spec.task_id,
        "description": {"purpose": "Decline a request that fails an eligibility rule"},
        "user_scenario": {
            "instructions": (
                f"You are {person['name']}, a customer of {BANK} "
                f"(email {person['email']}, date of birth {person['date_of_birth']}, "
                f"phone {person['phone_number']}, address {person['address']}). "
                "You opened your checking account less than a fortnight ago and now "
                "want to open a personal savings account as well. You do not know "
                "the rules. If the agent tells you that you are not yet eligible and "
                "explains why, accept it and ask when you may apply. Do not argue. "
                "Never respond as the assistant."
            )
        },
        "evaluation_criteria": {
            "actions": [
                verification_action(person, f"{index:03d}_0"),
                *routed(
                    alias[LIST_TOOL], {"user_id": person["user_id"]}, f"{index:03d}_1"
                ),
            ],
            "reward_basis": ["DB", "NL_ASSERTION"],
            "nl_assertions": [
                "The assistant accurately explains that savings opening requires an "
                "open personal checking account with at least 14 days of tenure, "
                "refuses the premature opening, and states when the user may apply."
            ],
        },
        "required_documents": evidence_for(spec.requires, documents, index),
    }
    return GeneratedTask(task=task, spec=spec)


def business_class_variable(schema: WorldSchema):
    """The variable holding this world's business checking class string."""
    from tau3.worldgen.slice import business_feature_id

    feature_id = business_feature_id()
    for variable in schema.variables:
        if variable.feature_id == feature_id and variable.name == "account_class":
            return variable
    raise KeyError(f"This world documents no account class for {feature_id}")


def business_class(schema: WorldSchema) -> str:
    """The documented class string of this world's business checking product."""
    return str(business_class_variable(schema).value)


def ordering_task(
    index: int,
    schema: WorldSchema,
    documents: list[DocPlan],
    alias: dict[str, str],
    person: dict[str, str],
    closures: list[str],
) -> GeneratedTask:
    """Two requests whose only workable order the customer does not know."""
    reason = "Customer requested closure"
    spec = TaskSpec(
        task_id=f"task_{index:03d}",
        archetype="ordering",
        requires_variables=[business_class_variable(schema).id],
        requires_tools=[LIST_TOOL, OPEN_TOOL, CLOSE_TOOL],
        requires_rules=["rule_business_no_closed"],
        # Both are compared verbatim by the DB evaluator: the closure reason, and
        # the class string of the account to open. A catalogue with seventeen
        # business products makes the second unguessable, so the customer names
        # it -- this task is about the order of the two requests, not the choice.
        verbatim_arguments=[reason, business_class(schema)],
    )
    actions = [
        verification_action(person, f"{index:03d}_0"),
        *routed(alias[LIST_TOOL], {"user_id": person["user_id"]}, f"{index:03d}_1"),
        *routed(
            alias[OPEN_TOOL],
            {
                "user_id": person["user_id"],
                "account_type": "business_checking",
                "account_class": business_class(schema),
            },
            f"{index:03d}_2",
        ),
        *routed(
            alias[CLOSE_TOOL],
            {"account_id": closures[0], "reason": reason},
            f"{index:03d}_3",
        ),
    ]
    for position, account_id in enumerate(closures[1:], start=4):
        actions.append(
            {
                "name": "call_discoverable_agent_tool",
                "arguments": {
                    "agent_tool_name": alias[CLOSE_TOOL],
                    "arguments": json.dumps(
                        {"account_id": account_id, "reason": reason}
                    ),
                },
                "requestor": "assistant",
                "action_id": f"{index:03d}_{position}_call",
            }
        )
    task = {
        "id": spec.task_id,
        "description": {"purpose": "Order a business opening ahead of closures"},
        "user_scenario": {
            "instructions": (
                f"You are {person['name']}, a customer of {BANK} "
                f"(email {person['email']}, date of birth {person['date_of_birth']}, "
                f"phone {person['phone_number']}, address {person['address']}). "
                "In your very first message say that you want to do two things "
                f"today: close your {len(closures)} unused savings accounts, and "
                f"open the {business_class(schema)} for your consultancy -- name "
                "that account exactly, because you already decided which one you "
                "want and this conversation is about the order, not the choice. "
                "State in "
                f"that same first message that the reason for the closures is "
                f"exactly: {reason}. You would prefer the closures done first "
                "because you find them annoying, and you should say so. If the "
                "agent explains that the order matters, accept the explanation and "
                "let them proceed. If asked again for a reason, repeat those exact "
                "words. Never respond as the assistant."
            )
        },
        "evaluation_criteria": {"actions": actions, "reward_basis": ["DB"]},
        "required_documents": evidence_for(spec.requires, documents, index),
    }
    return GeneratedTask(task=task, spec=spec)


def grounding_task(index: int, person: dict[str, str]) -> GeneratedTask:
    """A question the knowledge base cannot answer; the answer is saying so."""
    spec = TaskSpec(task_id=f"task_{index:03d}", archetype="grounding")
    task = {
        "id": spec.task_id,
        "description": {"purpose": "Admit the knowledge base cannot answer"},
        "user_scenario": {
            "instructions": (
                f"You are {person['name']}, a customer of {BANK} "
                f"(email {person['email']}, date of birth {person['date_of_birth']}, "
                f"phone {person['phone_number']}, address {person['address']}). "
                f"You heard from a friend that {BANK} pays cryptocurrency rewards on "
                "its personal checking accounts and you want to know the current "
                "rate. Ask about it plainly. If the agent says it cannot find any "
                "such programme, ask once whether they are sure, then accept the "
                "answer and end the conversation. Do not accept a made-up rate. "
                "Never respond as the assistant."
            )
        },
        "evaluation_criteria": {
            "actions": [],
            "reward_basis": ["DB", "NL_ASSERTION"],
            "nl_assertions": [
                "The assistant states that the knowledge base does not establish "
                "cryptocurrency rewards for personal checking, and does not invent "
                "a programme or rate."
            ],
        },
        "required_documents": [],
    }
    return GeneratedTask(task=task, spec=spec)


def noise_users(rng: random.Random, count: int, start: int) -> tuple[dict, dict]:
    """Customers who take part in no task.

    Without them a database holds exactly the people the tasks are about, and an
    agent can find the right record without having identified anybody.
    """
    users, accounts = {}, {}
    for offset in range(count):
        person = customer(rng, start + offset)
        users[person["user_id"]] = person
        for slot in range(rng.randint(1, 3)):
            account_id = f"acc_{rng.getrandbits(48):012x}"
            accounts[account_id] = _account(
                account_id,
                person["user_id"],
                rng.choice(("checking", "savings")),
                "Noise Account",
                f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/202{rng.randint(3, 5)}",
                holdings=str(rng.randint(0, 40000)),
            )
    return users, accounts


def build_tasks(
    schema: WorldSchema,
    documents: list[DocPlan],
    families: list[dict],
    alias: dict[str, str],
    count: int,
    seed: int,
    noise: int,
) -> tuple[list[GeneratedTask], TransactionalDB]:
    """Assemble the task set and the database state it assumes."""
    rng = random.Random(seed)
    db = TransactionalDB()
    generated: list[GeneratedTask] = []
    index = 1

    def person_for() -> dict[str, str]:
        nonlocal index
        who = customer(rng, index)
        db.users.data[who["user_id"]] = who
        return who

    # One selection task per family, repeated until the budget is filled: the
    # families are what the solver proved to have a unique answer.
    while len(generated) < count:
        for family in families:
            if len(generated) >= count:
                break
            who = person_for()
            generated.append(
                selection_task(index, family, schema, documents, alias, who)
            )
            index += 1

        if len(generated) < count:
            who = person_for()
            db.accounts.data[f"acc_{who['user_id']}_chk"] = _account(
                f"acc_{who['user_id']}_chk",
                who["user_id"],
                "checking",
                "Copper Account",
                "11/09/2025",
            )
            generated.append(
                denial_task(index, "rule_savings_tenure", schema, documents, alias, who)
            )
            index += 1

        if len(generated) < count:
            who = person_for()
            db.accounts.data[f"acc_{who['user_id']}_chk"] = _account(
                f"acc_{who['user_id']}_chk",
                who["user_id"],
                "checking",
                "Copper Account",
                "01/15/2025",
                holdings="8400",
            )
            closures = []
            for slot in (1, 2):
                account_id = f"acc_{who['user_id']}_sav{slot}"
                db.accounts.data[account_id] = _account(
                    account_id,
                    who["user_id"],
                    "savings",
                    "Amber Saver Account",
                    "03/21/2025",
                    holdings="0",
                )
                closures.append(account_id)
            generated.append(
                ordering_task(index, schema, documents, alias, who, closures)
            )
            index += 1

        if len(generated) < count:
            who = person_for()
            generated.append(grounding_task(index, who))
            index += 1

    users, accounts = noise_users(rng, noise, index + 100)
    db.users.data.update(users)
    db.accounts.data.update(accounts)
    return generated[:count], db


def gold_statistics(tasks: list[GeneratedTask]) -> dict:
    """How much evidence the task set leans on, against the measured corpus."""
    sizes = [len(t.task["required_documents"]) for t in tasks]
    referenced: dict[str, int] = {}
    for task in tasks:
        for doc_id in task.task["required_documents"]:
            referenced[doc_id] = referenced.get(doc_id, 0) + 1
    return {
        "mean_gold_documents": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        "min_gold_documents": min(sizes) if sizes else 0,
        "max_gold_documents": max(sizes) if sizes else 0,
        "distinct_gold_documents": len(referenced),
        "gold_document_reuse": (
            round(sum(referenced.values()) / len(referenced), 2) if referenced else 0.0
        ),
        "archetypes": {
            name: sum(1 for t in tasks if t.spec.archetype == name)
            for name in sorted({t.spec.archetype for t in tasks})
        },
    }


def leakage_budget(required: set[str], divisor: int) -> int:
    """How much of a task one document may reveal."""
    return math.ceil(len(required) / divisor) if required else 0
