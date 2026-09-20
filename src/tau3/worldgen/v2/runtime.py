"""Atomic execution of declarative operations over strictly checked tables."""

import hashlib
import json
from copy import deepcopy
from decimal import Decimal

from pydantic import Field

from tau3.environment.db import DB
from tau3.worldgen.v2.expressions import evaluate
from tau3.worldgen.v2.specs import WorldSpec


def serializable(value):
    """Keep decimal values exact in JSON artifacts and tool responses."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return value


def digest(value) -> str:
    """Content identity independent of dictionary insertion order."""
    return hashlib.sha256(
        json.dumps(serializable(value), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class WorldDB(DB):
    """Dynamic business storage; contracts, goals and gold are never stored here."""

    users: dict[str, dict] = Field(default_factory=dict)
    tables: dict[str, dict[str, dict]] = Field(default_factory=dict)
    events: list[dict] = Field(default_factory=list)


def validate_db(spec: WorldSpec, db: WorldDB) -> None:
    """Check types, keys, ownership, product categories and row invariants."""
    entities = {e.id: (c, e) for c in spec.categories for e in c.entities}
    if set(db.tables) != set(entities):
        raise ValueError("Database table set does not match world specification")
    emails = [u.get("email") for u in db.users.values()]
    if None in emails or len(set(emails)) != len(emails):
        raise ValueError("Users require unique email addresses")
    from datetime import date

    for user in db.users.values():
        joined = user.get("joined_on", "")
        if date.fromisoformat(joined).isoformat() != joined or joined > spec.clock:
            raise ValueError("Invalid customer history date")
    for table, rows in db.tables.items():
        category, entity = entities[table]
        products = {p.id for p in category.products}
        for row_id, row in rows.items():
            if not isinstance(row_id, str) or not row_id:
                raise ValueError("Invalid row id")
            if set(row) != set(entity.fields):
                raise ValueError(f"Row fields do not match {table}: {row_id}")
            typed = {k: entity.fields[k].coerce(v) for k, v in row.items()}
            if row["user_id"] not in db.users or row["product_id"] not in products:
                raise ValueError("Invalid customer or product/category relationship")
            for rule in entity.invariants:
                if evaluate(rule, {"row": typed, "clock": spec.clock}) is not True:
                    raise ValueError(f"Row invariant failed: {table}:{rule}")
            for fk in entity.foreign_keys:
                if row[fk.field] not in db.tables[fk.table]:
                    raise ValueError("Dangling foreign key")
                target = db.tables[fk.table][row[fk.field]]
                if target["user_id"] != row["user_id"]:
                    raise ValueError("Foreign key crosses customer ownership")
        for field in entity.unique:
            values = [r[field] for r in rows.values()]
            if len({json.dumps(v, sort_keys=True) for v in values}) != len(values):
                raise ValueError(f"Duplicate unique field: {table}.{field}")


class OperationRuntime:
    """World-local runtime. Failed operations leave all storage unchanged."""

    def __init__(self, spec: WorldSpec, db: WorldDB):
        self.spec, self.db = spec, db
        self.operations = {o.id: (c, o) for c in spec.categories for o in c.operations}
        self.aliases = {}
        for op_id in sorted(self.operations):
            for attempt in range(10000):
                suffix = 1000 + int(digest([spec.seed, op_id, attempt])[:12], 16) % 9000
                alias = f"{op_id}_{suffix}"
                if alias not in self.aliases:
                    self.aliases[alias] = op_id
                    break

    def execute(self, name: str, arguments: dict, actor: str = "assistant") -> dict:
        """Execute a capability by its public alias, enforcing technical guards."""
        if name not in self.aliases:
            raise ValueError("Unknown tool alias")
        category, operation = self.operations[self.aliases[name]]
        if actor != operation.actor:
            raise ValueError("Wrong execution actor")
        if set(arguments) != set(operation.parameters):
            raise ValueError("Missing or unknown arguments")
        args = {k: operation.parameters[k].coerce(v) for k, v in arguments.items()}
        if args["user_id"] not in self.db.users:
            raise ValueError("Unknown customer")
        product = next(
            (p for p in category.products if p.id == args["product_id"]), None
        )
        if product is None:
            raise ValueError("Product does not belong to this operation's category")
        if operation.kind == "read":
            query = operation.query
            offset = int(args["offset"])
            if offset < 0:
                raise ValueError("Negative query offset")
            entity = next(e for e in category.entities if e.id == query.table)
            rows = [
                {"record_id": rid, **{k: row[k] for k in query.fields}}
                for rid, row in sorted(self.db.tables[query.table].items())
                if row["user_id"] == args["user_id"]
                and row["product_id"] == args["product_id"]
                and all(
                    entity.fields[field].coerce(row[field]) == args[param]
                    for field, param in query.filters.items()
                )
            ]
            return {
                "records": rows[offset : offset + query.page_size],
                "total": len(rows),
                "next_offset": offset + query.page_size
                if offset + query.page_size < len(rows)
                else None,
            }
        entities = {e.id: e for c in self.spec.categories for e in c.entities}
        typed_tables = {
            name: {
                rid: {k: entities[name].fields[k].coerce(v) for k, v in row.items()}
                for rid, row in rows.items()
            }
            for name, rows in self.db.tables.items()
        }
        record = {}
        if operation.record_table:
            record = typed_tables[operation.record_table].get(args.get("record_id"))
            if record is None:
                raise ValueError("Record not found")
            if (
                record["user_id"] != args["user_id"]
                or record["product_id"] != product.id
            ):
                raise ValueError("Record ownership/product mismatch")
        context = {
            "args": args,
            "record": record,
            "tables": typed_tables,
            "facts": product.values(self.spec.clock),
            "clock": self.spec.clock,
            "user": self.db.users[args["user_id"]],
        }
        policies = {p.id: p for p in category.policies}
        violations = []
        for rule_id in operation.rules:
            policy = policies[rule_id]
            if evaluate(policy.predicate, context) is not True:
                if policy.enforcement == "runtime":
                    raise ValueError(f"POLICY_DENIED:{rule_id}")
                violations.append(rule_id)
        proposed = self.db.model_copy(deep=True)
        changes = []
        for effect in operation.effects:
            key = evaluate(effect.key, context)
            if not isinstance(key, str) or not key:
                raise ValueError("Effect key must be a nonempty string")
            rows = proposed.tables[effect.table]
            if effect.mode == "create" and key in rows:
                raise ValueError("Record already exists")
            if effect.mode == "update" and key not in rows:
                raise ValueError("Cannot update missing record")
            before = deepcopy(rows.get(key))
            if before and before["user_id"] != args["user_id"]:
                raise ValueError("Cross-customer write forbidden")
            row = dict(before or {})
            row.update(
                {
                    k: serializable(evaluate(v, context))
                    for k, v in effect.values.items()
                }
            )
            if (
                row.get("user_id") != args["user_id"]
                or row.get("product_id") != product.id
            ):
                raise ValueError("Effect changes customer/product identity")
            rows[key] = row
            changes.append(
                {"table": effect.table, "row_id": key, "before": before, "after": row}
            )
        validate_db(self.spec, proposed)
        typed_after = {
            name: {
                rid: {k: entities[name].fields[k].coerce(v) for k, v in row.items()}
                for rid, row in rows.items()
            }
            for name, rows in proposed.tables.items()
        }
        for condition in operation.postconditions:
            if evaluate(condition, {**context, "after": typed_after}) is not True:
                raise ValueError("Operation postcondition failed; no changes committed")
        proposed.events.append(
            {
                "operation": operation.id,
                "actor": actor,
                "arguments": serializable(args),
                "clock": self.spec.clock,
                "changes": changes,
                "policy_violations": violations,
            }
        )
        self.db.tables = proposed.tables
        self.db.events = proposed.events
        return {
            "status": "success",
            "records": [
                {"table": c["table"], "row_id": c["row_id"], **c["after"]}
                for c in changes
            ],
        }


def goal_errors(
    initial: WorldDB,
    actual: WorldDB,
    goals: list,
    user_id: str,
    spec: WorldSpec | None = None,
) -> list[str]:
    """Independent assertions: outcome, ownership and forbidden side effects."""
    errors = []
    allowed = {(g.table, g.row_id): g for g in goals}
    entities = {e.id: e for c in spec.categories for e in c.entities} if spec else {}

    def equal(table, field, a, b):
        if table in entities:
            try:
                scalar = entities[table].fields[field]
                return scalar.coerce(a) == scalar.coerce(b)
            except (ValueError, ArithmeticError):
                return False
        return a == b

    for goal in goals:
        row = actual.tables.get(goal.table, {}).get(goal.row_id)
        if row is None or any(
            not equal(goal.table, k, row.get(k), v) for k, v in goal.fields.items()
        ):
            errors.append(f"Goal not met: {goal.table}/{goal.row_id}")
        if row is not None and row.get("user_id") != user_id:
            errors.append("Goal belongs to another customer")
    if initial.users != actual.users:
        errors.append("Unexpected identity mutation")
    for table in initial.tables.keys() | actual.tables.keys():
        before, after = initial.tables.get(table, {}), actual.tables.get(table, {})
        for row_id in before.keys() | after.keys():
            if before.get(row_id) == after.get(row_id):
                continue
            goal = allowed.get((table, row_id))
            if goal is None:
                errors.append(f"Unexpected side effect: {table}/{row_id}")
            elif row_id in before and row_id in after:
                if any(
                    not equal(table, k, after[row_id].get(k), v)
                    for k, v in before[row_id].items()
                    if k not in goal.fields
                ):
                    errors.append(f"Unexpected field mutation: {table}/{row_id}")
    if actual.events[: len(initial.events)] != initial.events:
        errors.append("Seed history was rewritten")
    for event in actual.events[len(initial.events) :]:
        for change in event.get("changes", []):
            goal = allowed.get((change["table"], change["row_id"]))
            if goal is None or change["after"].get("user_id") != user_id:
                errors.append("Forbidden intermediate mutation")
            elif change.get("before") and any(
                not equal(change["table"], k, change["after"].get(k), v)
                for k, v in change["before"].items()
                if k not in goal.fields
            ):
                errors.append("Forbidden intermediate field mutation")
    if any(e.get("policy_violations") for e in actual.events[len(initial.events) :]):
        errors.append("Trajectory violates a policy obligation")
    return errors
