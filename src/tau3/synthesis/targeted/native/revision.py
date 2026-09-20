"""Carry spent budgets into an explicit repair version without rewriting evidence."""

from pathlib import Path

from tau3.synthesis.storage import digest, read_json, write_json


def inherit_budget(root, config):
    """Bind a stopped parent's ledger and seed cumulative counters exactly once."""
    if not config.budget_parent_round:
        return
    parent = Path(config.budget_parent_round).resolve()
    if parent == root.resolve() or str(parent) != str(Path(config.parent_round or "").resolve()):
        raise ValueError("Budget parent must be a distinct declared parent round")
    if read_json(parent / "supervisor.json")["status"] == "RUNNING":
        raise ValueError("Cannot inherit a running parent's budget")
    origin = {"parent": str(parent), "identity": read_json(parent / "identity.json"),
              "budget": read_json(parent / "budget.json"),
              "blind_budget": read_json(parent / "blind-budget.json")}
    if config.reuse_parent_pilot:
        measured = read_json(parent / "pilot/budget-measurement.json")
        if digest(measured["budget"]) != measured["budget_hash"]:
            raise ValueError("Parent pilot budget measurement changed")
        parent_origin = read_json(parent / "budget-inheritance.json")
        prior = parent_origin["budget"]
        # A parent may itself have reused captures. Carry their original cost
        # through every revision instead of estimating production from re-audits.
        reused = parent_origin.get("reused_pilot_cost", {})
        origin["reused_pilot_cost"] = {key: measured["budget"][key] - prior[key] + reused.get(key, 0)
                                       for key in ("calls", "audit_calls", "tokens")}
        origin["pilot_measurement_hash"] = digest(measured)
    origin["hash"] = digest(origin)
    path = root / "budget-inheritance.json"
    if path.exists():
        if read_json(path) != origin:
            raise ValueError("Inherited budget evidence changed")
        for filename, key, fields in (
            ("budget.json", "budget", ("calls", "audit_calls", "tokens", "rollouts")),
            ("blind-budget.json", "blind_budget", ("rollouts",)),
        ):
            ledger = read_json(root / filename)
            if any(ledger.get(field, 0) < origin[key].get(field, 0) for field in fields):
                raise ValueError("Repair ledger lost inherited spend")
        return
    # A crash before the final binding can replay these initial writes, but may
    # never overwrite counters that have already advanced.
    for filename, key in (("budget.json", "budget"), ("blind-budget.json", "blind_budget")):
        destination = root / filename
        if destination.exists() and read_json(destination) != origin[key]:
            raise ValueError("Cannot overwrite an active repair ledger")
        write_json(destination, origin[key])
    write_json(path, origin)
