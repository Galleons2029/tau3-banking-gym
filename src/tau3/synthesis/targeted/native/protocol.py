"""Audit attempted calls as well as successful calls without repairing captures."""

import inspect
import json
import re
from functools import lru_cache
from typing import Literal, get_args, get_origin, get_type_hints

from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools
from tau3.environment.toolkit import DISCOVERABLE_ATTR


@lru_cache(maxsize=256)
def _contract(method):
    return inspect.signature(method), get_type_hints(method)


def tool_error(message):
    """Recognize transport flags and business tools' textual/JSON errors."""
    if message.get("error"):
        return True
    content = message.get("content") or ""
    if not isinstance(content, str):
        return True
    if re.search(r"(?i)^\s*(?:error\s*:|failed to\b)", content):
        return True
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return False
    return isinstance(value, dict) and bool(value.get("error"))


def is_protocol_misuse(error):
    """Separate wrapper/unlock misuse from missing parameters of a real business tool."""
    if error["issue"] in {"invalid_selector_or_wrapper_signature", "nested_arguments_not_string",
                           "nested_arguments_not_object", "call_without_prior_unlock",
                           "user_call_without_grant", "hidden_tool_bypass"}:
        return True
    return (error["issue"] in {"argument_signature", "argument_type"}
            and error["operation"] == error["wrapper"]
            and error["wrapper"] in {"unlock_discoverable_agent_tool", "give_discoverable_user_tool"})


def _matches(value, annotation):
    if get_origin(annotation) is Literal:
        return value in get_args(annotation)
    if get_origin(annotation) is not None:
        return any(_matches(value, item) for item in get_args(annotation))
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int or (type(value) is float and value.is_integer())
    if annotation is float:
        return type(value) in {int, float}
    if annotation is str:
        return isinstance(value, str)
    if annotation is type(None):
        return value is None
    return True


def audit_protocol(messages):
    """Return localized errors and exact unlock→call evidence from original messages."""
    messages = [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in messages]
    responses = {m.get("id", m.get("tool_call_id")): (p, m)
                 for p, m in enumerate(messages) if m.get("role") == "tool"}
    unlocked, granted, errors, calls, ids = {}, {}, [], [], set()
    turn = -1
    for position, message in enumerate(messages):
        actor = message.get("role")
        if actor == "assistant":
            turn += 1
        for call in message.get("tool_calls") or []:
            fn = call.get("function", call)
            name, args = fn.get("name"), fn.get("arguments")
            issues = []
            if not isinstance(name, str) or not name:
                issues.append("invalid_tool_name")
            identifier = call.get("id")
            if not identifier or identifier in ids:
                issues.append("duplicate_or_missing_call_id")
            ids.add(identifier)
            response = responses.get(identifier)
            if response is None or response[0] <= position:
                issues.append("missing_or_misordered_result")
            failed = response is not None and tool_error(response[1])
            if failed:
                issues.append("tool_error")
                if re.search(r"(?i)(?:unknown (?:discoverable )?tool|tool .*not found)", response[1].get("content") or ""):
                    issues.append("unknown_tool")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            if not isinstance(args, dict):
                issues.append("invalid_json_object")
                args = {}
            wrapper = name
            owner = KnowledgeUserTools if actor == "user" else KnowledgeTools
            if name in {"call_discoverable_agent_tool", "call_discoverable_user_tool"}:
                agent = name == "call_discoverable_agent_tool"
                selector = "agent_tool_name" if agent else "discoverable_tool_name"
                if set(args) != {selector, "arguments"} or not isinstance(args.get(selector), str):
                    issues.append("invalid_selector_or_wrapper_signature")
                name = args.get(selector)
                nested = args.get("arguments")
                if not isinstance(nested, str):
                    issues.append("nested_arguments_not_string")
                    nested = None
                try:
                    args = json.loads(nested) if nested is not None else {}
                except ValueError:
                    issues.append("invalid_nested_json")
                    args = {}
                if not isinstance(args, dict):
                    issues.append("nested_arguments_not_object")
                    args = {}
                if agent and (actor != "assistant" or unlocked.get(name, len(messages)) >= position):
                    issues.append("call_without_prior_unlock")
                if not agent and (actor != "user" or granted.get(name, len(messages)) >= position):
                    issues.append("user_call_without_grant")
            method = getattr(owner, name, None) if isinstance(name, str) else None
            # Retrieval tools are furnished by the configured retrieval toolkit.
            if method is None and isinstance(name, str) and name not in {"KB_search", "grep"}:
                issues.append("unknown_tool")
            elif method is not None:
                try:
                    signature, hints = _contract(method)
                    signature.bind(None, **args)
                    if any(not _matches(value, hints.get(key)) for key, value in args.items()):
                        issues.append("argument_type")
                except (TypeError, ValueError):
                    issues.append("argument_signature")
                if getattr(method, DISCOVERABLE_ATTR, False) and wrapper == name:
                    issues.append("hidden_tool_bypass")
            # An unlock/grant only takes effect after its response, never within
            # the same parallel call batch before the model has seen the schema.
            if response and not failed and not issues:
                if wrapper == "unlock_discoverable_agent_tool":
                    unlocked[args.get("agent_tool_name")] = response[0]
                elif wrapper == "give_discoverable_user_tool":
                    granted[args.get("discoverable_tool_name")] = response[0]
            entry = {"call_id": identifier, "actor": actor, "assistant_turn": turn if actor == "assistant" else None,
                     "wrapper": wrapper, "operation": name, "issues": sorted(set(issues))}
            calls.append(entry)
            errors.extend({**entry, "issue": issue} for issue in entry["issues"])
    return {"clean": not errors, "errors": errors, "calls": calls,
            "erroneous_assistant_turns": sorted({e["assistant_turn"] for e in errors if e["assistant_turn"] is not None}),
            "user_errors": sum(e["actor"] == "user" for e in errors)}
