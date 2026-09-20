"""Small, bounded expression language; never evaluates Python code."""

import ast
import operator
from datetime import date
from decimal import Decimal

ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
COMPARISON = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def parse(expression: str) -> ast.Expression:
    """Parse only a bounded grammar, rejecting unsupported capabilities early."""
    if len(expression) > 4096:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Attribute,
        ast.BinOp,
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.UAdd,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.IfExp,
        ast.List,
        ast.Tuple,
        ast.Call,
        ast.Subscript,
        *ARITHMETIC,
        *COMPARISON,
    )
    if len(nodes) > 256 or any(not isinstance(n, allowed) for n in nodes):
        raise ValueError("Unsupported expression")
    for node in nodes:
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attributes are forbidden")
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in {"days", "money", "min", "max"}
                or node.keywords
            ):
                raise ValueError("Unsupported function")
    return tree


def references(expression: str, namespace: str) -> set[str]:
    """Return named members read from a dictionary namespace."""
    tree = parse(expression)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == namespace
        ):
            raise ValueError(f"Use named fields for {namespace} dependency tracking")
    return {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == namespace
    }


def evaluate(expression: str, context: dict):
    """Interpret arithmetic, predicates and dates over plain values only."""

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (float, int)) and not isinstance(
                node.value, bool
            ):
                return Decimal(str(node.value))
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in context:
                raise ValueError(f"Unknown name: {node.id}")
            return context[node.id]
        if isinstance(node, ast.Attribute):
            container = visit(node.value)
            if not isinstance(container, dict) or node.attr not in container:
                raise ValueError(f"Unknown field: {node.attr}")
            return container[node.attr]
        if isinstance(node, ast.Subscript):
            container, key = visit(node.value), visit(node.slice)
            if not isinstance(container, dict) or key not in container:
                raise ValueError("Unknown dictionary entry")
            return container[key]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [visit(n) for n in node.elts]
        if isinstance(node, ast.BinOp):
            return ARITHMETIC[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            return (
                not value
                if isinstance(node.op, ast.Not)
                else -value
                if isinstance(node.op, ast.USub)
                else +value
            )
        if isinstance(node, ast.BoolOp):
            values = (bool(visit(n)) for n in node.values)
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op, right in zip(node.ops, node.comparators, strict=True):
                value = visit(right)
                if not COMPARISON[type(op)](left, value):
                    return False
                left = value
            return True
        if isinstance(node, ast.IfExp):
            return visit(node.body if visit(node.test) else node.orelse)
        if isinstance(node, ast.Call):
            args = [visit(n) for n in node.args]
            name = node.func.id
            if name == "days":
                return Decimal(
                    (date.fromisoformat(args[0]) - date.fromisoformat(args[1])).days
                )
            if name == "money":
                return Decimal(str(args[0])).quantize(Decimal("0.01"))
            return (min if name == "min" else max)(args)
        raise ValueError("Unsupported expression node")

    try:
        return visit(parse(expression))
    except (
        TypeError,
        KeyError,
        IndexError,
        ArithmeticError,
        RecursionError,
        SyntaxError,
    ) as exc:
        raise ValueError(f"Invalid expression: {expression}") from exc
