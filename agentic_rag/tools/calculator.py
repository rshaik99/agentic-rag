"""
Arithmetic tool. Deliberately NOT `eval()` -- the expression comes from LLM
tool-call arguments, which is untrusted input as far as this process is
concerned. Walks a whitelisted AST instead so nothing beyond +-*/**%()
and numeric literals can execute.
"""
from __future__ import annotations

import ast
import operator

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression element: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Evaluate a numeric arithmetic expression, e.g. '(340 - 210) / 210 * 100'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except Exception as e:                                       # noqa: BLE001
        return f"ERROR: could not evaluate {expression!r}: {e}"
    return str(result)
