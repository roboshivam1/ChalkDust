"""Turn a model-written arithmetic string into a sympy expression WITHOUT
eval: sympy.sympify evaluates its input as Python, which is unsafe for model
output. The string is parsed with ast and walked against an allowlist
(numbers, + - * / ** %, unary +/-, parentheses, allowlisted names and
single-argument-or-more calls to allowlisted functions); the sympy object is
built from the tree here. Exponents are bounded so '9**9**9' cannot hang the
process.
"""

from __future__ import annotations

import ast
import operator

import sympy as sp

MAX_LEN = 300
MAX_EXP = 64

FUNCS = {"sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin,
         "acos": sp.acos, "atan": sp.atan, "exp": sp.exp, "log": sp.log, "ln": sp.log,
         "abs": sp.Abs, "Abs": sp.Abs}
CONSTS = {"pi": sp.pi, "E": sp.E, "e": sp.E, "deg": sp.pi / 180, "degrees": sp.pi / 180}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}


class UnsafeExpression(ValueError):
    pass


def safe_sympify(expr: str, symbols: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    if not isinstance(expr, str) or len(expr) > MAX_LEN:
        raise UnsafeExpression("too long")
    names = {**CONSTS, **(symbols or {})}
    tree = ast.parse(expr.replace("^", "**"), mode="eval")

    def build(n):
        if isinstance(n, ast.Expression):
            return build(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return sp.Rational(n.value) if isinstance(n.value, int) else sp.Float(n.value)
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            raise UnsafeExpression(f"name {n.id!r} not allowed")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = build(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = build(n.left), build(n.right)
            if isinstance(n.op, ast.Pow):
                if b.is_number and abs(float(sp.N(b))) > MAX_EXP:
                    raise UnsafeExpression("exponent too large")
            return _BIN[type(n.op)](a, b)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS
                and not n.keywords and 1 <= len(n.args) <= 2):
            return FUNCS[n.func.id](*[build(a) for a in n.args])
        raise UnsafeExpression(f"{type(n).__name__} not allowed")

    return build(tree)
