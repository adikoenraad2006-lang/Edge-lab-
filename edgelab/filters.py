"""
Filter expressions.

Filters are the flexibility escape hatch: any boolean expression over the
feature values available at the moment of the event. They are evaluated with a
restricted AST walker rather than eval(), so a spec that arrives from the
language model (or from a file someone emailed you) cannot execute anything.

Available names are whatever the scanner puts in the feature row. See
FEATURE_HELP for the standard set.
"""

from __future__ import annotations

import ast
import operator as op

def _pow(a, b):
    # Unbounded exponents (9 ** 9 ** 9) would hang the scan.
    if isinstance(b, (int, float)) and abs(b) > 64:
        raise ExpressionError("exponent too large (max 64)")
    return op.pow(a, b)


def _mul(a, b):
    # 'x' * 10 ** 9 would build a gigabyte string; filters only need numbers.
    if isinstance(a, (str, list)) or isinstance(b, (str, list)):
        raise ExpressionError("* is only allowed on numbers")
    return op.mul(a, b)


_BIN = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: _mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod,
    ast.Pow: _pow,
}
_CMP = {
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge, ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_UNARY = {ast.USub: op.neg, ast.UAdd: op.pos, ast.Not: op.not_}

_FUNCS = {
    "abs": abs, "min": min, "max": max, "round": round,
    "int": int, "float": float, "bool": bool,
}

FEATURE_HELP = {
    "hour": "hour of day, 0-23, in the data's timezone",
    "minute": "minute of hour",
    "dow": "day of week, 0=Monday",
    "month": "1-12",
    "year": "calendar year",
    "session": "'ASIA' | 'LONDON' | 'RTH' | 'AFTER' (string, quote it)",
    "atr": "ATR on the detection timeframe, in price points",
    "zone_height": "zone size in price points",
    "zone_height_atr": "zone size divided by ATR at formation",
    "bars_since_zone": "detection-timeframe bars between formation and the test",
    "test_number": "1 for first test, 2 for second, etc.",
    "displacement_atr": "size of the move that created the zone, in ATR",
    "bias": "1 for long setups, -1 for short setups",
    "dist_ema50_atr": "(price - EMA50) / ATR at the moment of the test",
    "day_range_atr": "the day's range so far, in ATR",
    "prior_day_range_atr": "previous day's full range, in ATR",
    "gap_atr": "today's open minus prior close, in ATR",
    # --- higher timeframe (requires htf.timeframe to be set) ---------------
    "htf_bias": "+1 if the last CLOSED HTF bar closed above its EMA, else -1",
    "htf_bar_dir": "+1 if the last CLOSED HTF bar closed up, else -1",
    "htf_slope_atr": "HTF EMA change over slope_bars, in HTF ATR",
    "htf_dist_ema_atr": "(HTF close - HTF EMA) / HTF ATR",
    "htf_range_atr": "the last closed HTF bar's range, in HTF ATR",
    "htf_bar_progress": ("how far into the current HTF bar this event sits; "
                         "0.0 just after a close, ~1.0 just before the next. "
                         "Exceeds 1.0 after a weekend or holiday gap, which is "
                         "itself a useful filter: htf_bar_progress > 1 means "
                         "the HTF reference is stale"),
    "aligned": "1 if the setup's direction matches htf_bias, else 0",
    "aligned_bar": "1 if the setup's direction matches htf_bar_dir, else 0",
}


class ExpressionError(ValueError):
    pass


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExpressionError(
                f"unknown name {node.id!r}; available: {sorted(env)}"
            )
        return env[node.id]
    if isinstance(node, ast.BinOp):
        fn = _BIN.get(type(node.op))
        if fn is None:
            raise ExpressionError("operator not allowed")
        return fn(_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY.get(type(node.op))
        if fn is None:
            raise ExpressionError("operator not allowed")
        return fn(_eval(node.operand, env))
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for o, comp in zip(node.ops, node.comparators):
            fn = _CMP.get(type(o))
            if fn is None:
                raise ExpressionError("comparison not allowed")
            right = _eval(comp, env)
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ExpressionError(
                f"only these functions are allowed: {sorted(_FUNCS)}"
            )
        return _FUNCS[node.func.id](*[_eval(a, env) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, env) for e in node.elts]
    if isinstance(node, ast.IfExp):
        return (_eval(node.body, env) if _eval(node.test, env)
                else _eval(node.orelse, env))
    raise ExpressionError(f"syntax element not allowed: {type(node).__name__}")


def evaluate(expr: str, env: dict) -> bool:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"could not parse: {e.msg}") from None
    return bool(_eval(tree, env))


def check_expression(expr: str) -> str | None:
    """
    Static check against the standard feature set. Returns an error string, or
    None if the expression looks runnable. Used by the spec editor so you find
    out about a typo before a twenty-minute scan, not after.
    """
    probe = {k: 1.0 for k in FEATURE_HELP}
    probe["session"] = "RTH"
    try:
        evaluate(expr, probe)
    except ExpressionError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001 - report anything odd rather than crash
        return f"{type(e).__name__}: {e}"
    return None
