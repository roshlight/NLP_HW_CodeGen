"""Bounded AST interpreter for validation diagnostics; never exec/eval generated code.

Only a small arithmetic Python subset is accepted. Unsupported references are
reported in coverage, not silently counted as ground truth. No training targets
are created or changed by this evaluator.
"""

import ast
import math
import operator


class Unsupported(ValueError):
    pass


class Returned(Exception):
    def __init__(self, value):
        self.value = value


BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
FUN = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "round": round,
    "int": int,
    "float": float,
}
MATH = {name: getattr(math, name) for name in ("sqrt", "ceil", "floor", "fabs")}


class Arithmetic:
    def __init__(self):
        self.env, self.steps = {}, 0

    def tick(self):
        self.steps += 1
        if self.steps > 2000:
            raise Unsupported("step budget")

    @staticmethod
    def bounded(value):
        if isinstance(value, (list, tuple, range)):
            if len(value) > 256:
                raise Unsupported("container budget")
            return value
        if (
            not isinstance(value, (int, float, bool))
            or not math.isfinite(value)
            or abs(value) > 1e100
        ):
            raise Unsupported("numeric budget")
        return value

    def expr(self, n):
        self.tick()
        if isinstance(n, ast.Constant):
            value = n.value
        elif isinstance(n, ast.Name):
            value = self.env[n.id]
        elif isinstance(n, (ast.List, ast.Tuple)):
            if len(n.elts) > 256:
                raise Unsupported("container budget")
            value = [self.expr(e) for e in n.elts]
        elif isinstance(n, ast.BinOp):
            a, b = self.expr(n.left), self.expr(n.right)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise Unsupported("non-numeric operation")
            if isinstance(n.op, ast.Pow):
                if abs(b) > 16:
                    raise Unsupported("power budget")
                value = a**b
            else:
                value = BIN[type(n.op)](a, b)
        elif isinstance(n, ast.UnaryOp):
            fn = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}[
                type(n.op)
            ]
            value = fn(self.expr(n.operand))
        elif isinstance(n, ast.Compare):
            values = [self.expr(n.left)] + [self.expr(x) for x in n.comparators]
            value = all(CMP[type(op)](a, b) for a, b, op in zip(values, values[1:], n.ops))
        elif isinstance(n, ast.BoolOp):
            value = self.expr(n.values[0])
            for child in n.values[1:]:
                if isinstance(n.op, ast.And) and not value:
                    break
                if isinstance(n.op, ast.Or) and value:
                    break
                value = self.expr(child)
        elif isinstance(n, ast.IfExp):
            value = self.expr(n.body if self.expr(n.test) else n.orelse)
        elif isinstance(n, ast.Call) and not n.keywords:
            args = [self.expr(a) for a in n.args]
            if isinstance(n.func, ast.Name):
                if n.func.id == "range":
                    value = range(*args)
                else:
                    value = FUN[n.func.id](*args)
            elif (
                isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "math"
            ):
                value = MATH[n.func.attr](*args)
            else:
                raise Unsupported("call")
        else:
            raise Unsupported(type(n).__name__)
        return self.bounded(value)

    def assign(self, target, value):
        if not isinstance(target, ast.Name):
            raise Unsupported("assignment")
        self.env[target.id] = self.bounded(value)

    def statements(self, statements):
        for n in statements:
            self.tick()
            if isinstance(n, ast.Assign):
                value = self.expr(n.value)
                for t in n.targets:
                    self.assign(t, value)
            elif isinstance(n, ast.AnnAssign):
                self.assign(n.target, self.expr(n.value))
            elif isinstance(n, ast.AugAssign):
                value = self.expr(ast.BinOp(left=n.target, op=n.op, right=n.value))
                self.assign(n.target, value)
            elif isinstance(n, ast.Return):
                raise Returned(self.expr(n.value))
            elif isinstance(n, ast.If):
                self.statements(n.body if self.expr(n.test) else n.orelse)
            elif isinstance(n, ast.For):
                iterable = self.expr(n.iter)
                if not isinstance(iterable, (range, list, tuple)):
                    raise Unsupported("for")
                for value in iterable:
                    self.assign(n.target, value)
                    self.statements(n.body)
                self.statements(n.orelse)
            elif (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            ):
                continue  # ignored docstring
            elif isinstance(n, ast.Import) and all(
                x.name == "math" and x.asname is None for x in n.names
            ):
                continue  # no import execution, only whitelisted math calls above
            else:
                raise Unsupported(type(n).__name__)


def arithmetic_value(code):
    if len(code) > 20000:
        return None
    try:
        tree = ast.parse(code)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
        if len(funcs) != 1 or funcs[0].args.args or funcs[0].decorator_list:
            return None
        # Python module code is never executed; reject unexpected top-level statements.
        for node in tree.body:
            if node is funcs[0]:
                continue
            if not (
                isinstance(node, ast.Import)
                and all(x.name == "math" and not x.asname for x in node.names)
            ):
                return None
        try:
            Arithmetic().statements(funcs[0].body)
        except Returned as result:
            value = result.value
            return float(value) if isinstance(value, (int, float)) else None
    except (Exception, RecursionError):
        return None
    return None


def arithmetic_metrics(predictions, references):
    supported, matched, pred_supported = 0, 0, 0
    for pred, ref in zip(predictions, references):
        a, b = arithmetic_value(pred), arithmetic_value(ref)
        pred_supported += a is not None
        if b is not None:
            supported += 1
            matched += a is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    return {
        "arithmetic_reference_coverage": supported / max(1, len(references)),
        "arithmetic_prediction_coverage": pred_supported / max(1, len(predictions)),
        "arithmetic_agreement_all": matched / max(1, len(references)),
        "arithmetic_agreement_supported": matched / max(1, supported),
    }
