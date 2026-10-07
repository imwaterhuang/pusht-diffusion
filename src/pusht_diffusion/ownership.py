"""Delivery-time ownership check; expected to fail once the learner writes code."""

import ast
from pathlib import Path


def check_ownership() -> dict:
    root = Path(__file__).parent / 'learner'
    checked = []
    for path in sorted(root.glob('*.py')):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                if (
                    len(body) != 1
                    or not isinstance(body[0], ast.Raise)
                    or not isinstance(body[0].exc, ast.Call)
                    or not isinstance(body[0].exc.func, ast.Name)
                    or body[0].exc.func.id != 'NotImplementedError'
                ):
                    raise ValueError(f'Learner implementation present: {path.name}:{node.name}')
                checked.append(f'{path.name}:{node.lineno}:{node.name}')
    return {
        'status': 'protected_skeletons_only',
        'functions': checked,
        'interpretation': 'Delivery boundary only. Learner is expected to replace these bodies later.',
    }
