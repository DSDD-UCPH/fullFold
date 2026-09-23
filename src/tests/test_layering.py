"""Import graph: no cycles; jax only at module level in worker.py."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'fullFold'
MODS = (
    'config', 'tokens', 'jobs', 'scheduling', 'benchmark',
    'runner', 'worker', 'engine', 'templates', 'cli', 'banner',
)


def _imports(name: str) -> list[tuple[str, bool]]:
    """Return (module, is_toplevel) for fullFold.* and jax imports."""
    tree = ast.parse((ROOT / f'{name}.py').read_text())
    out = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0

        def visit_FunctionDef(self, node):
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Import(self, node):
            for a in node.names:
                out.append((a.name, self.depth == 0))

        def visit_ImportFrom(self, node):
            mod = node.module or ''
            out.append((mod, self.depth == 0))

    V().visit(tree)
    return out


def test_no_cycles():
    graph = {m: set() for m in MODS}
    for m in MODS:
        for name, _ in _imports(m):
            if name.startswith('fullFold.'):
                graph[m].add(name.split('.')[1])
            elif name in MODS:
                graph[m].add(name)

    def dfs(n, stack):
        if n in stack:
            raise AssertionError(f'cycle: {" -> ".join(stack + [n])}')
        for c in graph.get(n, ()):
            dfs(c, stack + [n])

    for m in MODS:
        dfs(m, [])


def test_jax_import_sites():
    toplevel_jax = []
    for m in MODS:
        for name, top in _imports(m):
            if name == 'jax' or name.startswith('jax.'):
                if top:
                    toplevel_jax.append(m)
    assert toplevel_jax == ['worker'] or toplevel_jax == []
    # worker is allowed to import jax at module level; currently it is lazy.
    allowed_lazy = {'worker', 'tokens', 'benchmark', 'runner'}
    for m in MODS:
        for name, top in _imports(m):
            if (name == 'jax' or name.startswith('jax.')) and m not in allowed_lazy:
                raise AssertionError(f'{m} imports jax')


def test_scheduling_is_pure():
    for name, _ in _imports('scheduling'):
        assert not name.startswith('fullFold') or name == 'fullFold.scheduling'
        assert 'jax' not in name
        assert 'alphafold' not in name
