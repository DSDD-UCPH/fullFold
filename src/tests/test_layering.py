"""Import graph: no cycles, a pure planning core, and JAX only where it is expected."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'fullFold'
MODS = (
    'config', 'tokens', 'jobs', 'scheduling', 'planner', 'benchmark',
    'runner', 'worker', 'engine', 'templates', 'cli', 'banner', 'af3args',
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


def test_the_planning_core_is_pure_and_jax_stays_where_it_belongs():
    # scheduling and planner: standard library only (planner may use scheduling)
    allowed = {'scheduling': set(), 'planner': {'fullFold.scheduling'}}
    for name, ok in allowed.items():
        for mod, _ in _imports(name):
            assert mod not in ('jax', 'alphafold3') and not mod.startswith(('jax.', 'alphafold')), name
            assert not mod.startswith('fullFold') or mod in ok, (name, mod)
    # JAX is imported only lazily, and only by the modules that run the model
    allowed_lazy = {'worker', 'tokens', 'benchmark', 'runner'}
    for m in MODS:
        for name, top in _imports(m):
            if name == 'jax' or name.startswith('jax.'):
                assert m in allowed_lazy and not top, f'{m} imports {name}'
