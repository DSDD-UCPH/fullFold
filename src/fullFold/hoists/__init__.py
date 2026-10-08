"""Exact-math diffusion hoists targeting AlphaFold 3 v3.0.4."""

from __future__ import annotations

from fullFold.hoists.fingerprint import (
    TARGET_VERSION,
    evaluate_hoists,
    hoist_key,
    notices,
)

__all__ = [
    'TARGET_VERSION',
    'evaluate_hoists',
    'hoist_key',
    'install',
    'notices',
]


def install(names: list[str] | tuple[str, ...]) -> list[str]:
    """Install vendored hoists in TREE_LEVERS order. Lazy AF3/JAX imports."""
    wanted = set(names)
    done: list[str] = []
    if 'cond_share' in wanted:
        from fullFold.hoists import cond_share
        cond_share.install()
        done.append('cond_share')
    if 'atom_cond_hoist' in wanted:
        from fullFold.hoists import atom_cond_hoist
        atom_cond_hoist.install()
        done.append('atom_cond_hoist')
    if 'diffusion_hoist' in wanted:
        from fullFold.hoists import diffusion_hoist
        diffusion_hoist.install()
        done.append('diffusion_hoist')
    return done
