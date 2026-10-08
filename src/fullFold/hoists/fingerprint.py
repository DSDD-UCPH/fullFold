"""Enable a hoist only when the installed AlphaFold 3 source matches v3.0.4.

Reads ``alphafold3`` files through ``importlib.util.find_spec`` so the
parent process never imports AlphaFold or JAX. Hashes are AST dumps with
docstrings stripped; comment and formatting edits do not count.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

TARGET_VERSION = '3.0.4'
SHORT = {
    'cond_share': 'cs',
    'atom_cond_hoist': 'ach',
    'diffusion_hoist': 'dh',
    'hoist_logits': 'hl',
}
INSTALL_ORDER = ('cond_share', 'atom_cond_hoist', 'diffusion_hoist')

# relative to alphafold3 package root, then dotted path inside the file
TARGETS = {
    'diffusion_head.sample': (
        'model/network/diffusion_head.py', 'sample'),
    'diffusion_head.random_augmentation': (
        'model/network/diffusion_head.py', 'random_augmentation'),
    'diffusion_head.noise_schedule': (
        'model/network/diffusion_head.py', 'noise_schedule'),
    'diffusion_head.DiffusionHead.__call__': (
        'model/network/diffusion_head.py', 'DiffusionHead.__call__'),
    'diffusion_head.DiffusionHead._conditioning': (
        'model/network/diffusion_head.py', 'DiffusionHead._conditioning'),
    'atom_cross_attention.atom_cross_att_encoder': (
        'model/network/atom_cross_attention.py', 'atom_cross_att_encoder'),
    'atom_cross_attention.atom_cross_att_decoder': (
        'model/network/atom_cross_attention.py', 'atom_cross_att_decoder'),
    'atom_cross_attention._per_atom_conditioning': (
        'model/network/atom_cross_attention.py', '_per_atom_conditioning'),
    'diffusion_transformer.CrossAttTransformer.__call__': (
        'model/network/diffusion_transformer.py',
        'CrossAttTransformer.__call__'),
    'model.Model._sample_diffusion': (
        'model/model.py', 'Model._sample_diffusion'),
}

HOIST_TARGETS = {
    'cond_share': (
        'diffusion_head.sample',
        'diffusion_head.random_augmentation',
        'diffusion_head.noise_schedule',
    ),
    'atom_cond_hoist': (
        'atom_cross_attention.atom_cross_att_encoder',
        'atom_cross_attention.atom_cross_att_decoder',
        'atom_cross_attention._per_atom_conditioning',
        'diffusion_transformer.CrossAttTransformer.__call__',
    ),
    'diffusion_hoist': (
        'diffusion_head.DiffusionHead.__call__',
        'diffusion_head.DiffusionHead._conditioning',
        'model.Model._sample_diffusion',
    ),
}


@dataclasses.dataclass(frozen=True)
class HoistStatus:
    name: str
    install: bool
    active: bool
    reason: str


def fingerprints_path() -> Path:
    return Path(__file__).with_name('fingerprints.json')


def load_known(path: Path | None = None) -> dict:
    p = path or fingerprints_path()
    return json.loads(p.read_text())


def alphafold3_root() -> Path | None:
    try:
        spec = importlib.util.find_spec('alphafold3')
    except (ImportError, ValueError, AttributeError):
        return None
    locations = getattr(spec, 'submodule_search_locations', None) if spec else None
    if not locations:
        return None
    return Path(next(iter(locations)))


def colabfold_tree(root: Path | None = None) -> bool:
    root = root if root is not None else alphafold3_root()
    if root is None:
        return False
    return (
        (root / 'constants' / 'decoded_ccd.py').is_file()
        and (root / 'model' / 'model_registry.py').is_file()
    )


def _strip_docstring(node: ast.AST) -> ast.AST:
    node = copy.deepcopy(node)
    body = getattr(node, 'body', None)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        node.body = body[1:]
    return node


def _find(tree: ast.AST, qualname: str) -> ast.AST | None:
    parts = qualname.split('.')
    cur: ast.AST | None = tree
    for part in parts:
        if cur is None:
            return None
        body = getattr(cur, 'body', None)
        if body is None:
            return None
        nxt = None
        for child in body:
            name = getattr(child, 'name', None)
            if name == part:
                nxt = child
                break
        cur = nxt
    return cur


def hash_source(source: str, qualname: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    node = _find(tree, qualname)
    if node is None:
        return None
    dumped = ast.dump(_strip_docstring(node), include_attributes=False)
    return hashlib.sha256(dumped.encode()).hexdigest()


def hash_file(path: Path, qualname: str) -> str | None:
    try:
        return hash_source(path.read_text(), qualname)
    except OSError:
        return None


def _arg_names(node: ast.AST) -> list[str]:
    args = getattr(node, 'args', None)
    if args is None:
        return []
    names = [a.arg for a in getattr(args, 'posonlyargs', [])]
    names += [a.arg for a in args.args]
    names += [a.arg for a in getattr(args, 'kwonlyargs', [])]
    return names


def native_features(root: Path | None = None) -> dict[str, bool]:
    """Detect ColabFold-style native hoists from source, without importing."""
    root = root if root is not None else alphafold3_root()
    empty = {
        'conditioning_only': False,
        'atom_cond': False,
        'cond_share': False,
    }
    if root is None:
        return empty
    dh = root / 'model' / 'network' / 'diffusion_head.py'
    aca = root / 'model' / 'network' / 'atom_cross_attention.py'
    out = dict(empty)
    if dh.is_file():
        try:
            tree = ast.parse(dh.read_text())
        except (OSError, SyntaxError):
            tree = None
        if tree is not None:
            call = _find(tree, 'DiffusionHead.__call__')
            if call is not None:
                out['conditioning_only'] = 'conditioning_only' in _arg_names(call)
            sample = _find(tree, 'sample')
            if sample is not None:
                text = ast.dump(sample)
                out['cond_share'] = (
                    'noise_level_prev' in text and 'step_noise' in text)
    if aca.is_file():
        try:
            tree = ast.parse(aca.read_text())
        except (OSError, SyntaxError):
            tree = None
        if tree is not None:
            enc = _find(tree, 'atom_cross_att_encoder')
            if enc is not None:
                out['atom_cond'] = 'conditioning_only' in _arg_names(enc)
    return out


def current_hashes(root: Path | None = None) -> dict[str, str | None]:
    root = root if root is not None else alphafold3_root()
    out: dict[str, str | None] = {}
    if root is None:
        return {k: None for k in TARGETS}
    for key, (rel, qual) in TARGETS.items():
        out[key] = hash_file(root / rel, qual)
    return out


def _match(key: str, digest: str | None, known: dict) -> bool:
    if not digest:
        return False
    allowed = (known.get('hashes') or {}).get(key) or []
    return digest in allowed


def hoist_key(statuses: dict[str, HoistStatus]) -> str:
    parts = [
        SHORT[name] for name in (*INSTALL_ORDER, 'hoist_logits')
        if name in statuses and statuses[name].active
    ]
    return '+'.join(parts) if parts else 'none'


def notices(statuses: dict[str, HoistStatus]) -> list[str]:
    lines = []
    labels = {
        'cond_share': 'noise sharing',
        'atom_cond_hoist': 'atom-conditioning hoist',
        'diffusion_hoist': 'pair-conditioning hoist',
        'hoist_logits': 'pair-logit hoist',
    }
    for name in (*INSTALL_ORDER, 'hoist_logits'):
        st = statuses.get(name)
        if st is None or st.active:
            continue
        if st.reason in (
            'not_colabfold', 'no_af3_faster', 'mode', 'no alphafold3',
        ):
            continue
        label = labels.get(name, name)
        lines.append(
            f'fullFold: {label} disabled - {st.reason} '
            f'(hoists target {TARGET_VERSION}). Results are stock for that part.'
        )
    return lines


def evaluate_hoists(
    *,
    root: Path | None = None,
    known: dict | None = None,
    colabfold: bool | None = None,
    af3_faster: bool = False,
    mode: str = 'default',
) -> dict[str, HoistStatus]:
    """Decide which hoists to install and which are already native."""
    if mode != 'default':
        return {
            name: HoistStatus(name, False, False, 'mode')
            for name in (*INSTALL_ORDER, 'hoist_logits')
        }
    root = root if root is not None else alphafold3_root()
    known = known if known is not None else (
        load_known() if fingerprints_path().is_file() else {'hashes': {}})
    if colabfold is None:
        colabfold = colabfold_tree(root)
    if root is None:
        out = {
            name: HoistStatus(name, False, False, 'no alphafold3')
            for name in (*INSTALL_ORDER, 'hoist_logits')
        }
        return out
    native = native_features(root) if colabfold else {
        'conditioning_only': False, 'atom_cond': False, 'cond_share': False,
    }
    hashes = current_hashes(root)

    def fingerprint_ok(name: str) -> tuple[bool, str]:
        missing = []
        for key in HOIST_TARGETS[name]:
            digest = hashes.get(key)
            if digest is None:
                missing.append(f'{key} missing')
            elif not _match(key, digest, known):
                missing.append(f'differs in {key}')
        if missing:
            return False, '; '.join(missing)
        return True, 'fingerprint'

    out: dict[str, HoistStatus] = {}
    if colabfold:
        out['cond_share'] = HoistStatus(
            'cond_share', False, native['cond_share'],
            'native' if native['cond_share'] else 'no native noise sharing',
        )
        out['diffusion_hoist'] = HoistStatus(
            'diffusion_hoist', False, native['conditioning_only'],
            'native' if native['conditioning_only']
            else 'no native pair conditioning',
        )
        atom_ok = native['atom_cond'] and native['conditioning_only']
        out['atom_cond_hoist'] = HoistStatus(
            'atom_cond_hoist', False, atom_ok,
            'native' if atom_ok else 'no native atom conditioning',
        )
        out['hoist_logits'] = HoistStatus(
            'hoist_logits', af3_faster, af3_faster,
            'af3_faster' if af3_faster else 'no_af3_faster',
        )
        return out

    dh_ok, dh_reason = fingerprint_ok('diffusion_hoist')
    out['diffusion_hoist'] = HoistStatus(
        'diffusion_hoist', dh_ok, dh_ok, dh_reason)
    cs_ok, cs_reason = fingerprint_ok('cond_share')
    out['cond_share'] = HoistStatus(
        'cond_share', cs_ok, cs_ok, cs_reason)
    ach_ok, ach_reason = fingerprint_ok('atom_cond_hoist')
    if ach_ok and not dh_ok:
        ach_ok, ach_reason = False, 'needs pair-conditioning hoist'
    out['atom_cond_hoist'] = HoistStatus(
        'atom_cond_hoist', ach_ok, ach_ok, ach_reason)
    out['hoist_logits'] = HoistStatus(
        'hoist_logits', False, False, 'not_colabfold')
    return out


def collect_hashes(root: Path) -> dict[str, str]:
    found = current_hashes(root)
    missing = [k for k, v in found.items() if not v]
    if missing:
        raise RuntimeError(f'could not hash: {missing}')
    return found  # type: ignore[return-value]
