"""Fingerprint, mode default, memory fraction, and hoist install wiring."""

from __future__ import annotations

import ast
from pathlib import Path

from fullFold.config import (
    FALLBACK_MEM_FRACTION, Config, mem_fraction, worker_environ,
)
from fullFold.hoists.fingerprint import (
    TARGETS, _find, canonical_dump, collect_hashes, digests_for, evaluate_hoists,
    hash_source, hoist_key, load_known, notices,
)


DH_SRC = '''
def random_augmentation(rng_key, positions, mask):
    return positions

def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
    return t

def sample(denoising_step, batch, key, config):
    return {}

class DiffusionHead:
    def _conditioning(self, batch, embeddings, noise_level, use_conditioning):
        return None, None
    def __call__(self, positions_noisy, noise_level, batch, embeddings, use_conditioning):
        return positions_noisy
'''

ACA_SRC = '''
def _per_atom_conditioning(config, batch, name):
    return None, None

def atom_cross_att_encoder(token_atoms_act, trunk_single_cond, trunk_pair_cond, config, global_config, batch, name):
    return None

def atom_cross_att_decoder(token_act, enc, config, global_config, batch, name):
    return None
'''

DT_SRC = '''
class CrossAttTransformer:
    def __call__(self, queries_act, queries_mask, queries_to_keys, keys_mask, queries_single_cond, keys_single_cond, pair_cond):
        return queries_act
'''

MODEL_SRC = '''
class Model:
    def _sample_diffusion(self, batch, embeddings, *, sample_config):
        return {}
'''

NATIVE_DH = '''
class DiffusionHead:
    def __call__(self, positions_noisy, noise_level, batch, embeddings, use_conditioning, conditioning_only=False):
        return positions_noisy
def sample(denoising_step, batch, key, config):
    noise_level_prev = 0
    step_noise = 0
    return {}
'''

NATIVE_ACA = '''
def atom_cross_att_encoder(token_atoms_act, trunk_single_cond, trunk_pair_cond, config, global_config, batch, name, conditioning_only=False):
    return None
'''


def _tree(tmp_path: Path, *, colab=False, dh=DH_SRC, aca=ACA_SRC, dt=DT_SRC, model=MODEL_SRC) -> Path:
    root = tmp_path / 'alphafold3'
    net = root / 'model' / 'network'
    net.mkdir(parents=True)
    (root / 'model').mkdir(exist_ok=True)
    (net / 'diffusion_head.py').write_text(dh)
    (net / 'atom_cross_attention.py').write_text(aca)
    (net / 'diffusion_transformer.py').write_text(dt)
    (root / 'model' / 'model.py').write_text(model)
    if colab:
        (root / 'constants').mkdir()
        (root / 'constants' / 'decoded_ccd.py').write_text('#')
        (root / 'model' / 'model_registry.py').write_text('#')
    return root


def _known_with(root: Path, pick=lambda digests: digests[0]) -> dict:
    """A fingerprints.json whose digest for every target is ``pick(all its forms)``."""
    hashes = {}
    for key, (rel, qual) in TARGETS.items():
        tree = ast.parse((root / rel).read_text())
        hashes[key] = [pick(digests_for(_find(tree, qual)))]
    return {'target_version': '3.0.4', 'hashes': hashes}


def test_fingerprints_json_covers_targets():
    known = load_known()
    assert known['target_version'] == '3.0.4'
    assert set(known['hashes']) == set(TARGETS)
    for vals in known['hashes'].values():
        assert vals and all(len(v) == 64 for v in vals)


PINNED_SRC = '''
def f(a, b=1, *args, k=None, **kw):
    """doc"""
    x = [i * 2 for i in range(a) if i]
    return f"{a}:{b!r}", x, None, 0.5, b"z", ...
'''
# Identical on CPython 3.10, 3.12 and 3.14 (checked). If this changes, every
# fingerprints.json written before the change stops matching.
PINNED_DIGEST = '84cbf0f01f73340bbd3604cd2ca529816d14884a827f3fdff2873b726d1bb601'


def test_digest_ignores_docstrings_and_python_version():
    a = 'def sample():\n    """doc"""\n    return 1\n'
    assert hash_source(a, 'sample') == hash_source('def sample():\n    return 1\n', 'sample')
    assert hash_source(a, 'sample') != hash_source('def sample():\n    return 2\n', 'sample')
    assert hash_source(PINNED_SRC, 'f') == PINNED_DIGEST
    node = ast.parse('def f(a):\n    return a\n').body[0]
    text = canonical_dump(node)
    assert text == (
        "FunctionDef(name='f',args=arguments(args=[arg(arg='a')]),"
        "body=[Return(value=Name(id='a',ctx=Load()))])")
    node.type_params = []  # what newer interpreters add: empty or None, so ignored
    node.future_field = None
    assert canonical_dump(node) == text
    forms = digests_for(ast.parse(PINNED_SRC).body[0])
    assert forms[0] == PINNED_DIGEST  # canonical first, then the ast.dump forms
    assert len(forms) == len(set(forms)) >= 2 and all(len(d) == 64 for d in forms)


def test_fingerprints_gate_each_hoist(tmp_path: Path):
    root = _tree(tmp_path)

    def evaluate(known=None, mode='default'):
        return evaluate_hoists(
            root=root, known=known or _known_with(root), colabfold=False, mode=mode)

    st = evaluate()
    for name in ('cond_share', 'atom_cond_hoist', 'diffusion_hoist'):
        assert st[name].install and st[name].active, name
    assert hoist_key(st) == 'cs+ach+dh' and notices(st) == []
    # any recorded form of a digest matches: canonical, ast.dump, or the 3.12-style dump
    for pick in (lambda d: d[1], lambda d: d[-1]):
        assert hoist_key(evaluate(_known_with(root, pick))) == 'cs+ach+dh'
    off = evaluate(mode='off')
    assert hoist_key(off) == 'none' and notices(off) == []
    # a changed function matches no form: that hoist is off, the others stay on
    known = _known_with(root)
    dh = root / 'model' / 'network' / 'diffusion_head.py'
    dh.write_text(DH_SRC.replace('return {}', 'return {"x": 1}'))
    st = evaluate(known)
    assert not st['cond_share'].install
    assert 'differs in diffusion_head.sample' in st['cond_share'].reason
    assert any('noise sharing disabled' in line for line in notices(st))
    assert st['diffusion_hoist'].install
    dh.write_text(DH_SRC)
    # the atom hoist needs the pair-conditioning hoist, whether that one is missing or changed
    model = root / 'model' / 'model.py'
    model.write_text(MODEL_SRC.replace('return {}', 'return 1'))
    st = evaluate(known)
    assert not st['diffusion_hoist'].install and not st['atom_cond_hoist'].install
    model.unlink()
    st = evaluate(known)
    assert not st['diffusion_hoist'].install and not st['atom_cond_hoist'].install
    assert 'needs pair-conditioning hoist' in st['atom_cond_hoist'].reason


def test_colabfold_native_skips_vendored(tmp_path: Path):
    root = _tree(tmp_path, colab=True, dh=NATIVE_DH, aca=NATIVE_ACA)
    st = evaluate_hoists(
        root=root, known={'hashes': {}}, colabfold=True, mode='default', af3_faster=False)
    for name in ('cond_share', 'diffusion_hoist', 'atom_cond_hoist'):
        assert not st[name].install and st[name].active, name
    assert not st['hoist_logits'].install
    st2 = evaluate_hoists(
        root=root, known={'hashes': {}}, colabfold=True, mode='default', af3_faster=True)
    assert st2['hoist_logits'].install and st2['hoist_logits'].active
    assert hoist_key(st2) == 'cs+ach+dh+hl'


def test_collect_hashes_records_every_form_and_parses_each_file_once(
        tmp_path: Path, monkeypatch):
    from fullFold.hoists import fingerprint
    root = _tree(tmp_path)
    parsed = []
    real = ast.parse
    monkeypatch.setattr(
        fingerprint.ast, 'parse',
        lambda src, *a, **k: (parsed.append(src), real(src, *a, **k))[1])
    got = collect_hashes(root)
    assert set(got) == set(TARGETS)
    assert all(len(v) >= 2 and all(len(d) == 64 for d in v) for v in got.values())
    assert len(parsed) == 4  # diffusion_head, atom_cross_attention, diffusion_transformer, model


def test_memory_fraction_reserves_640mib(tmp_path: Path):
    gib = 1024 ** 3
    assert mem_fraction(0) == FALLBACK_MEM_FRACTION
    assert mem_fraction(8 * gib) == 0.9219
    assert mem_fraction(24 * gib) == 0.974
    assert mem_fraction(80 * gib) == 0.9922
    assert mem_fraction(640 * 1024 * 1024) == FALLBACK_MEM_FRACTION
    auto = worker_environ(Config(cache_dir=tmp_path), '0', memory_bytes=8 * gib)
    assert auto['XLA_CLIENT_MEM_FRACTION'] == '0.9219'
    over = worker_environ(
        Config(cache_dir=tmp_path, xla_mem_fraction=0.5), '0', memory_bytes=8 * gib)
    assert over['XLA_CLIENT_MEM_FRACTION'] == '0.5'
    unknown = worker_environ(Config(cache_dir=tmp_path), '0')
    assert unknown['XLA_CLIENT_MEM_FRACTION'] == str(FALLBACK_MEM_FRACTION)


def test_install_hoists(monkeypatch):
    from fullFold import runner
    assert runner.install_hoists(Config(mode='off')) == []
    assert runner.install_hoists(Config(mode='fast')) == []
    calls = []

    def status(name, active=True):
        return type('S', (), {'install': active, 'active': active,
                              'reason': 'fingerprint', 'name': name})()

    monkeypatch.setattr('fullFold.hoists.evaluate_hoists', lambda **k: {
        'cond_share': status('cond_share'), 'atom_cond_hoist': status('atom_cond_hoist'),
        'diffusion_hoist': status('diffusion_hoist'),
        'hoist_logits': status('hoist_logits', False)})

    def fake_install(names):
        for n in names:
            calls.append(n)
            if n == 'atom_cond_hoist':
                raise RuntimeError('boom')
        return list(names)

    monkeypatch.setattr('fullFold.hoists.install', fake_install)
    monkeypatch.setattr('fullFold.hoists.notices', lambda s: [])
    monkeypatch.setattr('fullFold.af3args.af3_faster_installed', lambda: False)
    # installed in the documented order; a failure stops there and falls back to stock
    assert runner.install_hoists(Config(mode='default')) == []
    assert calls == ['cond_share', 'atom_cond_hoist']
