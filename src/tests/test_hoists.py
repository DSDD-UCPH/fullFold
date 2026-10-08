"""Fingerprint, mode default, memory fraction, and hoist install wiring."""

from __future__ import annotations

from pathlib import Path

from fullFold.config import (
    FALLBACK_MEM_FRACTION, Config, mem_fraction, worker_environ,
)
from fullFold.hoists.fingerprint import (
    evaluate_hoists, hash_source, hoist_key, notices,
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


def _known_from(root: Path) -> dict:
    from fullFold.hoists.fingerprint import TARGETS, hash_file
    hashes = {}
    for key, (rel, qual) in TARGETS.items():
        hashes[key] = [hash_file(root / rel, qual)]
    return {'target_version': '3.0.4', 'hashes': hashes}


def test_fingerprints_json_covers_targets():
    from fullFold.hoists.fingerprint import TARGETS, load_known
    known = load_known()
    assert known['target_version'] == '3.0.4'
    assert set(known['hashes']) == set(TARGETS)
    for key, vals in known['hashes'].items():
        assert vals and all(len(v) == 64 for v in vals)


def test_hash_strips_docstring():
    a = 'def sample():\n    """doc"""\n    return 1\n'
    b = 'def sample():\n    return 1\n'
    assert hash_source(a, 'sample') == hash_source(b, 'sample')
    c = 'def sample():\n    return 2\n'
    assert hash_source(a, 'sample') != hash_source(c, 'sample')


def test_fingerprint_match_installs(tmp_path: Path):
    root = _tree(tmp_path)
    known = _known_from(root)
    st = evaluate_hoists(
        root=root, known=known, colabfold=False, mode='default')
    assert st['cond_share'].install and st['cond_share'].active
    assert st['atom_cond_hoist'].install and st['atom_cond_hoist'].active
    assert st['diffusion_hoist'].install and st['diffusion_hoist'].active
    assert hoist_key(st) == 'cs+ach+dh'
    assert notices(st) == []


def test_fingerprint_mismatch_disables(tmp_path: Path):
    root = _tree(tmp_path)
    known = _known_from(root)
    (root / 'model' / 'network' / 'diffusion_head.py').write_text(
        DH_SRC.replace('return {}', 'return {"x": 1}'))
    st = evaluate_hoists(
        root=root, known=known, colabfold=False, mode='default')
    assert not st['cond_share'].install
    assert 'differs in diffusion_head.sample' in st['cond_share'].reason
    lines = notices(st)
    assert any('noise sharing disabled' in x for x in lines)
    assert st['diffusion_hoist'].install


def test_missing_file_disables(tmp_path: Path):
    root = _tree(tmp_path)
    known = _known_from(root)
    (root / 'model' / 'model.py').unlink()
    st = evaluate_hoists(
        root=root, known=known, colabfold=False, mode='default')
    assert not st['diffusion_hoist'].install
    assert not st['atom_cond_hoist'].install
    assert 'needs pair-conditioning hoist' in st['atom_cond_hoist'].reason


def test_atom_cond_requires_diffusion(tmp_path: Path):
    root = _tree(tmp_path)
    known = _known_from(root)
    (root / 'model' / 'model.py').write_text(
        'class Model:\n    def _sample_diffusion(self, batch, embeddings, *, sample_config):\n        return 1\n')
    st = evaluate_hoists(
        root=root, known=known, colabfold=False, mode='default')
    assert not st['diffusion_hoist'].install
    assert not st['atom_cond_hoist'].install


def test_colabfold_native_skips_vendored(tmp_path: Path):
    root = _tree(tmp_path, colab=True, dh=NATIVE_DH, aca=NATIVE_ACA)
    st = evaluate_hoists(
        root=root, known={'hashes': {}}, colabfold=True, mode='default',
        af3_faster=False)
    assert not st['cond_share'].install and st['cond_share'].active
    assert not st['diffusion_hoist'].install and st['diffusion_hoist'].active
    assert not st['atom_cond_hoist'].install and st['atom_cond_hoist'].active
    assert not st['hoist_logits'].install
    st2 = evaluate_hoists(
        root=root, known={'hashes': {}}, colabfold=True, mode='default',
        af3_faster=True)
    assert st2['hoist_logits'].install and st2['hoist_logits'].active
    assert hoist_key(st2) == 'cs+ach+dh+hl'


def test_mode_off_installs_nothing(tmp_path: Path):
    root = _tree(tmp_path)
    known = _known_from(root)
    st = evaluate_hoists(
        root=root, known=known, colabfold=False, mode='off')
    assert hoist_key(st) == 'none'
    assert notices(st) == []


def test_mem_fraction_reserves_640mib():
    gib = 1024 ** 3
    assert mem_fraction(0) == FALLBACK_MEM_FRACTION
    assert mem_fraction(8 * gib) == 0.9219
    assert mem_fraction(24 * gib) == 0.974
    assert mem_fraction(80 * gib) == 0.9922
    assert mem_fraction(640 * 1024 * 1024) == FALLBACK_MEM_FRACTION


def test_worker_environ_mem_fraction(tmp_path: Path):
    auto = worker_environ(
        Config(cache_dir=tmp_path), '0', memory_bytes=8 * 1024 ** 3)
    assert auto['XLA_CLIENT_MEM_FRACTION'] == '0.9219'
    over = worker_environ(
        Config(cache_dir=tmp_path, xla_mem_fraction=0.5), '0',
        memory_bytes=8 * 1024 ** 3)
    assert over['XLA_CLIENT_MEM_FRACTION'] == '0.5'
    unknown = worker_environ(Config(cache_dir=tmp_path), '0')
    assert unknown['XLA_CLIENT_MEM_FRACTION'] == str(FALLBACK_MEM_FRACTION)


def test_install_hoists_noop_off():
    from fullFold.runner import install_hoists
    assert install_hoists(Config(mode='off')) == []
    assert install_hoists(Config(mode='fast')) == []


def test_install_hoists_order_and_fallback(monkeypatch, tmp_path: Path):
    from fullFold import runner
    calls = []

    class Fake:
        def __init__(self, name):
            self.name = name

        def install(self):
            calls.append(self.name)
            if self.name == 'atom_cond_hoist':
                raise RuntimeError('boom')

    monkeypatch.setattr(
        'fullFold.hoists.evaluate_hoists',
        lambda **k: {
            'cond_share': type('S', (), {
                'install': True, 'active': True, 'reason': 'fingerprint',
                'name': 'cond_share'})(),
            'atom_cond_hoist': type('S', (), {
                'install': True, 'active': True, 'reason': 'fingerprint',
                'name': 'atom_cond_hoist'})(),
            'diffusion_hoist': type('S', (), {
                'install': True, 'active': True, 'reason': 'fingerprint',
                'name': 'diffusion_hoist'})(),
            'hoist_logits': type('S', (), {
                'install': False, 'active': False, 'reason': 'not_colabfold',
                'name': 'hoist_logits'})(),
        },
    )

    def fake_install(names):
        for n in names:
            Fake(n).install()
        return list(names)

    monkeypatch.setattr('fullFold.hoists.install', fake_install)
    monkeypatch.setattr('fullFold.hoists.notices', lambda s: [])
    monkeypatch.setattr('fullFold.af3args.af3_faster_installed', lambda: False)
    assert runner.install_hoists(Config(mode='default')) == []
    assert calls == ['cond_share', 'atom_cond_hoist']
