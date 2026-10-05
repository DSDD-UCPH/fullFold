"""Model name, weight directory, and ColabFold feature hooks. No GPU."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from fullFold.config import Config
from fullFold.runner import (
    apply_model_features, make_model_config, resolve_model_dir,
)


def _install_fake_model(monkeypatch):
    class Global:
        def __init__(self):
            self.flash_attention_implementation = None
            self.model = None

    class Eval:
        num_samples = 5

    class Heads:
        def __init__(self):
            self.diffusion = types.SimpleNamespace(eval=Eval())

    class Cfg:
        def __init__(self):
            self.global_config = Global()
            self.heads = Heads()
            self.num_recycles = 10
            self.return_embeddings = False
            self.return_distogram = False

    model_mod = types.ModuleType('alphafold3.model.model')
    model_mod.Model = types.SimpleNamespace(Config=Cfg)
    pkg = types.ModuleType('alphafold3')
    pkg.__path__ = []
    model_pkg = types.ModuleType('alphafold3.model')
    model_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, 'alphafold3', pkg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model', model_pkg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model.model', model_mod)
    return model_mod


def test_orig_refuses_openfold3(monkeypatch):
    _install_fake_model(monkeypatch)
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: False)
    with pytest.raises(SystemExit, match='ColabFold'):
        make_model_config(model_name='openfold3')


def test_colabfold_configures_openbind0(monkeypatch):
    _install_fake_model(monkeypatch)
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: True)
    seen = {}

    class Spec:
        def configure(self, config):
            seen['name'] = config
            config.global_config.model = 'openbind0'

    reg = types.ModuleType('alphafold3.model.model_registry')
    reg.get = lambda name: Spec()
    seen['asked'] = []
    real_get = reg.get

    def get(name):
        seen['asked'].append(name)
        return real_get(name)

    reg.get = get
    monkeypatch.setitem(sys.modules, 'alphafold3.model.model_registry', reg)
    config = make_model_config(model_name='openbind0')
    assert seen['asked'] == ['openbind0']
    assert config.global_config.model == 'openbind0'


def test_features_applied_when_spec_declares_them(monkeypatch):
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: True)
    spec = types.SimpleNamespace(featurise={'drop_atoms': ('OXT',)})
    reg = types.ModuleType('alphafold3.model.model_registry')
    reg.get = lambda name: spec
    feats = types.ModuleType('alphafold3.model.model_features')
    calls = {}

    def apply(batch, got_spec, **kw):
        calls['batch'] = batch
        calls['spec'] = got_spec
        calls['kw'] = kw
        return 'applied'

    feats.apply = apply
    pkg = types.ModuleType('alphafold3')
    pkg.__path__ = []
    model_pkg = types.ModuleType('alphafold3.model')
    model_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, 'alphafold3', pkg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model', model_pkg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model.model_registry', reg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model.model_features', feats)
    out = apply_model_features({'tok': 1}, object(), 'openfold3', lambda: None, '/w')
    assert out == 'applied'
    assert calls['spec'] is spec
    assert calls['batch'] == {'tok': 1}


def test_empty_featurise_spec_is_unchanged(monkeypatch):
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: True)
    spec = types.SimpleNamespace(featurise={})
    reg = types.ModuleType('alphafold3.model.model_registry')
    reg.get = lambda name: spec
    pkg = types.ModuleType('alphafold3')
    pkg.__path__ = []
    model_pkg = types.ModuleType('alphafold3.model')
    model_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, 'alphafold3', pkg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model', model_pkg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model.model_registry', reg)
    batch = {'tok': 1}
    assert apply_model_features(batch, object(), 'alphafold3', lambda: None, '/w') is batch


def test_af3_blob_is_not_openfold_weights(tmp_path: Path, monkeypatch):
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: True)
    (tmp_path / 'af3.bin.zst').write_bytes(b'x')
    cfg = Config(model='openfold3', model_dir=tmp_path, download_weights=False)
    with pytest.raises(SystemExit, match='openfold3.bin.zst'):
        resolve_model_dir(cfg)


def test_openfold_blob_is_accepted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: True)
    (tmp_path / 'openfold3.bin.zst').write_bytes(b'x')
    cfg = Config(model='of3', model_dir=tmp_path, download_weights=False)
    assert resolve_model_dir(cfg) == tmp_path


def test_openbind_blob_is_accepted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: True)
    (tmp_path / 'openbind0.bin.zst').write_bytes(b'x')
    cfg = Config(model='openbind0', model_dir=tmp_path, download_weights=False)
    assert resolve_model_dir(cfg) == tmp_path


def test_alphafold3_dir_is_not_resolved(tmp_path: Path):
    cfg = Config(model='alphafold3', model_dir=tmp_path)
    assert resolve_model_dir(cfg) == tmp_path


def _install_weights(monkeypatch, found: str) -> dict:
    calls = {}

    def ensure_weights(model_name, model_dir=None, download=True, precision='fp32'):
        calls['args'] = (model_name, model_dir, download, precision)
        return found

    weights = types.ModuleType('alphafold3.model.weights')
    weights.ensure_weights = ensure_weights
    pkg = types.ModuleType('alphafold3')
    pkg.__path__ = []
    model_pkg = types.ModuleType('alphafold3.model')
    model_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, 'alphafold3', pkg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model', model_pkg)
    monkeypatch.setitem(sys.modules, 'alphafold3.model.weights', weights)
    return calls


def test_mixed_af3_dir_downloads_outside_it(tmp_path: Path, monkeypatch):
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: True)
    (tmp_path / 'af3.bin.zst').write_bytes(b'x')
    calls = _install_weights(monkeypatch, '/cache/openfold3')
    cfg = Config(
        model='openfold3', model_dir=tmp_path, download_weights=True,
        weights_precision='fp16')
    assert resolve_model_dir(cfg) == Path('/cache/openfold3')
    assert calls['args'] == ('openfold3', None, True, 'fp16')


def test_empty_dir_downloads_into_model_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: True)
    calls = _install_weights(monkeypatch, str(tmp_path))
    cfg = Config(model='openbind', model_dir=tmp_path, download_weights=True)
    assert resolve_model_dir(cfg) == tmp_path
    assert calls['args'][0] == 'openbind0'
    assert calls['args'][1] == str(tmp_path)
