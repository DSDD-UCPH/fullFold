"""Model name, weight directory, and ColabFold feature hooks. No GPU."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from fullFold.config import Config
from fullFold.runner import apply_model_features, make_model_config, resolve_model_dir


def _colabfold(monkeypatch, installed: bool) -> None:
    monkeypatch.setattr('fullFold.af3args.colabfold_installed', lambda: installed)


def _modules(monkeypatch, **submodules) -> None:
    """Stand-ins for ``alphafold3`` and ``alphafold3.model.<name>`` (no AF3 here)."""
    for name in ('alphafold3', 'alphafold3.model'):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    for name, mod in submodules.items():
        monkeypatch.setitem(sys.modules, f'alphafold3.model.{name}', mod)


def _model_module():
    class Global:
        flash_attention_implementation = None
        model = None

    class Cfg:
        def __init__(self):
            self.global_config = Global()
            self.heads = types.SimpleNamespace(
                diffusion=types.SimpleNamespace(eval=types.SimpleNamespace(num_samples=5)))
            self.num_recycles = 10
            self.return_embeddings = self.return_distogram = False

    mod = types.ModuleType('alphafold3.model.model')
    mod.Model = types.SimpleNamespace(Config=Cfg)
    return mod


def test_model_config_is_chosen_by_the_installed_alphafold3(monkeypatch):
    asked = []

    class Spec:
        def configure(self, config):
            config.global_config.model = 'openbind0'

    registry = types.ModuleType('alphafold3.model.model_registry')
    registry.get = lambda name: (asked.append(name), Spec())[1]
    _modules(monkeypatch, model=_model_module(), model_registry=registry)
    _colabfold(monkeypatch, False)
    with pytest.raises(SystemExit, match='ColabFold'):
        make_model_config(model_name='openfold3')
    _colabfold(monkeypatch, True)
    config = make_model_config(model_name='openbind0')
    assert asked == ['openbind0'] and config.global_config.model == 'openbind0'


def test_colabfold_feature_hooks_apply_only_when_the_spec_declares_them(monkeypatch):
    _colabfold(monkeypatch, True)
    calls = {}
    feats = types.ModuleType('alphafold3.model.model_features')

    def apply(batch, spec, **kw):
        calls.update(batch=batch, spec=spec, kw=kw)
        return 'applied'

    feats.apply = apply
    declares = types.SimpleNamespace(featurise={'drop_atoms': ('OXT',)})
    registry = types.ModuleType('alphafold3.model.model_registry')
    registry.get = lambda name: declares
    _modules(monkeypatch, model_registry=registry, model_features=feats)
    out = apply_model_features({'tok': 1}, object(), 'openfold3', lambda: None, '/w')
    assert out == 'applied' and calls['spec'] is declares and calls['batch'] == {'tok': 1}
    registry.get = lambda name: types.SimpleNamespace(featurise={})
    batch = {'tok': 1}
    assert apply_model_features(batch, object(), 'alphafold3', lambda: None, '/w') is batch


def test_each_model_needs_its_own_weights_blob(tmp_path: Path, monkeypatch):
    _colabfold(monkeypatch, True)
    cfg = Config(model='openfold3', model_dir=tmp_path, download_weights=False)
    (tmp_path / 'af3.bin.zst').write_bytes(b'x')  # AlphaFold 3's blob is not OpenFold's
    with pytest.raises(SystemExit, match='openfold3.bin.zst'):
        resolve_model_dir(cfg)
    for model, blob in (('of3', 'openfold3.bin.zst'), ('openbind0', 'openbind0.bin.zst')):
        (tmp_path / blob).write_bytes(b'x')
        cfg = Config(model=model, model_dir=tmp_path, download_weights=False)
        assert resolve_model_dir(cfg) == tmp_path, model
    assert resolve_model_dir(Config(model='alphafold3', model_dir=tmp_path)) == tmp_path


def test_missing_weights_download_through_the_colabfold_loader(tmp_path: Path, monkeypatch):
    _colabfold(monkeypatch, True)
    calls = {}

    def ensure_weights(model_name, model_dir=None, download=True, precision='fp32'):
        calls['args'] = (model_name, model_dir, download, precision)
        return calls['found']

    weights = types.ModuleType('alphafold3.model.weights')
    weights.ensure_weights = ensure_weights
    _modules(monkeypatch, weights=weights)
    # a folder that already holds other weights is left alone: download elsewhere
    (tmp_path / 'af3.bin.zst').write_bytes(b'x')
    calls['found'] = '/cache/openfold3'
    cfg = Config(model='openfold3', model_dir=tmp_path, download_weights=True,
                 weights_precision='fp16')
    assert resolve_model_dir(cfg) == Path('/cache/openfold3')
    assert calls['args'] == ('openfold3', None, True, 'fp16')
    # an empty folder is filled
    empty = tmp_path / 'empty'
    empty.mkdir()
    calls['found'] = str(empty)
    cfg = Config(model='openbind', model_dir=empty, download_weights=True)
    assert resolve_model_dir(cfg) == empty
    assert calls['args'][:2] == ('openbind0', str(empty))
