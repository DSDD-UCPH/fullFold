"""CPU equivalence of vendored hoists vs stock AF3. Needs alphafold3 3.0.4."""

from __future__ import annotations

import json

import pytest

pytest.importorskip('alphafold3')
pytest.importorskip('jax')
pytest.importorskip('haiku')

import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def _tiny_json() -> dict:
    return {
        'name': 'fullFold_hoist_eq',
        'modelSeeds': [0],
        'sequences': [
            {
                'protein': {
                    'id': 'A',
                    'sequence': 'ACDEFGHIKL',
                    'unpairedMsa': '>query\nACDEFGHIKL\n',
                    'pairedMsa': '',
                    'templates': [],
                }
            },
            {'ligand': {'id': 'B', 'smiles': 'CCO'}},
        ],
        'dialect': 'alphafold3',
        'version': 3,
    }


def test_fingerprints_match_installed_304():
    import alphafold3
    ver = getattr(alphafold3, '__version__', '') or ''
    try:
        from alphafold3.version import __version__ as ver2
        ver = ver or ver2
    except Exception:
        pass
    if not str(ver).startswith('3.0.4'):
        pytest.skip(f'alphafold3 {ver!r} is not 3.0.4')
    from fullFold.hoists.fingerprint import evaluate_hoists
    st = evaluate_hoists(mode='default', colabfold=False)
    assert st['cond_share'].install
    assert st['atom_cond_hoist'].install
    assert st['diffusion_hoist'].install


def test_hoisted_sample_matches_stock():
    from alphafold3.common import folding_input
    from alphafold3.model import model as af3_model
    from fullFold.hoists import cond_share, atom_cond_hoist, diffusion_hoist
    from fullFold.hoists.fingerprint import evaluate_hoists
    from fullFold.runner import load_ccd

    st = evaluate_hoists(mode='default')
    if not (st['cond_share'].install and st['diffusion_hoist'].install
            and st['atom_cond_hoist'].install):
        pytest.skip(f'hoists not fingerprint-approved: {st}')

    from alphafold3.data import featurisation
    fold = folding_input.Input.from_json(json.dumps(_tiny_json()))
    ccd = load_ccd(fold)
    try:
        batch = featurisation.featurise_input(
            fold_input=fold, ccd=ccd, buckets=[32], verbose=False)[0]
    except Exception as e:
        pytest.skip(f'featurise_input failed: {type(e).__name__}: {e}')

    cfg = af3_model.Model.Config()
    cfg.heads.diffusion.eval.num_samples = 2
    cfg.heads.diffusion.eval.steps = 4
    cfg.num_recycles = 0

    # BatchDict vs Batch: Model._sample_diffusion expects feat_batch.Batch
    from alphafold3.model import feat_batch
    fb = feat_batch.Batch.from_data_dict(batch)
    ntok = int(np.asarray(fb.token_features.mask).shape[0])
    seq_ch = cfg.evoformer.seq_channel
    pair_ch = cfg.evoformer.pair_channel
    # target_feat width follows the installed model; read it from a dry init.
    embeddings = {
        'single': jnp.zeros((ntok, seq_ch), dtype=jnp.float32),
        'pair': jnp.zeros((ntok, ntok, pair_ch), dtype=jnp.float32),
        'target_feat': jnp.zeros((ntok, 447), dtype=jnp.float32),
    }

    def forward(emb):
        m = af3_model.Model(cfg)
        return m._sample_diffusion(
            fb, emb, sample_config=cfg.heads.diffusion.eval)

    stock = hk.transform(forward)
    rng = jax.random.PRNGKey(0)
    try:
        params = stock.init(rng, embeddings)
        stock_out = stock.apply(params, rng, embeddings)
    except Exception as e:
        pytest.skip(f'stock init failed: {type(e).__name__}: {e}')

    cond_share.install()
    atom_cond_hoist.install()
    diffusion_hoist.install()
    hoisted = hk.transform(forward)
    try:
        hparams = hoisted.init(rng, embeddings)
        hout = hoisted.apply(hparams, rng, embeddings)
    finally:
        diffusion_hoist.uninstall()
        atom_cond_hoist.uninstall()
        cond_share.uninstall()

    def _leaves(tree):
        return jax.tree_util.tree_structure(tree)

    assert _leaves(params) == _leaves(hparams)
    stock_pos = np.asarray(stock_out['atom_positions'])
    hoist_pos = np.asarray(hout['atom_positions'])
    np.testing.assert_allclose(stock_pos, hoist_pos, rtol=1e-4, atol=1e-4)
    assert cond_share.report()['traced'] > 0
    assert diffusion_hoist.report()['traced'] > 0
    assert atom_cond_hoist.report()['precomputed'] > 0
