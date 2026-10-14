"""featfast: fingerprint table, and exactness of the two patches against stock AF3 v3.0.4 (skipped without AlphaFold 3)."""

from __future__ import annotations

import random

import pytest

from fullFold import featfast


def test_every_group_target_has_a_digest():
    for keys in featfast.GROUPS.values():
        for k in keys:
            rel, qual, digest = featfast.TARGETS[k]
            assert rel.endswith('.py') and qual and len(digest) == 64


def test_disabled_by_environment(monkeypatch):
    monkeypatch.setenv('FULLFOLD_FAST_FEAT', '0')
    assert featfast.install() == []



@pytest.fixture
def af3():
    pytest.importorskip('alphafold3.data.msa_features')
    from fullFold import featfast as ff
    ok, why = ff.check()['msa_extract']
    if not ok:
        pytest.skip(f'installed AlphaFold 3 is not v3.0.4 ({why})')
    return True


def test_msa_extract_matches_stock(af3):
    from alphafold3.constants import mmcif_names
    from alphafold3.data import msa_features as mf
    stock = mf.extract_msa_features
    maps = {mmcif_names.RNA_CHAIN: mf._RNA_TO_ID, mmcif_names.DNA_CHAIN: mf._DNA_TO_ID,
            mmcif_names.PROTEIN_CHAIN: mf._PROTEIN_TO_ID}
    rng = random.Random(0)
    for chain, alphabet in ((mmcif_names.PROTEIN_CHAIN, 'ACDEFGHIKLMNPQRSTVWYXBZUO-'),
                            (mmcif_names.RNA_CHAIN, 'ACGUN-'), (mmcif_names.DNA_CHAIN, 'ACGTN-')):
        for _ in range(40):
            n = rng.randint(1, 60)
            rows = []
            for _r in range(rng.randint(1, 30)):
                row = []
                for _c in range(n):
                    row.append(rng.choice(alphabet))
                    for _i in range(rng.choice((0, 0, 0, 1, 3))):
                        row.append(rng.choice(alphabet.lower().replace('-', 'a')))
                rows.append(''.join(row))
            want = stock(rows, chain)
            got = featfast.extract_msa_features_fast(stock, maps, rows, chain)
            assert (want[0] == got[0]).all() and (want[1] == got[1]).all()
            assert want[0].dtype == got[0].dtype and want[1].dtype == got[1].dtype


@pytest.mark.parametrize('rows', [['ACD', 'AC'], ['ACD', 'AC1D'], ['ACéD', 'ACDD'], []])
def test_msa_extract_errors_like_stock(af3, rows):
    from alphafold3.constants import mmcif_names
    from alphafold3.data import msa_features as mf
    stock = mf.extract_msa_features
    maps = {mmcif_names.PROTEIN_CHAIN: mf._PROTEIN_TO_ID}
    try:
        want = stock(rows, mmcif_names.PROTEIN_CHAIN)
    except Exception as e:
        with pytest.raises(type(e)):
            featfast.extract_msa_features_fast(stock, maps, rows, mmcif_names.PROTEIN_CHAIN)
        return
    got = featfast.extract_msa_features_fast(stock, maps, rows, mmcif_names.PROTEIN_CHAIN)
    assert (want[0] == got[0]).all()


_SCRIPT = r'''
import hashlib, json, sys
import numpy as np
from alphafold3.common import folding_input
from fullFold import featfast
from fullFold.runner import featurise_fold
if sys.argv[1] == "fast":
    assert set(featfast.install(conf_threads=3)) == {"msa_extract", "conformer_prefetch"}
fold = folding_input.Input.from_json(sys.argv[2])
h = hashlib.sha256()
for seed in (1, 2):
    batch = featurise_fold(fold, seed, [128])
    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x): h.update(k.encode()); walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif hasattr(x, "__dict__") and not hasattr(x, "shape"):
            walk(vars(x))
        else:
            a = np.asarray(x)
            if a.dtype != object: h.update(a.tobytes()); h.update(str(a.shape).encode())
    walk(batch)
print(h.hexdigest())
'''

_INPUT = {
    'name': 'tiny', 'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 2,
    'sequences': [
        {'protein': {'id': 'A', 'sequence': 'MKTAYIAKQRQISFVKSHFSRQ', 'unpairedMsa': '>q\nMKTAYIAKQRQISFVKSHFSRQ\n>h\nMKTAYIAKQRQ-SFVKSHFSRQ\n',
                     'pairedMsa': '', 'templates': []}},
        {'ligand': {'id': 'B', 'ccdCodes': ['ATP', 'MG']}},
        {'ligand': {'id': 'C', 'smiles': 'CC(=O)Oc1ccccc1C(=O)O'}},
    ],
}


def test_featurisation_is_bitwise_identical_with_and_without_the_patches(af3, tmp_path):
    import json
    import subprocess
    import sys
    pytest.importorskip('rdkit')
    ok, why = featfast.check()['conformer_prefetch']
    if not ok:
        pytest.skip(why)
    (tmp_path / 's.py').write_text(_SCRIPT)
    out = {}
    for mode in ('stock', 'fast'):
        r = subprocess.run([sys.executable, str(tmp_path / 's.py'), mode, json.dumps(_INPUT)], capture_output=True, text=True,
                           env={**__import__('os').environ, 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'})
        assert r.returncode == 0, r.stderr[-2000:]
        out[mode] = r.stdout.strip().splitlines()[-1]
    assert out['stock'] == out['fast'] and len(out['fast']) == 64
