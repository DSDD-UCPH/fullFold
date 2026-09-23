"""Token estimator tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fullFold.config import DEFAULT_BUCKETS
from fullFold.tokens import (
    bucket_for, count_tokens, count_tokens_exact, near_boundary,
)

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def _example_json(name: str) -> dict:
    path = EXAMPLES / name
    if not path.is_file():
        pytest.skip(f'{name} is not in examples/ (AlphaFold 3 repo examples)')
    return json.loads(path.read_text())


def test_bucket_for():
    assert bucket_for(1, DEFAULT_BUCKETS) == 128
    assert bucket_for(128, DEFAULT_BUCKETS) == 128
    assert bucket_for(129, DEFAULT_BUCKETS) == 256
    assert bucket_for(6000, DEFAULT_BUCKETS) == 6000


def test_ubiquitin_is_exact_76():
    raw = _example_json('ubiquitin_monomer.json')
    n, exact = count_tokens(raw)
    assert n == 76
    assert exact


def test_calmodulin_ions():
    raw = _example_json('calmodulin_4calcium.json')
    n, exact = count_tokens(raw)
    # 149 aa + 4 calcium ions (1 atom each)
    assert n == 149 + 4
    assert exact


def test_smiles_biotin_heavy_atoms():
    pytest.importorskip('rdkit')
    raw = _example_json('streptavidin_biotin_smiles.json')
    n, exact = count_tokens(raw)
    prot = len(raw['sequences'][0]['protein']['sequence'])
    assert exact
    assert n == prot + 16  # biotin C10H16N2O3S -> 16 heavy atoms


def test_unknown_ccd_is_inexact():
    raw = {
        'sequences': [{'ligand': {'id': 'B', 'ccdCodes': ['MOV']}}],
        'name': 'x', 'modelSeeds': [1],
    }
    n, exact = count_tokens(raw)
    assert not exact
    assert n > 0


def test_modifications_inexact_without_table_hit():
    raw = _example_json('erk2_phosphorylated.json')
    n, exact = count_tokens(raw)
    seq = len(raw['sequences'][0]['protein']['sequence'])
    assert n > seq  # TPO + PTR extra atoms
    assert exact  # both codes are in CCD_ATOM_COUNTS


def test_near_boundary():
    assert near_boundary(122, DEFAULT_BUCKETS, 0.05)
    assert not near_boundary(76, DEFAULT_BUCKETS, 0.05)


def test_chain_id_list_multiplies():
    raw = {
        'sequences': [{
            'protein': {'id': ['A', 'B'], 'sequence': 'ACDE'},
        }],
    }
    n, exact = count_tokens(raw)
    assert n == 8
    assert exact


@pytest.mark.parametrize('name', sorted(p.name for p in EXAMPLES.glob('*.json')))
def test_fast_matches_sequence_length_for_polymers(name):
    raw = json.loads((EXAMPLES / name).read_text())
    n, exact = count_tokens(raw)
    assert n > 0
    if not exact:
        pytest.skip('inexact (ligand CCD not in table)')


def test_exact_optional():
    path = EXAMPLES / 'ubiquitin_monomer.json'
    if not path.is_file():
        pytest.skip('ubiquitin_monomer.json is not in examples/')
    try:
        n = count_tokens_exact(path)
    except Exception as e:
        pytest.skip(f'exact tokenizer unavailable: {e}')
    fast, _ = count_tokens(json.loads(path.read_text()))
    assert n == fast == 76
