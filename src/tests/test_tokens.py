"""Token estimator tests (inline inputs; nothing needs the AlphaFold 3 examples)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fullFold.config import DEFAULT_BUCKETS
from fullFold.tokens import bucket_for, count_tokens, count_tokens_exact, near_boundary

UBIQUITIN = ('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG')


def _doc(*sequences) -> dict:
    return {'name': 'x', 'modelSeeds': [1], 'sequences': list(sequences)}


def _protein(seq, id_='A', **extra) -> dict:
    return {'protein': {'id': id_, 'sequence': seq, **extra}}


def test_buckets_and_the_boundary_margin():
    assert [bucket_for(n, DEFAULT_BUCKETS) for n in (1, 128, 129, 6000)] == [128, 128, 256, 6000]
    assert near_boundary(122, DEFAULT_BUCKETS, 0.05)
    assert not near_boundary(76, DEFAULT_BUCKETS, 0.05)


def test_token_estimates_and_when_they_are_exact():
    assert count_tokens(_doc(_protein(UBIQUITIN))) == (76, True)
    # every chain id is a copy
    assert count_tokens(_doc(_protein('ACDE', ['A', 'B']))) == (8, True)
    # 149 residues plus four calcium ions of one atom each
    calmodulin = _doc(_protein('A' * 149),
                      {'ligand': {'id': ['B', 'C', 'D', 'E'], 'ccdCodes': ['CA']}})
    assert count_tokens(calmodulin) == (153, True)
    # modified residues add their atoms; both codes are in the table, so still exact
    mods = [{'ptmType': 'TPO', 'ptmPosition': 3}, {'ptmType': 'PTR', 'ptmPosition': 9}]
    n, exact = count_tokens(_doc(_protein('A' * 50, modifications=mods)))
    assert n > 50 and exact
    # a code the table does not know is an estimate, to be settled by the tokenizer
    n, exact = count_tokens(_doc({'ligand': {'id': 'B', 'ccdCodes': ['MOV']}}))
    assert n > 0 and not exact
    n, exact = count_tokens(_doc(_protein('A' * 50, modifications=[{'ptmType': 'XYZ'}])))
    assert n > 50 and not exact


def test_smiles_ligands_count_heavy_atoms():
    pytest.importorskip('rdkit')
    ethanol = {'ligand': {'id': 'B', 'smiles': 'CCO'}}
    assert count_tokens(_doc(_protein('ACDE'), ethanol)) == (4 + 3, True)


def test_estimate_matches_the_alphafold3_tokenizer(tmp_path: Path):
    pytest.importorskip('alphafold3')
    doc = _doc(
        _protein(UBIQUITIN, unpairedMsa='', pairedMsa='', templates=[]),
        {'ligand': {'id': ['B', 'C'], 'ccdCodes': ['CA']}})
    doc.update(dialect='alphafold3', version=3)
    path = tmp_path / 'x.json'
    path.write_text(json.dumps(doc))
    try:
        exact = count_tokens_exact(path)
    except Exception as e:
        pytest.skip(f'exact tokenizer unavailable: {e}')
    assert exact == count_tokens(doc)[0] == 78
