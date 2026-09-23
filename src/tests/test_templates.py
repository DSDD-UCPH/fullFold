"""Template generator: schema preservation, SMILES-as-ligand, empty MSA."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fullFold.templates import (
    add_entity, alloc_ids, entity_dict, generate, read_records, Record,
)


TEMPLATE = {
    'name': 'receptor',
    'modelSeeds': [7, 11],
    'dialect': 'alphafold3',
    'version': 3,
    'bondedAtomPairs': [[['A', 1, 'CA'], ['A', 2, 'C']]],
    'userCCD': 'data_X\n',
    'sequences': [{
        'protein': {
            'id': 'A',
            'sequence': 'ACDE',
            'unpairedMsa': None,
            'pairedMsa': None,
        }
    }],
}


def test_smiles_from_smi_is_ligand(tmp_path: Path):
    p = tmp_path / 'x.smi'
    p.write_text('CCCC butane\n')
    recs = read_records(p)
    assert recs[0].kind == 'ligand_smiles'
    body = entity_dict(recs[0], 'B')
    assert 'ligand' in body
    assert body['ligand']['smiles'] == 'CCCC'
    assert 'sequence' not in body['ligand']
    assert 'unpairedMsa' not in body['ligand']


def test_smiles_from_csv(tmp_path: Path):
    p = tmp_path / 'x.csv'
    p.write_text('id,smiles\nl1,CCCC\n')
    recs = read_records(p)
    assert recs[0].kind == 'ligand_smiles'
    assert entity_dict(recs[0], 'B')['ligand']['smiles'] == 'CCCC'


def test_fasta_cccc_is_protein(tmp_path: Path):
    p = tmp_path / 'x.fasta'
    p.write_text('>pep\nCCCC\n')
    recs = read_records(p, 'protein')
    assert recs[0].kind == 'protein'
    body = entity_dict(recs[0], 'B')
    assert body['protein']['sequence'] == 'CCCC'
    assert body['protein']['unpairedMsa'] == ''
    assert body['protein']['pairedMsa'] == ''
    assert body['protein']['templates'] == []


def test_fasta_type_ligand_refused(tmp_path: Path):
    p = tmp_path / 'x.fa'
    p.write_text('>x\nCCCC\n')
    with pytest.raises(ValueError, match='ligands'):
        read_records(p, 'ligand')


def test_invalid_smiles_line_number(tmp_path: Path):
    pytest.importorskip('rdkit')
    rec = Record('ligand_smiles', 'bad', 'not_a_smiles_@@@', '', 1, 4)
    with pytest.raises(ValueError, match='line 4'):
        entity_dict(rec, 'B')


def test_schema_preservation_and_seeds():
    rec = Record('protein', 'binder', 'GGGG', 'a binder', 1, 1)
    out = add_entity(TEMPLATE, rec)
    assert out['modelSeeds'] == [7, 11]
    assert out['bondedAtomPairs'] == TEMPLATE['bondedAtomPairs']
    assert out['userCCD'] == TEMPLATE['userCCD']
    assert out['dialect'] == 'alphafold3'
    assert out['version'] == 3
    orig = out['sequences'][0]['protein']
    assert orig['unpairedMsa'] is None
    assert orig['id'] == 'A'
    added = out['sequences'][1]['protein']
    assert added['unpairedMsa'] == '' and added['pairedMsa'] == ''
    assert added['id'] == 'B'


def test_id_collision_and_count():
    rec = Record('dna', 'd', 'AT', '', 2, 1)
    out = add_entity(TEMPLATE, rec)
    assert out['sequences'][1]['dna']['id'] == ['B', 'C']


def test_missing_seeds_errors():
    rec = Record('protein', 'x', 'AA', '', 1, 1)
    with pytest.raises(ValueError, match='modelSeeds'):
        add_entity({'sequences': []}, rec)


def test_generate_one_per_record(tmp_path: Path):
    tmpl = {
        'name': 'receptor', 'modelSeeds': [7, 11],
        'dialect': 'alphafold3', 'version': 3,
        'sequences': [{'protein': {'id': 'A', 'sequence': 'ACDE',
                                   'unpairedMsa': '', 'pairedMsa': '',
                                   'templates': []}}],
    }
    recs = [
        Record('protein', 'p1', 'AA', '', 1, 1),
        Record('protein', 'p2', 'GG', '', 1, 2),
    ]
    n, err = generate(tmpl, recs, tmp_path, 'receptor')
    assert n == 2 and not err
    files = sorted(tmp_path.glob('*.json'))
    assert len(files) == 2
    for f in files:
        doc = json.loads(f.read_text())
        assert doc['name'] == f.stem
        assert doc['modelSeeds'] == [7, 11]
        try:
            from alphafold3.common import folding_input
            folding_input.Input.from_json(f.read_text())
        except ImportError:
            pass


def test_csv_both_sequence_and_smiles_errors(tmp_path: Path):
    p = tmp_path / 'x.csv'
    p.write_text('sequence,smiles\nAAAA,CCCC\n')
    with pytest.raises(ValueError, match='both'):
        read_records(p)
