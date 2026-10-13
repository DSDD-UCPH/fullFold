"""Template generator: schema preservation, SMILES-as-ligand, empty MSA."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fullFold.config import Config
from fullFold.templates import (
    add_entity, alloc_ids, cmd_template, entity_dict, generate, read_records, Record,
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


def test_the_kind_of_a_record_comes_from_the_file_not_from_the_string(tmp_path: Path):
    smi = tmp_path / 'x.smi'
    smi.write_text('CCCC butane\n')
    recs = read_records(smi)
    assert recs[0].kind == 'ligand_smiles'
    ligand = entity_dict(recs[0], 'B')['ligand']
    assert ligand['smiles'] == 'CCCC'
    assert 'sequence' not in ligand and 'unpairedMsa' not in ligand
    csv_ = tmp_path / 'x.csv'
    csv_.write_text('id,smiles\nl1,CCCC\n')
    assert entity_dict(read_records(csv_)[0], 'B')['ligand']['smiles'] == 'CCCC'
    # the same string in a FASTA is a protein, with empty MSAs so no data pipeline is needed
    fa = tmp_path / 'x.fasta'
    fa.write_text('>pep\nCCCC\n')
    recs = read_records(fa, 'protein')
    assert recs[0].kind == 'protein'
    protein = entity_dict(recs[0], 'B')['protein']
    assert protein['sequence'] == 'CCCC'
    assert (protein['unpairedMsa'], protein['pairedMsa'], protein['templates']) == ('', '', [])
    with pytest.raises(ValueError, match='ligands'):
        read_records(fa, 'ligand')  # a FASTA cannot hold one


def test_invalid_smiles_line_number(tmp_path: Path):
    pytest.importorskip('rdkit')
    rec = Record('ligand_smiles', 'bad', 'not_a_smiles_@@@', '', 1, 4)
    with pytest.raises(ValueError, match='line 4'):
        entity_dict(rec, 'B')


def test_adding_an_entity_leaves_the_template_alone():
    rec = Record('protein', 'binder', 'GGGG', 'a binder', 1, 1)
    out = add_entity(TEMPLATE, rec)
    assert out['modelSeeds'] == [7, 11]
    assert out['bondedAtomPairs'] == TEMPLATE['bondedAtomPairs']
    assert out['userCCD'] == TEMPLATE['userCCD']
    assert (out['dialect'], out['version']) == ('alphafold3', 3)
    orig = out['sequences'][0]['protein']
    assert orig['unpairedMsa'] is None and orig['id'] == 'A'
    added = out['sequences'][1]['protein']
    assert added['unpairedMsa'] == '' and added['pairedMsa'] == '' and added['id'] == 'B'
    # several copies take the next free chain ids
    dna = add_entity(TEMPLATE, Record('dna', 'd', 'AT', '', 2, 1))
    assert dna['sequences'][1]['dna']['id'] == ['B', 'C']
    assert alloc_ids({'A', 'B'}, 2) == ['C', 'D']
    with pytest.raises(ValueError, match='modelSeeds'):
        add_entity({'sequences': []}, Record('protein', 'x', 'AA', '', 1, 1))


def test_generate_one_per_record(tmp_path: Path):
    tmpl = {
        'name': 'receptor', 'modelSeeds': [7, 11], 'dialect': 'alphafold3', 'version': 3,
        'sequences': [{'protein': {'id': 'A', 'sequence': 'ACDE', 'unpairedMsa': '',
                                   'pairedMsa': '', 'templates': []}}],
    }
    recs = [Record('protein', 'p1', 'AA', '', 1, 1), Record('protein', 'p2', 'GG', '', 1, 2)]
    n, err = generate(tmpl, recs, tmp_path, 'receptor')
    assert n == 2 and not err
    files = sorted(tmp_path.glob('*.json'))
    assert len(files) == 2
    for f in files:
        doc = json.loads(f.read_text())
        assert doc['name'] == f.stem and doc['modelSeeds'] == [7, 11]
        try:
            from alphafold3.common import folding_input
            folding_input.Input.from_json(f.read_text())
        except ImportError:
            pass


def test_record_files_are_read_strictly(tmp_path: Path):
    def write(name, text):
        p = tmp_path / name
        p.write_text(text)
        return p

    # each FASTA record reports its own header line, and an empty file has no records
    fa = write('x.fa', '# note\n>first\nAAAA\nCC\n\n>second\nDDDD\n>third\nEEEE\n')
    recs = read_records(fa, 'protein')
    assert [(r.id_hint, r.line) for r in recs] == [('first', 2), ('second', 6), ('third', 8)]
    assert recs[0].payload == 'AAAACC'
    assert read_records(write('empty.fa', ''), 'protein') == []
    # a type that is not protein, rna or dna is an error, never silently a protein
    with pytest.raises(ValueError, match="unknown record type 'prtoein'"):
        read_records(fa, 'prtoein')
    with pytest.raises(ValueError, match=r"x\.csv:2: unknown record type 'peptide'"):
        read_records(write('x.csv', 'id,sequence,type\nx,AAAA,peptide\n'), 'protein')
    kinds = write('kinds.csv', 'id,sequence,type\na,AAAA,RNA\nb,CCCC,\nc,GGGG,dna\n')
    assert [r.kind for r in read_records(kinds, 'protein')] == ['rna', 'protein', 'dna']
    with pytest.raises(ValueError, match='both'):
        read_records(write('both.csv', 'sequence,smiles\nAAAA,CCCC\n'))
    # a count is a positive integer, and optional
    for count in ('0', '-2', 'two', '1.5'):
        bad = write('count.csv', f'id,sequence,count\na,AAAA,1\nb,CCCC,{count}\n')
        with pytest.raises(ValueError, match=r'count\.csv:3 count must be a positive integer'):
            read_records(bad, 'protein')
    assert read_records(write('nocount.csv', 'id,sequence\na,AAAA\n'), 'protein')[0].count == 1


def test_template_command_exits_cleanly_on_bad_input(tmp_path: Path):
    from fullFold.cli import main
    tpl = tmp_path / 'r.json'
    tpl.write_text(json.dumps({'name': 'r', 'modelSeeds': [1], 'sequences': []}))
    bad = tmp_path / 'x.csv'
    bad.write_text('id,sequence,count\na,AAAA,0\n')
    cfg = Config(template=tpl, records=bad, output_dir=tmp_path / 'out')
    with pytest.raises(SystemExit, match='count must be a positive integer'):
        cmd_template(cfg)
    with pytest.raises(SystemExit) as exc:  # --type is restricted by the parser
        main(['template', '--template', str(tpl), '--records', 'r.fa',
              '--output-dir', 'o', '--type', 'peptide'])
    assert exc.value.code == 2
