"""One-JSON-per-record generator. SMILES are always ligands, never sequences."""

from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from fullFold.config import Config, sanitise, write_atomic

_FASTA = {'.fa', '.fasta', '.faa', '.fna', '.fas'}
_SMI = {'.smi', '.smiles'}
_CSV = {'.csv', '.tsv'}


@dataclass(frozen=True)
class Record:
    kind: str
    id_hint: str
    payload: str
    description: str
    count: int
    line: int


def int_id_to_str_id(num: int) -> str:
    if num <= 0:
        raise ValueError(f'Only positive integers allowed, got {num}')
    num = num - 1
    out = []
    while num >= 0:
        out.append(chr(num % 26 + ord('A')))
        num = num // 26 - 1
    return ''.join(out)


def _used_ids(template: dict) -> set[str]:
    ids: set[str] = set()
    for seq in template.get('sequences') or []:
        if not isinstance(seq, dict) or not seq:
            continue
        body = next(iter(seq.values()))
        if not isinstance(body, dict):
            continue
        i = body.get('id')
        if isinstance(i, list):
            ids.update(str(x) for x in i)
        elif i:
            ids.add(str(i))
    return ids


def alloc_ids(used: set[str], n: int) -> list[str]:
    out, k = [], 1
    while len(out) < n:
        s = int_id_to_str_id(k)
        if s not in used:
            out.append(s)
        k += 1
    return out


def _flush_fasta(recs, header, chunks, kind, i):
    if header is None:
        return
    recs.append(Record(kind, sanitise(header) or f'seq{len(recs)+1}',
                       ''.join(chunks), header, 1, i))


def _read_fasta(path: Path, kind: str) -> list[Record]:
    if kind in ('ligand', 'ligand_smiles', 'ligand_ccd'):
        raise ValueError(
            f'FASTA cannot be used for ligands; use a .smi file or a CSV smiles column ({path})'
        )
    kind = kind if kind in ('protein', 'rna', 'dna') else 'protein'
    recs, header, chunks = [], None, []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith('>'):
            _flush_fasta(recs, header, chunks, kind, i)
            header, chunks = line[1:].strip(), []
        else:
            chunks.append(line.strip())
    _flush_fasta(recs, header, chunks, kind, i if recs or header else 1)
    return recs


def _read_smi(path: Path) -> list[Record]:
    recs = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        smiles, name = parts[0], parts[1] if len(parts) > 1 else f'lig{i}'
        recs.append(Record('ligand_smiles', sanitise(name) or f'lig{i}',
                           smiles, name, 1, i))
    return recs


def _read_csv(path: Path, default_kind: str) -> list[Record]:
    delim = '\t' if path.suffix.lower() == '.tsv' else ','
    recs = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f'{path} has no header')
        fields = {n.lower(): n for n in reader.fieldnames}
        for i, row in enumerate(reader, 2):
            smiles = (row.get(fields.get('smiles', ''), '') or '').strip()
            seq = (row.get(fields.get('sequence', ''), '') or '').strip()
            ccd = (row.get(fields.get('ccd_codes', fields.get('ccdcodes', '')), '') or '').strip()
            if smiles and seq:
                raise ValueError(f'{path}:{i} has both sequence and smiles')
            if smiles:
                kind, payload = 'ligand_smiles', smiles
            elif ccd:
                kind, payload = 'ligand_ccd', ccd
            elif seq:
                t = (row.get(fields.get('type', ''), '') or default_kind).lower()
                if t in ('ligand', 'ligand_smiles'):
                    raise ValueError(f'{path}:{i} type=ligand requires a smiles column')
                kind, payload = (t if t in ('protein', 'rna', 'dna') else 'protein'), seq
            else:
                raise ValueError(f'{path}:{i} has neither sequence, smiles, nor ccd_codes')
            hid = (row.get(fields.get('id', ''), '') or '').strip() or f'rec{i}'
            desc = (row.get(fields.get('description', ''), '') or '').strip()
            count = int(row.get(fields.get('count', ''), '1') or 1)
            recs.append(Record(kind, sanitise(hid) or f'rec{i}', payload, desc, count, i))
    return recs


def read_records(path: Path, default_kind: str = 'protein') -> list[Record]:
    suf = path.suffix.lower()
    if suf in _FASTA:
        return _read_fasta(path, default_kind)
    if suf in _SMI:
        return _read_smi(path)
    if suf in _CSV:
        return _read_csv(path, default_kind)
    raise ValueError(f'unsupported records file: {path}')


def _validate_smiles(smiles: str, line: int) -> None:
    try:
        from rdkit import Chem
    except ImportError:
        return
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f'invalid SMILES on line {line}: {smiles!r}')


def entity_dict(rec: Record, chain_id) -> dict:
    if rec.kind == 'ligand_smiles':
        _validate_smiles(rec.payload, rec.line)
        body, key = {'id': chain_id, 'smiles': rec.payload}, 'ligand'
    elif rec.kind == 'ligand_ccd':
        codes = [c.strip() for c in re.split(r'[,;\s]+', rec.payload) if c.strip()]
        body, key = {'id': chain_id, 'ccdCodes': codes}, 'ligand'
    elif rec.kind == 'protein':
        body, key = ({
            'id': chain_id, 'sequence': rec.payload,
            'unpairedMsa': '', 'pairedMsa': '', 'templates': [],
        }, 'protein')
    elif rec.kind == 'rna':
        body, key = {'id': chain_id, 'sequence': rec.payload, 'unpairedMsa': ''}, 'rna'
    elif rec.kind == 'dna':
        body, key = {'id': chain_id, 'sequence': rec.payload}, 'dna'
    else:
        raise ValueError(f'unknown kind {rec.kind}')
    if rec.description:
        body['description'] = rec.description
    return {key: body}


def add_entity(template: dict, rec: Record) -> dict:
    if 'modelSeeds' not in template:
        raise ValueError('template is missing modelSeeds')
    out = deepcopy(template)
    used = _used_ids(out)
    ids = alloc_ids(used, rec.count)
    chain_id = ids if rec.count > 1 else ids[0]
    out.setdefault('sequences', []).append(entity_dict(rec, chain_id))
    return out


def _unique_name(stem: str, rec_id: str, used: set[str]) -> str:
    base = sanitise(f'{stem}__{rec_id}') or f'{stem}__rec'
    name, n = base, 2
    while name in used:
        name = f'{base}_{n}'
        n += 1
    used.add(name)
    return name


def generate(template: dict, records: list[Record], out_dir: Path, stem: str) -> tuple[int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    ok, errors = 0, []
    try:
        from alphafold3.common import folding_input
    except ImportError:
        folding_input = None
    for rec in records:
        try:
            doc = add_entity(template, rec)
            name = _unique_name(stem, rec.id_hint, used)
            doc['name'] = name
            text = json.dumps(doc, indent=2) + '\n'
            if folding_input is not None:
                folding_input.Input.from_json(text)
            write_atomic(out_dir / f'{name}.json', text)
            ok += 1
        except Exception as e:
            errors.append(f'line {rec.line} {rec.id_hint}: {e}')
    return ok, errors


def cmd_template(cfg: Config) -> int:
    if cfg.template is None or cfg.records is None:
        raise SystemExit('--template and --records are required')
    template = json.loads(cfg.template.read_text())
    if not isinstance(template, dict):
        raise SystemExit('template must be an AlphaFold 3 JSON object')
    recs = read_records(cfg.records, cfg.record_type)
    ok, errors = generate(template, recs, cfg.output_dir, cfg.template.stem)
    print(f'wrote {ok} JSON files to {cfg.output_dir}')
    for e in errors:
        print(f'  skip: {e}')
    return 1 if errors and not ok else 0
