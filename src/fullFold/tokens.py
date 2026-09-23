"""Native-compatible token counting for AlphaFold 3 JSON inputs."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

# Standard polymer CCD codes: 1 token each (MSE is treated as standard MET).
_STANDARD = frozenset({
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'UNK', 'MSE',
    'A', 'G', 'C', 'U', 'DA', 'DG', 'DC', 'DT', 'N', 'DN',
})

# Heavy-atom counts for common CCD ligands / PTMs. Missing codes -> inexact.
CCD_ATOM_COUNTS = {
    'CA': 1, 'MG': 1, 'ZN': 1, 'NA': 1, 'CL': 1, 'K': 1, 'MN': 1, 'FE': 1,
    'CU': 1, 'NI': 1, 'CO': 1, 'BR': 1, 'F': 1, 'I': 1,
    'TPO': 14, 'PTR': 16, 'SEP': 10, 'HYP': 8, 'MLY': 10, 'CSO': 7,
    'NAG': 14, 'BMA': 11, 'MAN': 11, 'FUC': 10, 'GAL': 11, 'SIA': 20,
    'BTN': 16, 'HEM': 43, 'NAD': 44, 'ATP': 31, 'GTP': 32, 'FAD': 53,
}
_DEFAULT_CCD_ATOMS = 20


def bucket_for(n: int, ladder: tuple[int, ...] | list[int]) -> int:
    if n < 0:
        raise ValueError(f'token count must be >= 0, got {n}')
    i = bisect.bisect_left(list(ladder), n)
    return n if i == len(ladder) else ladder[i]


def _n_ids(id_field: Any) -> int:
    if isinstance(id_field, list):
        return max(len(id_field), 1)
    return 1


def _smiles_heavy(smiles: str) -> int | None:
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _mod_extra(mods: list, type_key: str) -> tuple[int, bool]:
    extra, exact = 0, True
    for m in mods or []:
        code = m.get(type_key, '')
        if not code or code in _STANDARD:
            continue
        if code in CCD_ATOM_COUNTS:
            extra += max(CCD_ATOM_COUNTS[code] - 1, 0)
        else:
            extra += _DEFAULT_CCD_ATOMS - 1
            exact = False
    return extra, exact


def count_tokens(raw_json: dict) -> tuple[int, bool]:
    """Fast estimate. Returns (n, exact). No AlphaFold 3 import."""
    total, exact = 0, True
    for item in raw_json.get('sequences') or []:
        if not isinstance(item, dict) or len(item) != 1:
            exact = False
            continue
        kind, body = next(iter(item.items()))
        if not isinstance(body, dict):
            exact = False
            continue
        copies = _n_ids(body.get('id'))
        if kind in ('protein', 'rna', 'dna'):
            seq = body.get('sequence') or ''
            n = len(seq)
            tkey = 'ptmType' if kind == 'protein' else 'modificationType'
            extra, ok = _mod_extra(body.get('modifications') or [], tkey)
            n += extra
            exact = exact and ok
            total += n * copies
        elif kind == 'ligand':
            if body.get('smiles'):
                h = _smiles_heavy(body['smiles'])
                if h is None:
                    exact = False
                    h = _DEFAULT_CCD_ATOMS
                total += h * copies
            elif body.get('ccdCodes'):
                for code in body['ccdCodes']:
                    if code in CCD_ATOM_COUNTS:
                        total += CCD_ATOM_COUNTS[code] * copies
                    else:
                        total += _DEFAULT_CCD_ATOMS * copies
                        exact = False
            else:
                exact = False
        else:
            exact = False
    return total, exact


def count_tokens_exact(path: Path) -> int:
    """Exact count via AF3 tokenizer. Lazy-imports JAX-using modules."""
    from alphafold3.common import folding_input
    from alphafold3.constants import chemical_components
    from alphafold3.model import features
    from alphafold3.model.pipeline import structure_cleaning
    from alphafold3.model.pipeline.inter_chain_bonds import (
        get_polymer_ligand_and_ligand_ligand_bonds,
    )

    text = Path(path).read_text()
    fold = folding_input.Input.from_json(text, path)
    ccd = chemical_components.Ccd(user_ccd=fold.user_ccd)
    struct = fold.to_structure(ccd=ccd)
    cleaned = structure_cleaning.clean_structure(
        struct, ccd=ccd,
        drop_non_standard_atoms=True, drop_missing_sequence=True,
        filter_waters=True, filter_hydrogens=True,
        filter_leaving_atoms=True, only_glycan_ligands_for_leaving_atoms=True,
        covalent_bonds_only=True, remove_polymer_polymer_bonds=True,
        remove_bad_bonds=True, fix_standalone_glycans=False,
    )
    polymer_ligand_bonds, ligand_ligand_bonds = (
        get_polymer_ligand_and_ligand_ligand_bonds(
            cleaned, only_glycan_ligands=False, allow_multiple_bonds_per_atom=True,
        )
    )
    if ligand_ligand_bonds is not None and not getattr(
            ligand_ligand_bonds, 'atom_name', None).size:
        ligand_ligand_bonds = None
    if polymer_ligand_bonds is not None and not getattr(
            polymer_ligand_bonds, 'atom_name', None).size:
        polymer_ligand_bonds = None
    _, layout = structure_cleaning.create_empty_output_struc_and_layout(
        struc=cleaned, ccd=ccd,
        polymer_ligand_bonds=polymer_ligand_bonds,
        ligand_ligand_bonds=ligand_ligand_bonds,
        drop_ligand_leaving_atoms=True, fix_standalone_glycans=False,
    )
    all_tokens, _, _ = features.tokenizer(
        layout, ccd=ccd, max_atoms_per_token=24,
        flatten_non_standard_residues=True, logging_name='',
    )
    return len(all_tokens.atom_name)


def near_boundary(n: int, ladder: tuple[int, ...], margin: float) -> bool:
    b = bucket_for(n, ladder)
    return b > n and (b - n) <= max(int(b * margin), 1)
