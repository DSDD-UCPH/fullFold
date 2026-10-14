"""Exact featurisation speed-ups for AlphaFold 3 v3.0.4 (CPU side, bitwise identical output).

AF3 v3.0.4 spends 5-6 s of CPU per seed featurising a ~450 token complex, which is longer than the GPU needs for the
same seed on a modern card. Two stock hot spots are replaced here, at runtime, without touching the installed files:

``msa_extract``  ``alphafold3.data.msa_features.extract_msa_features`` converts an MSA to int arrays with a pure-Python loop
                 over every character of every row (~24 M dict look-ups, three calls per seed, the result does not depend on
                 the seed). The replacement does the same with one byte look-up table per row. Output is integer arrays,
                 so equality with the stock function is exact; anything the fast path is not certain about (non-ASCII, a
                 character that is neither mapped nor lower case, rows of different widths) is delegated to the stock function,
                 which then raises its own error.
``conformer_prefetch``  ``RefStructure.compute_features`` draws, per residue in layout order, a conformer seed from the job's
                 RandomState, embeds an RDKit ETKDGv3 conformer with it and then draws the random-augmentation numbers.
                 Everything but the embedding is a cheap function of the RandomState stream, so the seeds are replayed on a
                 *copy* of the state, all embeddings run on threads (RDKit releases the GIL) and
                 ``rdkit_utils.get_random_conformer`` is served from that table by key (res_name, seed, max_iterations).
                 A wrong replay or an unforeseen residue only misses the table and embeds as stock does.

These are the same two ideas as the vectorised MSA extraction and threaded conformer pre-embedding of the ColabFold
AlphaFold 3 fork (>= 3.1.x), made available for stock v3.0.4. Each patch is applied only if the installed source of every
function it relies on has the fingerprint of v3.0.4 (see ``TARGETS``); on any other source (including the ColabFold fork, which
has them natively) it is skipped and a notice is printed. ``--no-fast-featurisation`` (or ``FULLFOLD_FAST_FEAT=0``) disables it.
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
import os

# canonical (Python-version independent) digests of the AF3 v3.0.4 functions the patches depend on
TARGETS = {
    'msa_features.extract_msa_features': (
        'data/msa_features.py', 'extract_msa_features',
        'bb44f443590618f3d84c1a58389368eebac52f9cf4b9aad3417a1f9da4162370'),
    'features.RefStructure.compute_features': (
        'model/features.py', 'RefStructure.compute_features',
        'eca7460ed32094b62b9d4f5bceb699f6714c8090c502a819bbd7867352e775b3'),
    'features.get_reference': (
        'model/features.py', 'get_reference',
        '7d7541b4465e9d7749752df20ae9ccb60825fc6c5b8ab750496f2a857608ce1e'),
    'features.random_augmentation': (
        'model/features.py', 'random_augmentation',
        '78a456af3a91a402c97a7ca8fb6a0e0ad2118ed5dcf696eca94347940b3ca584'),
    'rdkit_utils.get_random_conformer': (
        'data/tools/rdkit_utils.py', 'get_random_conformer',
        '849eb61896b71d64d25f0d739654bbdd19f98166d8792b8297a4e3a5fe6ba9e1'),
    'rdkit_utils.mol_from_ccd_cif': (
        'data/tools/rdkit_utils.py', 'mol_from_ccd_cif',
        '19b4aab4a735f8e6aebde10c2081eec1b8c790119f9618173112b1e3b56e620f'),
    'rdkit_utils.assign_atom_names_from_graph': (
        'data/tools/rdkit_utils.py', 'assign_atom_names_from_graph',
        '14521a315ea9d38a6a62f35da7dc6f718965bbb5858c47a19b64fb7edc3231b4'),
}

GROUPS = {
    'msa_extract': ('msa_features.extract_msa_features',),
    'conformer_prefetch': (
        'features.RefStructure.compute_features', 'features.get_reference', 'features.random_augmentation',
        'rdkit_utils.get_random_conformer', 'rdkit_utils.mol_from_ccd_cif', 'rdkit_utils.assign_atom_names_from_graph'),
}

_STATE = {'msa_extract': False, 'conformer_prefetch': False, 'conf_threads': 0}
_LUTS = {}
_TABLE = {}


def enabled(environ=os.environ) -> bool:
    return environ.get('FULLFOLD_FAST_FEAT', '1').strip().lower() not in ('0', 'false', 'no', 'off')


def check() -> dict[str, tuple[bool, str]]:
    """Per patch: (source matches v3.0.4, reason). Reads the files only, imports neither AlphaFold nor JAX."""
    from fullFold.hoists import fingerprint as F
    root = F.alphafold3_root()
    out: dict[str, tuple[bool, str]] = {}
    trees: dict[str, ast.AST | None] = {}
    for group, keys in GROUPS.items():
        if root is None:
            out[group] = (False, 'alphafold3 not found')
            continue
        bad = []
        for key in keys:
            rel, qual, want = TARGETS[key]
            if rel not in trees:
                try:
                    trees[rel] = ast.parse((root / rel).read_text())
                except (OSError, SyntaxError):
                    trees[rel] = None
            node = F._find(trees[rel], qual) if trees[rel] is not None else None
            if node is None:
                bad.append(f'{key} missing')
            elif F.digests_for(node)[0] != want:
                bad.append(f'differs in {key}')
        out[group] = (not bad, '; '.join(bad))
    return out


# ---- msa_extract -------------------------------------------------------------------------------------------------------

def _lut(char_map):
    import numpy as np
    key = id(char_map)
    hit = _LUTS.get(key)
    if hit is None:
        lut = np.full(256, -1, dtype=np.int32)
        for ch, i in char_map.items():
            lut[ord(ch)] = i
        lower = np.zeros(256, dtype=bool)
        lower[ord('a'):ord('z') + 1] = True
        hit = _LUTS[key] = (lut, lower)
    return hit


def extract_msa_features_fast(stock, maps, msa_sequences, chain_poly_type):
    import numpy as np
    char_map = maps.get(chain_poly_type)
    if char_map is None or not msa_sequences:
        return stock(msa_sequences, chain_poly_type)
    lut, lower = _lut(char_map)
    try:
        first = np.frombuffer(msa_sequences[0].encode('ascii'), dtype=np.uint8)
    except (UnicodeEncodeError, AttributeError):
        return stock(msa_sequences, chain_poly_type)
    num_cols = int((lut[first] >= 0).sum())
    num_rows = len(msa_sequences)
    msa_arr = np.zeros((num_rows, num_cols), dtype=np.int32)
    del_arr = np.zeros((num_rows, num_cols), dtype=np.int32)
    for r, seq in enumerate(msa_sequences):
        try:
            b = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        except (UnicodeEncodeError, AttributeError):
            return stock(msa_sequences, chain_poly_type)
        ids = lut[b]
        is_up = ids >= 0
        idx = np.flatnonzero(is_up)
        if idx.size != num_cols or not bool(np.all(is_up | lower[b])):
            return stock(msa_sequences, chain_poly_type)  # stock raises the exact error
        msa_arr[r] = ids[idx]
        if num_cols:
            del_arr[r, 0] = idx[0]
            del_arr[r, 1:] = idx[1:] - idx[:-1] - 1
    return msa_arr, del_arr


def _install_msa_extract() -> None:
    from alphafold3.constants import mmcif_names
    from alphafold3.data import msa_features as mf
    stock = mf.extract_msa_features
    maps = {mmcif_names.RNA_CHAIN: mf._RNA_TO_ID, mmcif_names.DNA_CHAIN: mf._DNA_TO_ID,
            mmcif_names.PROTEIN_CHAIN: mf._PROTEIN_TO_ID}

    @functools.wraps(stock)
    def extract_msa_features(msa_sequences, chain_poly_type):
        return extract_msa_features_fast(stock, maps, msa_sequences, chain_poly_type)

    extract_msa_features.__wrapped__ = stock
    mf.extract_msa_features = extract_msa_features


# ---- conformer_prefetch ------------------------------------------------------------------------------------------------

def _mol_for(res_name, chemical_components_data, ccd, rdkit_utils, Chem):
    """The molecule stock get_reference would embed (None when stock would not draw a seed)."""
    ccd_cif = ccd.get(res_name)
    if ccd_cif:
        try:
            return rdkit_utils.mol_from_ccd_cif(ccd_cif, remove_hydrogens=False)
        except rdkit_utils.MolFromMmcifError:
            return None
    cc = chemical_components_data
    if not (cc.chem_comp and res_name in cc.chem_comp and cc.chem_comp[res_name].pdbx_smiles):
        raise ValueError('no smiles')
    mol = Chem.MolFromSmiles(cc.chem_comp[res_name].pdbx_smiles)
    if mol is None:
        raise ValueError('bad smiles')
    mol = Chem.AddHs(mol)
    return rdkit_utils.assign_atom_names_from_graph(mol)


def _prefetch(layout, ccd, chemical_components_data, random_state, max_iterations, rdkit_utils, Chem, rd_all_chem):
    import numpy as np
    state = copy.deepcopy(random_state)
    mols, jobs, seen = {}, [], set()
    try:
        for idx in np.ndindex(layout.shape):
            if not layout.atom_name[idx]:
                continue
            key = (layout.chain_id[idx], layout.res_id[idx])
            if key in seen:
                continue
            seen.add(key)
            res_name = layout.res_name[idx]
            if res_name not in mols:
                mols[res_name] = _mol_for(res_name, chemical_components_data, ccd, rdkit_utils, Chem)
            if mols[res_name] is not None:
                jobs.append((res_name, int(state.randint(1, 1 << 31))))
            state.normal(size=(2, 3))
            state.normal(size=(3,))
    except ValueError:
        pass  # stock raises the same error in its own loop; the seeds drawn so far still hold
    if len(jobs) < 2:
        return

    def embed(job):
        res_name, seed = job
        params = rd_all_chem.ETKDGv3()
        params.randomSeed = seed
        if max_iterations is not None:
            params.maxIterations = max_iterations
        mol_copy = Chem.Mol(mols[res_name])
        try:
            return (res_name, seed, max_iterations), mol_copy.GetConformer(rd_all_chem.EmbedMolecule(mol_copy, params))
        except ValueError:
            return (res_name, seed, max_iterations), None

    from concurrent.futures import ThreadPoolExecutor
    n = int(_STATE['conf_threads']) or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max(1, n)) as pool:
        _TABLE.update(pool.map(embed, jobs))


def _install_conformer_prefetch() -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem as rd_all_chem
    from alphafold3.data.tools import rdkit_utils
    from alphafold3.model import features

    stock_conf = rdkit_utils.get_random_conformer
    stock_cf = features.RefStructure.__dict__['compute_features']
    sig = inspect.signature(stock_cf.__func__)

    @functools.wraps(stock_conf)
    def get_random_conformer(mol, random_seed, max_iterations, logging_name):
        key = (logging_name, random_seed, max_iterations)
        if key in _TABLE:
            return _TABLE[key]  # a failed embedding is stored as None: stock would only log a warning
        return stock_conf(mol=mol, random_seed=random_seed, max_iterations=max_iterations, logging_name=logging_name)

    def compute_features(cls, *args, **kwargs):
        try:
            a = sig.bind(cls, *args, **kwargs).arguments
            _prefetch(a['all_token_atoms_layout'], a['ccd'], a['chemical_components_data'], a['random_state'],
                      a['conformer_max_iterations'], rdkit_utils, Chem, rd_all_chem)
        except Exception:
            _TABLE.clear()  # any problem: stock path only
        try:
            return stock_cf.__func__(cls, *args, **kwargs)
        finally:
            _TABLE.clear()

    rdkit_utils.get_random_conformer = get_random_conformer
    features.RefStructure.compute_features = classmethod(compute_features)


def install(*, conf_threads: int = 0, quiet: bool = False) -> list[str]:
    """Install the patches whose AF3 source fingerprints match. Idempotent. Returns the names installed (now or before)."""
    if not enabled():
        return []
    if conf_threads:
        _STATE['conf_threads'] = conf_threads
    verdict = check()
    done = []
    for name, installer in (('msa_extract', _install_msa_extract), ('conformer_prefetch', _install_conformer_prefetch)):
        if _STATE[name]:
            done.append(name)
            continue
        ok, why = verdict[name]
        if not ok:
            if not quiet:
                print(f'fullFold: fast featurisation {name} skipped ({why})', flush=True)
            continue
        installer()
        _STATE[name] = True
        done.append(name)
    return done
