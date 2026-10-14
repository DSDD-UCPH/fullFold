"""AlphaFold 3 v3.0.4 compatibility: write_output without ``keep_license``; hoist fingerprints on every Python version."""

from __future__ import annotations

import json

import pytest

from fullFold import worker


def test_write_output_drops_keep_license_when_af3_lacks_it(monkeypatch):
    pp = pytest.importorskip('alphafold3.model.post_processing')
    seen = {}

    def v304(inference_result, output_dir, terms_of_use=None, name=None, compress=False):
        seen.update(inference_result=inference_result, terms_of_use=terms_of_use, name=name, compress=compress)

    monkeypatch.setattr(pp, 'write_output', v304)
    worker.write_output(inference_result='r', output_dir='d', terms_of_use='t', name='n', compress=True, keep_license=True)
    assert seen == dict(inference_result='r', terms_of_use='t', name='n', compress=True)


def test_write_output_passes_keep_license_when_af3_has_it(monkeypatch):
    pp = pytest.importorskip('alphafold3.model.post_processing')
    seen = {}

    def colabfold(inference_result, output_dir, terms_of_use=None, name=None, compress=False, keep_license=False):
        seen['keep_license'] = keep_license

    monkeypatch.setattr(pp, 'write_output', colabfold)
    worker.write_output(inference_result='r', output_dir='d', keep_license=True)
    assert seen == {'keep_license': True}


def test_fingerprints_json_has_the_version_independent_digest_for_every_target():
    from fullFold.hoists import fingerprint as F
    known = json.loads(F.fingerprints_path().read_text())['hashes']
    assert set(known) == set(F.TARGETS)
    assert all(len(v) >= 2 for v in known.values())  # canonical + at least one ast.dump form


def test_hoists_match_the_installed_v304_source_on_this_python():
    from fullFold.hoists import fingerprint as F
    root = F.alphafold3_root()
    if root is None or F.colabfold_tree(root):
        pytest.skip('stock AlphaFold 3 v3.0.4 not installed')
    digests = F.current_digests(root)
    known = json.loads(F.fingerprints_path().read_text())
    if not all(F._match(k, digests[k], known) for k in F.TARGETS):
        pytest.skip('installed AlphaFold 3 is not the unmodified v3.0.4 source')
    st = F.evaluate_hoists(mode='default')
    assert {n for n in ('cond_share', 'atom_cond_hoist', 'diffusion_hoist') if st[n].install} == {
        'cond_share', 'atom_cond_hoist', 'diffusion_hoist'}
