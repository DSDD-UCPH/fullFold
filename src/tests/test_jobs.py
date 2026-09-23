"""Jobs scan + atomic claim markers."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from fullFold.config import Config, write_atomic
from fullFold.jobs import Job, claim, end_claim, lock_reclaimable, scan, try_begin_claim


def _job_json(tmp: Path, name: str, seq: str, seeds=(1,)) -> Path:
    p = tmp / f'{name}.json'
    p.write_text(json.dumps({
        'name': name,
        'modelSeeds': list(seeds),
        'sequences': [{'protein': {'id': 'A', 'sequence': seq}}],
        'dialect': 'alphafold3',
        'version': 4,
    }))
    return p


def test_scan_sorted_and_skips_unparseable(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    inp.mkdir(); out.mkdir()
    _job_json(inp, 'b', 'ACDE')
    _job_json(inp, 'a', 'ACDEFGH')
    (inp / 'bad.json').write_text('{not json')
    (inp / 'z.json').write_text('[]')
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, rejected = scan(cfg)
    assert [j.name for j in jobs] == ['b', 'a']  # 4 tokens then 7, both bucket 128
    assert len(rejected) == 2


def test_scan_skip_done(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    inp.mkdir(); out.mkdir()
    p = _job_json(inp, 'keep', 'AAAA')
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, _ = scan(cfg)
    job = jobs[0]
    d = job.state_dir(cfg)
    d.mkdir(parents=True)
    write_atomic(d / 'done.json', json.dumps({'sha256': job.sha256}))
    jobs2, _ = scan(cfg)
    assert jobs2 == []
    # changed input -> rerun
    p.write_text(p.read_text().replace('AAAA', 'AAAC'))
    jobs3, _ = scan(cfg)
    assert len(jobs3) == 1


def test_claim_exclusive(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    inp.mkdir(); out.mkdir()
    _job_json(inp, 'j', 'ACDE')
    cfg = Config(input_dir=inp, output_dir=out)
    job = scan(cfg)[0][0]
    got = []

    def worker():
        try:
            with claim(job, cfg):
                got.append('in')
                time.sleep(0.2)
        except FileExistsError:
            got.append('denied')

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert got.count('in') == 1
    assert got.count('denied') == 1
    assert (job.state_dir(cfg) / 'done.json').is_file()
    assert not (job.state_dir(cfg) / 'lock').exists()


def test_try_begin_claim_skips_done_and_locked(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    inp.mkdir(); out.mkdir()
    _job_json(inp, 'j', 'ACDE')
    cfg = Config(input_dir=inp, output_dir=out)
    job = scan(cfg)[0][0]
    extra, reason = try_begin_claim(job, cfg)
    assert extra is not None and reason is None
    extra2, reason2 = try_begin_claim(job, cfg)
    assert extra2 is None and reason2 == 'locked'
    end_claim(job, cfg, extra, error=None)
    extra3, reason3 = try_begin_claim(job, cfg)
    assert extra3 is None and reason3 == 'done'


def test_claim_failure_writes_failed(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    inp.mkdir(); out.mkdir()
    _job_json(inp, 'j', 'ACDE')
    cfg = Config(input_dir=inp, output_dir=out)
    job = scan(cfg)[0][0]
    with pytest.raises(RuntimeError):
        with claim(job, cfg):
            raise RuntimeError('boom')
    failed = job.state_dir(cfg) / 'failed.json'
    assert failed.is_file()
    assert 'boom' in failed.read_text()
    jobs, _ = scan(cfg)
    assert jobs == []  # skipped until --retry-failed
    cfg2 = Config(input_dir=inp, output_dir=out, retry_failed=True)
    assert len(scan(cfg2)[0]) == 1


def test_stale_lock_reclaim(tmp_path: Path):
    lock = tmp_path / 'lock'
    lock.write_text(json.dumps({'host': 'other', 'pid': 1}))
    os.utime(lock, (0, 0))
    cfg = Config(stale_lock_seconds=1)
    assert lock_reclaimable(lock, cfg)


def test_write_atomic_replaces(tmp_path: Path):
    p = tmp_path / 'x.json'
    write_atomic(p, 'one')
    write_atomic(p, 'two')
    assert p.read_text() == 'two'
    assert not list(tmp_path.glob('*.tmp'))


def test_config_roundtrip(tmp_path: Path):
    from fullFold.config import Config, config_from_dict, config_to_dict
    cfg = Config(input_dir=tmp_path / 'in', prefetch=3, gpus=('0', '1'))
    got = config_from_dict(config_to_dict(cfg))
    assert got.prefetch == 3
    assert got.gpus == ('0', '1')
    assert got.input_dir == tmp_path / 'in'
    assert got.buckets == cfg.buckets


def test_scan_order_independent_of_mtime(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    inp.mkdir(); out.mkdir()
    _job_json(inp, 'm', 'ACDEACDE')
    time.sleep(0.02)
    _job_json(inp, 'k', 'ACDE')
    cfg = Config(input_dir=inp, output_dir=out)
    names = [j.name for j in scan(cfg)[0]]
    assert names == ['k', 'm']
