"""Scanning inputs, job ids, and the lock / marker protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fullFold import jobs as jobs_mod
from fullFold.config import Config, config_from_dict, config_to_dict, host_id, write_atomic
from fullFold.jobs import (
    _evict_stale, _pid_alive, begin_claim, claim, end_claim, lock_reclaimable,
    release_claim, scan, try_begin_claim,
)


def _job_json(tmp: Path, name: str, seq: str, seeds=(1,), filename: str | None = None) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / (filename or f'{name}.json')
    p.write_text(json.dumps({
        'name': name,
        'modelSeeds': list(seeds),
        'sequences': [{'protein': {'id': 'A', 'sequence': seq}}],
        'dialect': 'alphafold3',
        'version': 4,
    }))
    return p


def _first_job(tmp_path: Path, name: str = 'j', **cfg_kw):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _job_json(inp, name, 'ACDE')
    cfg = Config(input_dir=inp, output_dir=out, **cfg_kw)
    return scan(cfg)[0][0], cfg


# ---------------------------------------------------------------- scanning

def test_scan_orders_skips_and_resumes(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _job_json(inp, 'm', 'ACDEACDE')
    time.sleep(0.02)  # the newer file is the shorter one: order must not follow mtime
    _job_json(inp, 'k', 'ACDE')
    _job_json(inp, 'a', 'ACDEFGH')
    (inp / 'bad.json').write_text('{not json')
    (inp / 'z.json').write_text('[]')
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, rejected = scan(cfg)
    assert [j.name for j in jobs] == ['k', 'a', 'm']  # bucket, then tokens
    assert len(rejected) == 2
    keep = jobs[0]
    p = inp / 'k.json'
    assert keep.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    # a finished job is skipped on resume, until its input changes
    d = keep.state_dir(cfg)
    d.mkdir(parents=True)
    write_atomic(d / 'done.json', json.dumps({'sha256': keep.sha256}))
    assert [j.name for j in scan(cfg)[0]] == ['a', 'm']
    p.write_text(p.read_text().replace('ACDE', 'ACDG'))
    assert 'k' in [j.name for j in scan(cfg)[0]]
    # --json-path scans exactly that file
    only, rej = scan(Config(input_dir=inp, output_dir=out, json_path=inp / 'a.json'))
    assert rej == [] and [j.name for j in only] == ['a']
    # --num_seeds expands a single seed, and refuses to guess for several
    one = tmp_path / 'one'
    _job_json(one, 'a', 'ACDE', seeds=(7,))
    jobs, rej = scan(Config(input_dir=one, output_dir=out, num_seeds=3))
    assert rej == [] and jobs[0].seeds == (7, 8, 9)
    many = tmp_path / 'many'
    _job_json(many, 'a', 'ACDE', seeds=(1, 2))
    jobs, rej = scan(Config(input_dir=many, output_dir=out, num_seeds=3))
    assert jobs == [] and 'num_seeds' in rej[0]


def test_names_that_would_share_an_output_directory_are_rejected(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _job_json(inp, 'same', 'ACDE', filename='a.json')
    _job_json(inp, 'same', 'CCCC', filename='b.json')
    _job_json(inp, 'Same!', 'DDDD', filename='c.json')  # sanitises to 'Same'
    _job_json(inp, 'other', 'ACDE', filename='d.json')
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, rejected = scan(cfg)
    assert sorted(j.job_id for j in jobs) == ['other', 'same']
    assert {r.split(':')[0] for r in rejected} == {'b.json', 'c.json'}
    assert all('collides with a.json' in r for r in rejected)  # first file in sorted order wins
    # still rejected when the first one is already done: it must not be overwritten
    first = next(j for j in jobs if j.job_id == 'same')
    begin_claim(first, cfg)
    end_claim(first, cfg, {}, error=None)
    jobs, rejected = scan(cfg)
    assert [j.job_id for j in jobs] == ['other'] and len(rejected) == 2
    # names that cannot be a directory, or that would take the scheduler's own
    bad = tmp_path / 'bad'
    for i, name in enumerate(('..', '###', '_af3sched')):
        _job_json(bad, name, 'ACDE', filename=f'{i}.json')
    _job_json(bad, 'fine', 'ACDE')
    jobs, rejected = scan(Config(input_dir=bad, output_dir=out))
    assert [j.job_id for j in jobs] == ['fine'] and len(rejected) == 3


# ---------------------------------------------------------------- claims

def test_claim_is_exclusive_and_leaves_markers(tmp_path: Path):
    job, cfg = _first_job(tmp_path)
    got = []

    def worker():
        try:
            with claim(job, cfg):
                got.append('in')
                time.sleep(0.2)
        except FileExistsError:
            got.append('denied')

    t1, t2 = threading.Thread(target=worker), threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert sorted(got) == ['denied', 'in']
    assert (job.state_dir(cfg) / 'done.json').is_file()
    assert not (job.state_dir(cfg) / 'lock').exists()
    extra, reason = try_begin_claim(job, cfg)
    assert extra is None and reason == 'done'
    # a fresh job: first claim wins, the second is told why it lost
    other, cfg2 = _first_job(tmp_path / 'b')
    extra, reason = try_begin_claim(other, cfg2)
    assert extra is not None and reason is None
    assert try_begin_claim(other, cfg2) == (None, 'locked')
    end_claim(other, cfg2, extra, error=None)
    assert try_begin_claim(other, cfg2) == (None, 'done')


def test_a_failed_claim_is_skipped_until_retry(tmp_path: Path):
    job, cfg = _first_job(tmp_path)
    with pytest.raises(RuntimeError):
        with claim(job, cfg):
            raise RuntimeError('boom')
    failed = job.state_dir(cfg) / 'failed.json'
    assert failed.is_file() and 'boom' in failed.read_text()
    assert scan(cfg)[0] == []
    assert len(scan(Config(input_dir=cfg.input_dir, output_dir=cfg.output_dir,
                           retry_failed=True))[0]) == 1


def test_a_running_job_keeps_its_lock_through_the_heartbeat(tmp_path: Path):
    """A job that outlives stale_lock_seconds is alive, not stale."""
    job, cfg = _first_job(tmp_path, stale_lock_seconds=1)
    lock = job.state_dir(cfg) / 'lock'
    extra = begin_claim(job, cfg)
    try:
        assert lock in jobs_mod._HELD
        time.sleep(1.3)  # older than the 1 s window; only a heartbeat keeps it fresh
        assert time.time() - lock.stat().st_mtime < 1.0
        assert not lock_reclaimable(lock, cfg)
        assert try_begin_claim(job, cfg) == (None, 'locked')
    finally:
        end_claim(job, cfg, extra, error=None)
    assert lock not in jobs_mod._HELD  # the heartbeat stops with the claim
    begin_claim(job, cfg)
    release_claim(job, cfg)
    assert lock not in jobs_mod._HELD and not lock.exists()


def test_which_locks_may_be_taken_over(tmp_path: Path, monkeypatch):
    cfg = Config(stale_lock_seconds=900)
    lock = tmp_path / 'lock'

    def write(content, age=0):
        lock.write_text(content)
        old = time.time() - age
        os.utime(lock, (old, old))

    me = json.dumps({'host': host_id(), 'pid': os.getpid()})
    write(me)
    assert not lock_reclaimable(lock, cfg)  # a live owner on this host
    write(me, age=1000)
    assert lock_reclaimable(lock, cfg)  # no heartbeat for longer than the window
    write(json.dumps({'host': 'elsewhere', 'pid': 1}), age=1000)
    assert lock_reclaimable(lock, Config(stale_lock_seconds=1))
    child = subprocess.Popen([sys.executable, '-c', 'pass'])
    child.wait()
    dead = json.dumps({'host': host_id(), 'pid': child.pid})
    write(dead)
    assert lock_reclaimable(lock, cfg)  # owner died on this host: at once
    write('')
    assert not lock_reclaimable(lock, cfg)  # may be mid-write: short grace
    write('', age=30)
    assert lock_reclaimable(lock, cfg)  # but not forever after a crash
    assert lock_reclaimable(tmp_path / 'gone', cfg)

    def deny(pid, sig):
        raise PermissionError

    monkeypatch.setattr(os, 'kill', deny)
    assert _pid_alive(12345)  # owned by another user, so it exists
    monkeypatch.undo()
    # and a claimant really does take over a dead owner's lock
    job, cfg = _first_job(tmp_path / 'takeover')
    job.state_dir(cfg).mkdir(parents=True)
    (job.state_dir(cfg) / 'lock').write_text(dead)
    extra, reason = try_begin_claim(job, cfg)
    assert extra is not None and reason is None
    assert json.loads((job.state_dir(cfg) / 'lock').read_text())['pid'] == os.getpid()
    release_claim(job, cfg)


def test_taking_over_a_lock_is_race_safe(tmp_path: Path, monkeypatch):
    cfg = Config(stale_lock_seconds=1)
    lock = tmp_path / 'lock'
    lock.write_text(json.dumps({'host': 'elsewhere', 'pid': 1}))
    os.utime(lock, (0, 0))
    _evict_stale(lock, cfg)  # a stale lock goes, leaving no debris
    assert not lock.exists() and not list(tmp_path.glob('lock.stale.*'))
    # another claimant wins between our verdict and our rename: their lock is put back
    lock.write_text('stale-owner')
    fresh = json.dumps({'host': host_id(), 'pid': os.getpid(), 'started_at': 1})

    def verdict_then_race(path, cfg_):
        lock.write_text(fresh)
        return True

    monkeypatch.setattr(jobs_mod, 'lock_reclaimable', verdict_then_race)
    _evict_stale(lock, Config(stale_lock_seconds=900))
    assert lock.read_text() == fresh and not list(tmp_path.glob('lock.stale.*'))


def test_config_helpers(tmp_path: Path):
    p = tmp_path / 'x.json'
    write_atomic(p, 'one')
    write_atomic(p, 'two')
    assert p.read_text() == 'two' and not list(tmp_path.glob('*.tmp'))
    cfg = Config(input_dir=tmp_path / 'in', prefetch=3, gpus=('0', '1'))
    got = config_from_dict(config_to_dict(cfg))
    assert (got.prefetch, got.gpus, got.input_dir) == (3, ('0', '1'), tmp_path / 'in')
    assert got.buckets == cfg.buckets
