"""FeaturePool bookkeeping (thread executor stands in for the spawned processes) and the --feat-workers / --no-fast-featurisation flags."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fullFold import featpool
from fullFold.cli import main as cli_main
from fullFold.config import Config
from fullFold.jobs import Job


def _job(name: str, n_seeds: int = 4) -> Job:
    return Job(path=Path(f'/x/{name}.json'), name=name, sha256='a', tokens=8, bucket=128,
               seeds=tuple(range(1, n_seeds + 1)), exact=True)


def _batch(job, seed, shape):
    return {'x': np.full((3,), seed, dtype=np.int32), 'name': job.name, 'shape': shape}


def _pool(monkeypatch, chunks, *, child=None, stock=None, **kw):
    calls = {'child': [], 'stock': []}

    def fake_child(job, seed, shape, cfg):
        calls['child'].append((job.job_id, seed))
        if child:
            return child(job, seed, shape), 0.01, 1
        return _batch(job, seed, shape), 0.01, 1

    def fake_stock(job, seed, shape):
        calls['stock'].append((job.job_id, seed))
        return stock(job, seed, shape) if stock else _batch(job, seed, shape)

    monkeypatch.setattr(featpool, '_child_feat', fake_child)
    pool = featpool.FeaturePool(chunks, Config(), stock_featurise=fake_stock, workers=2, **kw)
    pool.pool = ThreadPoolExecutor(2)
    return pool, calls


def test_items_come_from_pool_in_any_order(monkeypatch):
    a, b, c = _job('a'), _job('b'), _job('c')
    pool, calls = _pool(monkeypatch, [(a, 128), (b, 128), (c, 128)], verify=0)
    # the worker asks for the next job's first seed before the current job's last seeds
    order = [(a, 1), (b, 1), (a, 2), (a, 3), (a, 4), (c, 1), (b, 2), (b, 3), (b, 4), (c, 2), (c, 3), (c, 4)]
    for job, seed in order:
        got = pool.featurise(job, seed, 128)
        assert got['name'] == job.name and int(got['x'][0]) == seed
    assert calls['stock'] == [] and pool.stats['fallbacks'] == 0 and pool.stats['items'] == 12
    pool.close()


def test_lookahead_is_bounded(monkeypatch):
    jobs = [_job(f'j{i}') for i in range(10)]
    pool, calls = _pool(monkeypatch, [(j, 128) for j in jobs], verify=0, ahead=3)
    pool.featurise(jobs[0], 1, 128)
    assert len(set(calls['child'])) <= 4  # item 0 plus 3 ahead (all done by now: the executor is quick, but none beyond)
    pool.close()


def test_unknown_item_and_wrong_shape_fall_back(monkeypatch):
    a = _job('a')
    pool, calls = _pool(monkeypatch, [(a, 128)], verify=0)
    pool.featurise(_job('zzz'), 1, 128)
    pool.featurise(a, 1, 256)
    assert calls['stock'] == [('zzz', 1), ('a', 1)] and pool.stats['fallbacks'] == 2
    pool.close()


def test_child_error_falls_back_in_process(monkeypatch):
    a = _job('a')

    def boom(job, seed, shape):
        raise ValueError('bad msa')
    pool, calls = _pool(monkeypatch, [(a, 128)], child=boom, verify=0)
    got = pool.featurise(a, 1, 128)
    assert got['name'] == 'a' and calls['stock'] == [('a', 1)] and pool.stats['fallbacks'] == 1
    pool.close()


def test_first_items_are_checked_and_a_mismatch_switches_the_pool_off(monkeypatch):
    a = _job('a')
    pool, calls = _pool(monkeypatch, [(a, 128)], child=lambda j, s, sh: {**_batch(j, s, sh), 'x': np.zeros(3, np.int32)}, verify=2)
    got = pool.featurise(a, 1, 128)  # mismatch: the in-process result is returned
    assert int(got['x'][0]) == 1 and pool.off
    pool.featurise(a, 2, 128)  # served in-process from now on
    assert calls['stock'] == [('a', 1), ('a', 2)]


def test_matching_items_pass_the_check(monkeypatch):
    a = _job('a')
    pool, calls = _pool(monkeypatch, [(a, 128)], verify=2)
    for s in (1, 2, 3):
        pool.featurise(a, s, 128)
    assert not pool.off and pool.stats['verified'] == 2 and calls['stock'] == [('a', 1), ('a', 2)]
    pool.close()


def test_skipped_jobs_are_released(monkeypatch):
    jobs = [_job(f'j{i}', 1) for i in range(40)]
    pool, _ = _pool(monkeypatch, [(j, 128) for j in jobs], verify=0, ahead=2)
    pool.featurise(jobs[0], 1, 128)
    pool.featurise(jobs[39], 1, 128)  # 1..38 were skipped
    assert len(pool.futs) <= pool.margin + pool.ahead + 1
    pool.close()


def test_batches_equal():
    a = {'x': np.arange(3), 'y': [np.ones(2)], 'z': object()}
    assert featpool.batches_equal(a, {'x': np.arange(3), 'y': [np.ones(2)], 'z': a['z']})
    assert not featpool.batches_equal(a, {**a, 'x': np.arange(3) + 1})
    assert not featpool.batches_equal({'x': np.zeros(2, np.int32)}, {'x': np.zeros(2, np.int64)})


def test_cli_flags_reach_the_config(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr('fullFold.engine.cmd_plan', lambda cfg: seen.setdefault('cfg', cfg) and 0)
    cli_main(['plan', '--input-dir', str(tmp_path), '--output-dir', str(tmp_path / 'o')])
    assert seen['cfg'].feat_workers == 2 and seen['cfg'].fast_featurisation is True
    seen.clear()
    cli_main(['plan', '--input-dir', str(tmp_path), '--output-dir', str(tmp_path / 'o'), '--feat-workers', '5',
              '--no-fast-featurisation'])
    assert seen['cfg'].feat_workers == 5 and seen['cfg'].fast_featurisation is False
