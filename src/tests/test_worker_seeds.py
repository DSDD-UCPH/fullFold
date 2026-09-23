"""Per-seed pipeline, bounded queue, streamed ranking output."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from pathlib import Path

from fullFold.config import Config
from fullFold.jobs import Job
from fullFold.worker import gpu_stdio, padded_len, run_jobs, run_pipeline, worker_main


def _job(tmp: Path, name: str, n_seeds: int, tokens=8) -> Job:
    p = tmp / f'{name}.json'
    p.write_text('{}')
    return Job(path=p, name=name, sha256='abc', tokens=tokens, bucket=128,
               seeds=tuple(range(n_seeds)), exact=True)


class FakeResult:
    def __init__(self, score):
        self.metadata = {'ranking_score': score}


def test_padded_len():
    import numpy as np
    assert padded_len({'aatype': np.zeros(256)}) == 256


def test_prefetch_bound_400_seeds(tmp_path: Path):
    job = _job(tmp_path, 'big', 400, tokens=8)
    cfg = Config(output_dir=tmp_path, prefetch=1)
    live: list[int] = []
    feat_times = []
    inf_times = []
    started = []

    def feat(j, seed, shape):
        started.append(('f', seed, time.perf_counter()))
        time.sleep(0.001)
        return {'aatype': type('A', (), {'shape': (shape,)})()}

    def infer(j, seed, batch):
        started.append(('i', seed, time.perf_counter()))
        inf_times.append(seed)
        time.sleep(0.001)
        return ([FakeResult(float(seed))], None, None)

    def wseed(**kw):
        feat_times.append(kw['seed'])

    def wfinal(**kw):
        pass

    stats = run_pipeline(
        [(job, s, 8) for s in job.seeds], cfg,
        featurise_fn=feat, infer_fn=infer,
        write_seed_fn=wseed, write_final_fn=wfinal, live=live,
    )
    assert stats['max_live'] <= cfg.prefetch + 1
    assert stats['jobs_ok'] == 1
    assert stats['compiles'] == 1
    # overlap: some featurise(k+1) starts before infer(k) ends
    f = {s: t for k, s, t in started if k == 'f'}
    i = {s: t for k, s, t in started if k == 'i'}
    overlaps = sum(1 for s in range(399) if f.get(s + 1, 0) < i.get(s, 0) + 0.001)
    assert overlaps >= 1


def test_write_job_outputs_ranking(tmp_path: Path):
    rank = {'scores': [], 'best_score': None, 'best_result': None}
    out = tmp_path / 'job'
    out.mkdir()
    for seed, score in ((1, 0.2), (2, 0.9), (3, 0.5)):
        result = FakeResult(score)
        rank['scores'].append((seed, 0, score))
        if rank['best_score'] is None or score > rank['best_score']:
            rank['best_score'] = score
            rank['best_result'] = result
    ranking = out / 'job_ranking_scores.csv'
    with ranking.open('w') as f:
        cw = csv.writer(f)
        cw.writerow(['seed', 'sample', 'ranking_score'])
        cw.writerows(rank['scores'])
    rows = list(csv.DictReader(ranking.open()))
    assert [r['seed'] for r in rows] == ['1', '2', '3']
    assert rank['best_score'] == 0.9
    assert rank['best_result'].metadata['ranking_score'] == 0.9


def test_shape_mismatch_fails_job(tmp_path: Path):
    job = _job(tmp_path, 'x', 1, tokens=8)
    cfg = Config(output_dir=tmp_path, prefetch=0)
    stats = run_pipeline(
        [(job, 0, 128)], cfg,
        featurise_fn=lambda j, s, sh: {'aatype': type('A', (), {'shape': (64,)})()},
        infer_fn=lambda j, s, b: ([FakeResult(1.0)], None, None),
        write_seed_fn=lambda **k: None,
        write_final_fn=lambda **k: None,
    )
    assert stats['jobs_fail'] == 1
    assert stats['jobs_ok'] == 0


def test_worker_main_strips_argv_before_probe(monkeypatch, tmp_path: Path):
    seen = {}

    def fake_probe(gpu, cfg):
        seen['argv'] = list(sys.argv)
        seen['gpu'] = gpu
        seen['flash'] = cfg.flash_attention
        return [1.0, 1.0, 1.0, 1.0]

    monkeypatch.setattr('fullFold.worker.run_probe', fake_probe)
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--probe', '--gpu', '3', '--model-dir', 'm',
        '--output-dir', str(tmp_path),
    ])
    rc = worker_main([
        '--probe', '--gpu', '3', '--model-dir', 'm',
        '--output-dir', str(tmp_path), '--flash-attention', 'xla',
    ])
    assert rc == 0
    assert seen['argv'] == ['prog']
    assert seen['gpu'] == '3'
    assert seen['flash'] == 'xla'


def _batch(shape):
    return {'aatype': type('A', (), {'shape': (shape,)})()}


def test_next_job_prepared_during_infer(tmp_path: Path):
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b', 'c')]
    cfg = Config(output_dir=tmp_path, prefetch=1)
    feat_t, inf_end = {}, {}

    def feat(j, seed, shape):
        feat_t[j.name] = time.perf_counter()
        time.sleep(0.03)
        return _batch(shape)

    def infer(j, seed, batch):
        if j.name == 'a':
            lock = jobs[1].state_dir(cfg) / 'lock'
            deadline = time.perf_counter() + 0.2
            while time.perf_counter() < deadline and not lock.is_file():
                time.sleep(0.005)
            assert lock.is_file()
            assert not (jobs[1].state_dir(cfg) / 'done.json').is_file()
        time.sleep(0.08)
        inf_end[j.name] = time.perf_counter()
        return ([FakeResult(1.0)], None, None)

    stats = run_jobs(
        [(j, 8) for j in jobs], cfg,
        featurise_fn=feat, infer_fn=infer,
        write_seed_fn=lambda **k: None, write_final_fn=lambda **k: None,
    )
    assert stats['jobs_ok'] == 3
    assert feat_t['b'] < inf_end['a']
    assert feat_t['c'] < inf_end['b']
    for j in jobs:
        d = j.state_dir(cfg)
        assert (d / 'done.json').is_file()
        assert not (d / 'lock').exists()


def test_job_prep_event_overlaps_previous_infer(tmp_path: Path):
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b')]
    cfg = Config(output_dir=tmp_path, prefetch=1)
    log = io.StringIO()

    def feat(j, seed, shape):
        time.sleep(0.02)
        return _batch(shape)

    def infer(j, seed, batch):
        time.sleep(0.08)
        return ([FakeResult(1.0)], None, None)

    run_jobs(
        [(j, 8) for j in jobs], cfg,
        featurise_fn=feat, infer_fn=infer,
        write_seed_fn=lambda **k: None, write_final_fn=lambda **k: None,
        log_fp=log,
    )
    events = [json.loads(line) for line in log.getvalue().splitlines() if line]
    prep_b = next(e for e in events if e['event'] == 'job_prep' and e['job'] == 'b')
    done_a = next(e for e in events if e['event'] == 'job_done' and e['job'] == 'a')
    start_b = next(e for e in events if e['event'] == 'job_start' and e['job'] == 'b')
    assert prep_b['ts'] < done_a['ts']
    assert prep_b['ts'] < start_b['ts']
    assert done_a['name'] == 'a'
    assert done_a['n_seeds'] == 1
    assert done_a['shape'] == 8
    assert done_a['tokens'] == 8
    assert done_a['infer_ms'] > 0
    seed_a = next(e for e in events if e['event'] == 'seed_done' and e['job'] == 'a')
    assert done_a['infer_ms'] == seed_a['infer_ms']


def test_prefetch_zero_no_job_overlap(tmp_path: Path):
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b')]
    cfg = Config(output_dir=tmp_path, prefetch=0)
    feat_t, inf_end = {}, {}

    def feat(j, seed, shape):
        feat_t[j.name] = time.perf_counter()
        time.sleep(0.02)
        return _batch(shape)

    def infer(j, seed, batch):
        time.sleep(0.02)
        inf_end[j.name] = time.perf_counter()
        return ([FakeResult(1.0)], None, None)

    run_jobs(
        [(j, 8) for j in jobs], cfg,
        featurise_fn=feat, infer_fn=infer,
        write_seed_fn=lambda **k: None, write_final_fn=lambda **k: None,
    )
    assert feat_t['b'] >= inf_end['a']


def test_run_jobs_failed_writes_failed_json(tmp_path: Path):
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b', 'c')]
    cfg = Config(output_dir=tmp_path, prefetch=1)

    def infer(j, seed, batch):
        if j.name == 'b':
            raise RuntimeError('boom')
        return ([FakeResult(1.0)], None, None)

    stats = run_jobs(
        [(j, 8) for j in jobs], cfg,
        featurise_fn=lambda j, s, sh: _batch(sh), infer_fn=infer,
        write_seed_fn=lambda **k: None, write_final_fn=lambda **k: None,
    )
    assert stats['jobs_ok'] == 2
    assert stats['jobs_fail'] >= 1
    assert (jobs[0].state_dir(cfg) / 'done.json').is_file()
    assert (jobs[1].state_dir(cfg) / 'failed.json').is_file()
    assert not (jobs[1].state_dir(cfg) / 'done.json').is_file()
    assert (jobs[2].state_dir(cfg) / 'done.json').is_file()


def test_run_jobs_skips_done_without_failing(tmp_path: Path):
    job = _job(tmp_path, 'a', 1)
    cfg = Config(output_dir=tmp_path, prefetch=0)
    kw = dict(
        featurise_fn=lambda j, s, sh: _batch(sh),
        infer_fn=lambda j, s, b: ([FakeResult(1.0)], None, None),
        write_seed_fn=lambda **k: None, write_final_fn=lambda **k: None,
    )
    assert run_jobs([(job, 8)], cfg, **kw)['jobs_ok'] == 1
    stats = run_jobs([(job, 8)], cfg, **kw)
    assert stats['jobs_ok'] == 0
    assert stats['jobs_fail'] == 0


def test_run_jobs_skips_locked_prefetch(tmp_path: Path):
    from fullFold.jobs import begin_claim
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b')]
    cfg = Config(output_dir=tmp_path, prefetch=1)
    begin_claim(jobs[1], cfg)
    log = io.StringIO()
    stats = run_jobs(
        [(j, 8) for j in jobs], cfg,
        featurise_fn=lambda j, s, sh: _batch(sh),
        infer_fn=lambda j, s, b: ([FakeResult(1.0)], None, None),
        write_seed_fn=lambda **k: None, write_final_fn=lambda **k: None,
        log_fp=log,
    )
    assert stats['jobs_ok'] == 1
    assert stats['jobs_fail'] == 0
    events = [json.loads(line) for line in log.getvalue().splitlines() if line]
    skip = next(e for e in events if e['event'] == 'job_skip')
    assert skip['job'] == 'b'
    assert skip['reason'] == 'locked'


def test_gpu_stdio_captures_compile_warnings(tmp_path: Path):
    log = tmp_path / 'gpu0.log'
    with gpu_stdio(log):
        os.write(2, b'xla compile warning\n')
        print('python warn')
        sys.stderr.flush()
    text = log.read_text()
    assert 'xla compile warning' in text
    assert 'python warn' in text
    print('after')
    assert 'after' not in log.read_text()


def _bg_kw(extract, infer=None):
    return dict(
        featurise_fn=lambda j, s, sh: _batch(sh),
        infer_fn=infer or (lambda j, s, b: ([FakeResult(1.0)], None, None)),
        extract_fn=extract,
        write_seed_fn=lambda **k: None,
        write_final_fn=lambda **k: None,
    )


def test_background_extract_overlaps_same_bucket(tmp_path: Path):
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b')]
    cfg = Config(output_dir=tmp_path, prefetch=0, background_extract=True)
    extract_end, infer_start = {}, {}

    def infer(j, seed, batch):
        infer_start[j.name] = time.perf_counter()
        time.sleep(0.02)
        return ([FakeResult(1.0)], None, None)

    def extract(j, seed, batch, raw):
        time.sleep(0.06)
        extract_end[j.name] = time.perf_counter()
        return raw

    stats = run_jobs([(j, 8) for j in jobs], cfg, **_bg_kw(extract, infer))
    assert stats['jobs_ok'] == 2
    assert infer_start['b'] < extract_end['a']
    for j in jobs:
        assert (j.state_dir(cfg) / 'done.json').is_file()


def test_background_extract_drains_before_new_shape(tmp_path: Path):
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b')]
    cfg = Config(output_dir=tmp_path, prefetch=0, background_extract=True)
    extract_end, infer_start = {}, {}

    def infer(j, seed, batch):
        infer_start[j.name] = time.perf_counter()
        time.sleep(0.01)
        return ([FakeResult(1.0)], None, None)

    def extract(j, seed, batch, raw):
        time.sleep(0.05)
        extract_end[j.name] = time.perf_counter()
        return raw

    stats = run_jobs(
        [(jobs[0], 8), (jobs[1], 16)], cfg, **_bg_kw(extract, infer))
    assert stats['jobs_ok'] == 2
    assert infer_start['b'] >= extract_end['a']


def test_background_extract_failure_writes_failed_json(tmp_path: Path):
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b', 'c')]
    cfg = Config(output_dir=tmp_path, prefetch=0, background_extract=True)

    def extract(j, seed, batch, raw):
        if j.name == 'b':
            raise RuntimeError('boom')
        return raw

    stats = run_jobs([(j, 8) for j in jobs], cfg, **_bg_kw(extract))
    assert stats['jobs_ok'] == 2
    assert stats['jobs_fail'] >= 1
    assert (jobs[0].state_dir(cfg) / 'done.json').is_file()
    assert (jobs[1].state_dir(cfg) / 'failed.json').is_file()
    assert not (jobs[1].state_dir(cfg) / 'done.json').is_file()
    assert (jobs[2].state_dir(cfg) / 'done.json').is_file()
