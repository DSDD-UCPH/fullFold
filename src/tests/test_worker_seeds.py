"""Per-seed pipeline, bounded queue, job markers, and worker plumbing."""

from __future__ import annotations

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


def _batch(shape):
    return {'aatype': type('A', (), {'shape': (shape,)})()}


def _kw(extract=None, infer=None, **over):
    kw = dict(
        featurise_fn=lambda j, s, sh: _batch(sh),
        infer_fn=infer or (lambda j, s, b: ([FakeResult(1.0)], None, None)),
        extract_fn=extract,
        write_seed_fn=lambda **k: None,
        write_final_fn=lambda **k: None,
    )
    kw.update(over)
    return kw


def _markers(cfg: Config, jobs) -> dict[str, str]:
    """Which marker each job has on disk right now."""
    out = {}
    for j in jobs:
        d = j.state_dir(cfg)
        for name in ('done.json', 'failed.json', 'lock'):
            if (d / name).is_file():
                out[j.name] = name
                break
    return out


def test_prefetch_bound_400_seeds(tmp_path: Path):
    job = _job(tmp_path, 'big', 400, tokens=8)
    cfg = Config(output_dir=tmp_path, prefetch=1)
    live: list[int] = []
    started = []

    def feat(j, seed, shape):
        started.append(('f', seed, time.perf_counter()))
        time.sleep(0.001)
        return _batch(shape)

    def infer(j, seed, batch):
        started.append(('i', seed, time.perf_counter()))
        time.sleep(0.001)
        return ([FakeResult(float(seed))], None, None)

    stats = run_pipeline(
        [(job, s, 8) for s in job.seeds], cfg, featurise_fn=feat, infer_fn=infer,
        write_seed_fn=lambda **kw: None, write_final_fn=lambda **kw: None, live=live)
    assert stats['max_live'] <= cfg.prefetch + 1
    assert stats['jobs_ok'] == 1 and stats['compiles'] == 1
    # overlap: some featurise(k+1) starts before infer(k) ends
    f = {s: t for k, s, t in started if k == 'f'}
    i = {s: t for k, s, t in started if k == 'i'}
    assert sum(1 for s in range(399) if f.get(s + 1, 0) < i.get(s, 0) + 0.001) >= 1


def test_a_batch_of_the_wrong_size_fails_the_job(tmp_path: Path):
    assert padded_len({'aatype': _batch(256)['aatype']}) == 256
    job = _job(tmp_path, 'x', 1, tokens=8)
    stats = run_pipeline(
        [(job, 0, 128)], Config(output_dir=tmp_path, prefetch=0),
        featurise_fn=lambda j, s, sh: _batch(64),
        infer_fn=lambda j, s, b: ([FakeResult(1.0)], None, None),
        write_seed_fn=lambda **k: None, write_final_fn=lambda **k: None)
    assert stats['jobs_fail'] == 1 and stats['jobs_ok'] == 0


def test_worker_main_strips_argv_before_probe(monkeypatch, tmp_path: Path):
    seen = {}

    def fake_probe(gpu, cfg):
        seen.update(argv=list(sys.argv), gpu=gpu, flash=cfg.flash_attention)
        return [1.0, 1.0, 1.0, 1.0]

    monkeypatch.setattr('fullFold.worker.run_probe', fake_probe)
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--probe', '--gpu', '3', '--model-dir', 'm', '--output-dir', str(tmp_path)])
    rc = worker_main([
        '--probe', '--gpu', '3', '--model-dir', 'm', '--output-dir', str(tmp_path),
        '--flash-attention', 'xla'])
    assert rc == 0 and seen == {'argv': ['prog'], 'gpu': '3', 'flash': 'xla'}


def test_next_job_is_prepared_while_the_current_one_infers(tmp_path: Path):
    jobs = [_job(tmp_path, n, 1) for n in ('a', 'b', 'c')]
    cfg = Config(output_dir=tmp_path, prefetch=1)
    log = io.StringIO()
    feat_t, inf_end = {}, {}

    def feat(j, seed, shape):
        feat_t[j.name] = time.perf_counter()
        time.sleep(0.03)
        return _batch(shape)

    def infer(j, seed, batch):
        if j.name == 'a':  # b is claimed and being prepared, but not finished
            lock = jobs[1].state_dir(cfg) / 'lock'
            deadline = time.perf_counter() + 0.2
            while time.perf_counter() < deadline and not lock.is_file():
                time.sleep(0.005)
            assert lock.is_file() and not (jobs[1].state_dir(cfg) / 'done.json').is_file()
        time.sleep(0.08)
        inf_end[j.name] = time.perf_counter()
        return ([FakeResult(1.0)], None, None)

    stats = run_jobs([(j, 8) for j in jobs], cfg, log_fp=log,
                     **_kw(infer=infer, featurise_fn=feat))
    assert stats['jobs_ok'] == 3
    assert feat_t['b'] < inf_end['a'] and feat_t['c'] < inf_end['b']
    for j in jobs:
        assert (j.state_dir(cfg) / 'done.json').is_file()
        assert not (j.state_dir(cfg) / 'lock').exists()
    events = [json.loads(line) for line in log.getvalue().splitlines() if line]

    def first(kind, job):
        return next(e for e in events if e['event'] == kind and e['job'] == job)

    assert first('job_prep', 'b')['ts'] < first('job_done', 'a')['ts']
    assert first('job_prep', 'b')['ts'] < first('job_start', 'b')['ts']
    done_a = first('job_done', 'a')
    assert (done_a['name'], done_a['n_seeds'], done_a['shape'], done_a['tokens']) == (
        'a', 1, 8, 8)
    assert done_a['infer_ms'] > 0 and done_a['infer_ms'] == first('seed_done', 'a')['infer_ms']


def test_prefetch_zero_does_not_overlap_jobs(tmp_path: Path):
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

    run_jobs([(j, 8) for j in jobs], cfg, **_kw(infer=infer, featurise_fn=feat))
    assert feat_t['b'] >= inf_end['a']


def test_a_failing_job_writes_failed_json_and_the_rest_complete(tmp_path: Path):
    def infer(j, seed, batch):
        if j.name == 'b':
            raise RuntimeError('CUDA out of memory at bucket 4096')
        return ([FakeResult(1.0)], None, None)

    def extract(j, seed, batch, raw):
        if j.name == 'b':
            raise RuntimeError('extract broke')
        return raw

    for label, background, kw, message in (
            ('infer fails', False, dict(infer=infer), 'out of memory'),
            ('extract fails', True, dict(extract=extract), 'extract broke')):
        sub = tmp_path / label.replace(' ', '-')
        sub.mkdir()
        jobs = [_job(sub, n, 1) for n in ('a', 'b', 'c')]
        cfg = Config(output_dir=sub, prefetch=1, background_extract=background)
        stats = run_jobs([(j, 8) for j in jobs], cfg, **_kw(**kw))
        assert stats['jobs_ok'] == 2 and stats['jobs_fail'] >= 1, label
        assert _markers(cfg, jobs) == {
            'a': 'done.json', 'b': 'failed.json', 'c': 'done.json'}, label
        failed = json.loads((jobs[1].state_dir(cfg) / 'failed.json').read_text())
        assert message in failed['error'], label  # the real cause, not just "job failed"


def test_finished_and_claimed_jobs_are_skipped_without_failing(tmp_path: Path):
    from fullFold.jobs import begin_claim
    job = _job(tmp_path, 'a', 1)
    cfg = Config(output_dir=tmp_path, prefetch=0)
    assert run_jobs([(job, 8)], cfg, **_kw())['jobs_ok'] == 1
    again = run_jobs([(job, 8)], cfg, **_kw())
    assert (again['jobs_ok'], again['jobs_fail']) == (0, 0)
    jobs = [_job(tmp_path, n, 1) for n in ('x', 'y')]
    cfg = Config(output_dir=tmp_path, prefetch=1)
    begin_claim(jobs[1], cfg)  # another worker holds y
    log = io.StringIO()
    stats = run_jobs([(j, 8) for j in jobs], cfg, log_fp=log, **_kw())
    assert (stats['jobs_ok'], stats['jobs_fail']) == (1, 0)
    skip = next(json.loads(line) for line in log.getvalue().splitlines()
                if '"job_skip"' in line)
    assert (skip['job'], skip['reason']) == ('y', 'locked')


def test_gpu_stdio_captures_compile_warnings(tmp_path: Path):
    log = tmp_path / 'gpu0.log'
    with gpu_stdio(log):
        os.write(2, b'xla compile warning\n')
        print('python warn')
        sys.stderr.flush()
    text = log.read_text()
    assert 'xla compile warning' in text and 'python warn' in text
    print('after')
    assert 'after' not in log.read_text()


def test_background_extract_overlaps_a_shape_but_drains_before_a_new_one(tmp_path: Path):
    for first, second, overlaps in ((8, 8, True), (8, 16, False)):
        sub = tmp_path / f'{first}-{second}'
        sub.mkdir()
        jobs = [_job(sub, n, 1) for n in ('a', 'b')]
        cfg = Config(output_dir=sub, prefetch=0, background_extract=True)
        extract_end, infer_start = {}, {}

        def infer(j, seed, batch):
            infer_start[j.name] = time.perf_counter()
            time.sleep(0.02)
            return ([FakeResult(1.0)], None, None)

        def extract(j, seed, batch, raw):
            time.sleep(0.06)
            extract_end[j.name] = time.perf_counter()
            return raw

        stats = run_jobs([(jobs[0], first), (jobs[1], second)], cfg, **_kw(extract, infer))
        assert stats['jobs_ok'] == 2
        if overlaps:  # same shape: the next inference starts while the last extract runs
            assert infer_start['b'] < extract_end['a']
        else:  # a new shape compiles: the pending raw result is written out first
            assert infer_start['b'] >= extract_end['a']
        assert _markers(cfg, jobs) == {'a': 'done.json', 'b': 'done.json'}


def test_markers_follow_each_job_not_the_end_of_the_bucket(tmp_path: Path):
    jobs = [_job(tmp_path, 'a', 1), _job(tmp_path, 'b', 3), _job(tmp_path, 'c', 1),
            _job(tmp_path, 'd', 1)]
    cfg = Config(output_dir=tmp_path, prefetch=1, background_extract=True)
    seen, b_done_at_write = {}, []

    def infer(j, seed, batch):
        time.sleep(0.15)  # let earlier jobs' extract and marker finish
        seen[j.name] = {n for n, m in _markers(cfg, jobs).items() if m == 'done.json'}
        return ([FakeResult(1.0)], None, None)

    def b_has_marker():
        return (jobs[1].state_dir(cfg) / 'done.json').is_file()

    def write_seed(**kw):
        time.sleep(0.02)
        if kw['job_name'] == 'b':
            b_done_at_write.append((kw['seed'], b_has_marker()))

    def write_final(**kw):
        if kw['job_name'] == 'b':
            b_done_at_write.append(('final', b_has_marker()))

    stats = run_jobs([(j, 8) for j in jobs], cfg,
                     **_kw(lambda j, s, b, raw: raw, infer,
                           write_seed_fn=write_seed, write_final_fn=write_final))
    assert stats['jobs_ok'] == 4
    assert seen['c'] >= {'a', 'b'} and seen['d'] >= {'a', 'b', 'c'}  # not all at the end
    # a many-seed job gets its marker only after its last seed and the final write
    assert [s for s, _ in b_done_at_write] == [0, 1, 2, 'final']
    assert not any(flag for _, flag in b_done_at_write)
    done = json.loads((jobs[1].state_dir(cfg) / 'done.json').read_text())
    assert done['seeds'] == [0, 1, 2]


def test_aborted_runs_hand_their_claims_back(tmp_path: Path, monkeypatch):
    from fullFold import worker

    def interrupt_on_b(j, seed, batch):
        if j.name == 'b':
            raise KeyboardInterrupt
        return ([FakeResult(1.0)], None, None)

    for prefetch in (0, 1):
        sub = tmp_path / f'p{prefetch}'
        sub.mkdir()
        jobs = [_job(sub, n, 1) for n in ('a', 'b', 'c')]
        cfg = Config(output_dir=sub, prefetch=prefetch, background_extract=True)
        t0 = time.perf_counter()
        try:
            run_jobs([(j, 8) for j in jobs], cfg, **_kw(infer=interrupt_on_b))
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('KeyboardInterrupt was swallowed')
        assert time.perf_counter() - t0 < 3.0  # it used to sit in a 5 s thread join
        # b (interrupted) and c (never ran) have no verdict, so a resume re-runs them
        assert _markers(cfg, jobs) == {'a': 'done.json'}, prefetch
    # an error outside any job: the failing job is failed, prepared jobs are released
    sub = tmp_path / 'pipeline'
    sub.mkdir()
    jobs = [_job(sub, n, 1) for n in ('a', 'b', 'c')]
    cfg = Config(output_dir=sub, prefetch=1, background_extract=True)
    real = worker.run_pipeline

    def boom(items, cfg_, **kw):
        if items[0][0].name == 'a':
            raise RuntimeError('pipeline broke')
        return real(items, cfg_, **kw)

    monkeypatch.setattr(worker, 'run_pipeline', boom)
    try:
        run_jobs([(j, 8) for j in jobs], cfg, **_kw())
    except RuntimeError:
        pass
    assert _markers(cfg, jobs) == {'a': 'failed.json'}


def test_fold_is_parsed_once_per_job(tmp_path: Path, monkeypatch):
    """Parsing and the CCD load used to repeat for every seed."""
    import types
    from fullFold import worker

    parses, ccds, seen = [], [], []
    fi = types.ModuleType('alphafold3.common.folding_input')
    fi.Input = type('Input', (), {'from_json': staticmethod(
        lambda text, path=None: (parses.append(path), object())[1])})
    for name, mod in (('alphafold3', types.ModuleType('alphafold3')),
                      ('alphafold3.common', types.ModuleType('alphafold3.common')),
                      ('alphafold3.common.folding_input', fi)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr('fullFold.runner.load_ccd', lambda fold: (ccds.append(fold), 'ccd')[1])
    monkeypatch.setattr(
        worker, 'featurise_one',
        lambda fold, seed, buckets, ccd=None, cfg=None: seen.append(ccd) or {})
    worker._FOLDS.clear()
    job = _job(tmp_path, 'many', 6)
    for seed in job.seeds:
        worker._default_feat(job, seed, 128)
    assert len(parses) == len(ccds) == 1 and seen == ['ccd'] * 6
    for n in ('x', 'y', 'z'):  # only the running job and the next one are kept
        worker._default_feat(_job(tmp_path, n, 1), 0, 128)
    assert len(worker._FOLDS) == worker._FOLDS_KEEP
    worker._FOLDS.clear()
