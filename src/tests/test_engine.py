"""Planning with stubbed GPUs and benches, reporting, and run control."""

from __future__ import annotations

import io
import json
import signal
from pathlib import Path

import pytest

from fullFold import engine
from fullFold.benchmark import Bench, Gpu, COST_UNIT_US, unprobed_bench
from fullFold.config import Config
from fullFold.engine import (
    consume_worker_events, event_log_offsets, format_gpu_progress, make_plan,
    progress_lines, render_progress, report, report_scan, run_workers,
    summarise_run, write_ledger, write_manifests,
)
from fullFold.jobs import Job, scan
from fullFold.scheduling import Group


def _write_job(d: Path, name: str, n: int, seeds=(1,)):
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{name}.json').write_text(json.dumps({
        'name': name, 'modelSeeds': list(seeds),
        'sequences': [{'protein': {'id': 'A', 'sequence': 'A' * n}}],
        'dialect': 'alphafold3', 'version': 4,
    }))


def _benches():
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'L40S', 0)
    b = Bench((1000, 200, 200, 200), 200.0, 400.0, False, 'h', 'k')
    return [(g0, b), (g1, b)]


def _plan(tmp: Path, sizes: dict[str, int], benches, *, seeds=None, **cfg_kw):
    """Write one job per ``sizes`` entry, scan, and plan: (cfg, jobs, groups, meta)."""
    inp = tmp / 'in'
    for name, n in sizes.items():
        _write_job(inp, name, n, seeds=(seeds or {}).get(name, (1,)))
    cfg = Config(input_dir=inp, output_dir=tmp / 'out', **cfg_kw)
    jobs, _ = scan(cfg)
    groups, meta = make_plan(cfg, jobs, benches)
    return cfg, jobs, groups, meta


def _primary_shapes(groups):
    return [g.shape for gr in groups for g in gr if not g.shared]


def test_plan_compiles_equal_distinct_shapes_and_manifests_are_stable(tmp_path: Path):
    cfg, jobs, groups, meta = _plan(
        tmp_path, {'s': 10, 'm': 200, 'l': 300}, _benches())
    out = tmp_path / 'out'
    gpus = [g for g, _ in _benches()]
    write_manifests(cfg, jobs, gpus, groups, meta)
    for gpu_groups, n in zip(groups, meta['compiles_per_gpu']):
        shapes = [g.shape for g in gpu_groups if not g.shared]
        assert n == len(shapes) == len(set(shapes))
        assert shapes == sorted(shapes)
    first = (out / '_af3sched' / 'plan.json').read_bytes()
    write_manifests(cfg, jobs, gpus, groups, meta)
    assert (out / '_af3sched' / 'plan.json').read_bytes() == first
    man = json.loads((out / '_af3sched' / 'manifest-gpu0.json').read_text())
    assert man['config']['prefetch'] == 1 and 'shared' in man['groups'][0]
    assert meta['probed'] is True and meta['hoists'] == [] and 'm_gpu' in meta['gpus'][0]


def test_make_plan_reads_the_benches(tmp_path: Path):
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'L40S', 0)
    # no probe: unit costs, no multiplier
    _, _, groups, meta = _plan(tmp_path / 'unprobed', {'s': 10}, [(g0, unprobed_bench())])
    assert groups and meta['probed'] is False
    assert meta['gpus'][0]['compile_us'] == meta['gpus'][0]['infer_us_1024'] == COST_UNIT_US
    assert 'm_gpu' not in meta['gpus'][0]
    # t3 and t4 disagree, so S is unusable: round-robin
    bad = Bench((500, 400, 100, 200), 150.0, 600.0, True, 'h', 'k')
    _, _, _, meta = _plan(tmp_path / 'noisy', {'a': 8, 'b': 8}, [(g0, bad)])
    assert meta['policy'] == 'roundrobin' and meta['contaminated']
    # a warm compile cache inverts R but S is fine: still planned by search
    warm = Bench((10, 10, 100, 100), 100.0, -180.0, False, 'h', 'k')
    _, _, _, meta = _plan(tmp_path / 'warm', {'a': 8, 'b': 8}, [(g0, warm), (g1, warm)])
    assert meta['policy'] == 'search' and not meta['contaminated']
    # one 400-seed job cannot be split: it is the makespan floor
    _, _, _, meta = _plan(
        tmp_path / 'floor', {'tiny': 8, 'huge': 8}, _benches(),
        seeds={'huge': tuple(range(400))})
    assert meta['floor_job'] == 'huge' and meta['floor_us'] == meta['makespan_us']


def test_faster_gpu_gets_more_jobs(tmp_path: Path):
    g0 = Gpu(0, '0', 'u0', 'pci0', 'RTX 5090', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'RTX 5080', 0)
    fast = Bench((70_000, 67_000, 45_000, 45_000), 45_000.0, 47_000.0, False, 'h', 'k')
    slow = Bench((110_000, 105_000, 74_000, 74_000), 74_000.0, 67_000.0, False, 'h', 'k')
    _, _, groups, meta = _plan(
        tmp_path, {f'j{i}': 64 for i in range(20)}, [(g0, fast), (g1, slow)],
        bucket_mode='ladder')
    assert meta['policy'] == 'search'
    n0 = sum(len(g.job_ids) for g in groups[0] if not g.shared)
    n1 = sum(len(g.job_ids) for g in groups[1] if not g.shared)
    assert n0 > n1 and n0 + n1 == 20
    assert meta['gpus'][0]['infer_us_1024'] < meta['gpus'][1]['infer_us_1024']
    for thief, victim in ((0, 1), (1, 0)):  # tails are stolen last-job-first
        stolen = [j for g in groups[thief] if g.shared for j in g.job_ids]
        last = [j for g in groups[victim] if not g.shared for j in g.job_ids][-1]
        assert not stolen or stolen[0] == last


def test_compiled_shapes_follow_the_mode_and_reference(tmp_path: Path):
    # fast mode compiles multiples of 64
    _, _, groups, meta = _plan(
        tmp_path / 'tiles', {'tiny': 9}, _benches(), mode='fast', reference='fast')
    assert _primary_shapes(groups) == [64] and meta['mode'] == 'fast'
    # ...unless the buckets were given explicitly
    _, _, groups, _ = _plan(
        tmp_path / 'explicit', {'tiny': 9}, _benches(), mode='fast', reference='fast',
        buckets=(100, 200), buckets_explicit=True)
    assert _primary_shapes(groups) == [100]
    # the fast reference alone schedules on its own table
    _, jobs, groups, meta = _plan(
        tmp_path / 'ref', {'tiny': 9}, _benches(), reference='fast')
    assert meta['reference'] == 'fast' and jobs[0].tokens == 9
    assert _primary_shapes(groups) == [16] and 10 not in meta['shapes']


def _three_gpus():
    fast = Bench((70_000, 67_000, 45_000, 45_000), 45_000.0, 47_000.0, False, 'h', 'k')
    slow = Bench((110_000, 105_000, 74_000, 74_000), 74_000.0, 67_000.0, False, 'h', 'k')
    return [
        (Gpu(0, '0', 'u0', 'pci0', 'RTX 5090', 0), fast),
        (Gpu(1, '1', 'u1', 'pci1', 'RTX 5080', 0), slow),
        (Gpu(2, '2', 'u2', 'pci2', 'RTX 5090', 0), fast),
    ]


def _search(tmp: Path, n_jobs: int, **kw):
    sizes = {f'j{i:02d}': (24, 40, 64, 96, 150, 220)[i % 6] for i in range(n_jobs)}
    seeds = {n: (1, 2)[i % 2:] for i, n in enumerate(sizes)}
    cfg, jobs, groups, meta = _plan(
        tmp, sizes, _three_gpus(), seeds=seeds, bucket_mode='ladder', **kw)
    return cfg, jobs, groups, meta


def test_policies_and_what_the_plan_records(tmp_path: Path, capsys):
    gpus = [g for g, _ in _three_gpus()]
    # search: records its bound, plans every job once, is deterministic, reports the bound
    cfg, jobs, groups, meta = _search(tmp_path / 'search', 30)
    assert meta['policy'] == meta['split_mode'] == 'search'
    info = meta['planner']
    assert info['effort'] == 1 and info['stage'] == 'search'
    assert 0 < info['lower_bound_us'] <= meta['makespan_us']
    primary = [j for gr in groups for g in gr if not g.shared for j in g.job_ids]
    assert sorted(primary) == sorted(j.job_id for j in jobs)
    again, meta2 = make_plan(cfg, jobs, _three_gpus())
    assert again == groups and meta2 == meta
    json.dumps(meta, sort_keys=True)
    report(groups, meta, gpus, jobs)
    out = capsys.readouterr().out
    assert 'policy=search' in out and 'no plan can beat' in out
    assert 'duplicated across GPUs' not in out
    # few jobs are solved exactly, and every effort level is accepted
    _, _, _, few = _search(tmp_path / 'few', 6)
    assert few['split_mode'] == few['planner']['stage'] == 'exact'
    _, _, _, quick = _search(tmp_path / 'quick', 30, plan_effort=0)
    assert quick['planner']['effort'] == 0
    # round-robin has no planner block, and the search is not worse than it
    _, rr_jobs, _, rr = _search(tmp_path / 'rr', 20, policy='roundrobin')
    assert rr['policy'] == rr['split_mode'] == 'roundrobin' and 'planner' not in rr
    _, _, _, searched = _search(tmp_path / 'rr-search', 20)
    assert searched['makespan_us'] <= rr['makespan_us']
    # the old contiguous planner is gone
    cfg, jobs, _, _ = _search(tmp_path / 'old', 6)
    with pytest.raises(ValueError, match='contiguous'):
        make_plan(Config(**{**cfg.__dict__, 'policy': 'contiguous'}), jobs, _three_gpus())


def _job(name, tokens, bucket, seeds=(1,)):
    return Job(path=Path(name), name=name, sha256='x', tokens=tokens, bucket=bucket,
               seeds=seeds, exact=True)


def test_report_prints_per_gpu_bucket_order(capsys):
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'L40S', 0)
    jobs = [_job('a', 10, 128), _job('b', 20, 128, (1, 2)), _job('c', 400, 512)]
    groups = [
        [Group(shape=128, job_ids=('a', 'b')), Group(shape=512, job_ids=('c',)),
         Group(shape=512, job_ids=('c',), shared=True)],
        [],
    ]
    inf1024 = 59_423_000
    meta = {
        'policy': 'roundrobin', 'makespan_us': inf1024, 'floor_us': 0, 'floor_job': '',
        'gpus': [
            {'slot': 0, 'infer_us_1024': inf1024, 'compile_us': 0, 'm_gpu': 1.0},
            {'slot': 1, 'infer_us_1024': inf1024, 'compile_us': 0, 'm_gpu': 1.23},
        ],
    }
    report(groups, meta, [g0, g1], jobs)
    out = capsys.readouterr().out
    assert 'estimated total running time=' in out
    assert 'GPU 0 (A100): 3 jobs, m_gpu=1.0' in out
    assert '1. bucket 128: 2 jobs, ' in out and '/job, total ' in out
    assert '2. bucket 512: 1 jobs, ' in out
    assert out.index('bucket 128') < out.index('bucket 512')
    assert 'GPU 1 (L40S): 0 jobs, m_gpu=1.23' in out and '(no buckets)' in out
    assert 'steal tail: 1 jobs from other GPUs' in out


def test_report_relative_cost_when_unprobed(capsys):
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    jobs = [_job('a', 10, 128), _job('b', 20, 128, (1, 2))]
    meta = {
        'policy': 'roundrobin', 'makespan_us': 1_500_000, 'floor_us': 0,
        'floor_job': '', 'probed': False,
        'gpus': [{'slot': 0, 'infer_us_1024': 1_000_000, 'compile_us': 1_000_000}],
    }
    report([[Group(shape=128, job_ids=('a', 'b'))]], meta, [g0], jobs)
    out = capsys.readouterr().out
    assert 'relative total cost=1.5' in out and 'relative total cost=1.5s' not in out
    assert 'relative GPU cost:' in out and '1.0 = one 1024-token inference' in out
    assert 'fullfold benchmark' in out
    for absent in ('estimated total running time=', 'm_gpu=', 'estimated GPU time'):
        assert absent not in out


def _events(path: Path, *rows: dict) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def test_summarise_run(tmp_path: Path, capsys):
    cfg = Config(output_dir=tmp_path)
    summarise_run(cfg, {'makespan_us': 100_000_000, 'compiles_per_gpu': []}, 120.0)
    out = capsys.readouterr().out
    assert 'estimated running time: 100.0s' in out
    assert 'actual running time:    120.0s' in out and 'actual / estimated:     1.20x' in out
    summarise_run(
        cfg, {'makespan_us': 2_000_000, 'probed': False, 'compiles_per_gpu': []}, 12.0)
    out = capsys.readouterr().out
    assert 'actual running time:    12.0s' in out
    for absent in ('estimated running time', 'actual / estimated', 'relative total cost'):
        assert absent not in out
    # the event logs are appended to across runs: count only this run's compiles,
    # and ignore a line that is still being written
    root = tmp_path / '_af3sched'
    root.mkdir()
    log = root / 'gpu0.jsonl'
    _events(log, {'event': 'group_start', 'shape': 128})
    offsets = event_log_offsets(root, [Gpu(0, '0', 'u0', 'pci0', 'A100', 0)])
    assert offsets == {0: log.stat().st_size}
    _events(log, {'event': 'group_start', 'shape': 256})
    with log.open('a') as f:
        f.write('{"event": "gro')
    meta = {'makespan_us': 1, 'compiles_per_gpu': [1]}
    summarise_run(cfg, meta, 1.0, offsets)
    assert 'expected compiles=1 realised=1' in capsys.readouterr().out
    summarise_run(cfg, meta, 1.0)
    assert 'realised=2' in capsys.readouterr().out


def test_consume_worker_events_tracks_bucket_progress(tmp_path: Path):
    root = tmp_path / '_af3sched'
    root.mkdir()
    path = root / 'gpu0.jsonl'
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    jobs = [_job('a', 10, 128, (1, 2)), _job('c', 20, 128), _job('b', 200, 256)]
    groups = [[Group(shape=128, job_ids=('a', 'c')), Group(shape=256, job_ids=('b',))]]
    # an earlier run's events sit before the offset and must not count
    _events(path, {'event': 'job_done', 'job': 'old', 'shape': 128, 'tokens': 5})
    state: dict = {'offsets': dict(event_log_offsets(root, [g0]))}
    _events(path,
            {'event': 'job_skip', 'job': 'c', 'reason': 'locked', 'shape': 128},
            {'event': 'job_start', 'job': 'a', 'shape': 128},
            {'event': 'job_done', 'job': 'a', 'name': 'a', 'shape': 128,
             'tokens': 10, 'infer_ms': 8100, 'n_seeds': 2})
    assert consume_worker_events(root, [g0], jobs, state, groups)
    assert progress_lines([g0], state)[0] == format_gpu_progress(
        0, 'A100', 1, 3, bucket=128, tokens=10, job_idx=1, bucket_n=2)
    _events(path, {'event': 'job_done', 'job': 'b', 'name': 'b', 'shape': 256,
                   'tokens': 200, 'infer_ms': 15000, 'n_seeds': 1})
    assert consume_worker_events(root, [g0], jobs, state, groups)
    assert progress_lines([g0], state)[0] == format_gpu_progress(
        0, 'A100', 2, 3, bucket=256, tokens=200, job_idx=1, bucket_n=1)
    assert state['by_slot'][0]['done'] == {'a', 'b'}
    assert consume_worker_events(root, [g0], jobs, state, groups) is False


def test_progress_rendering(tmp_path: Path):
    assert format_gpu_progress(
        0, 'A100', 24, 40, bucket=256, tokens=198, job_idx=3, bucket_n=8) == (
        'GPU 0 (A100) [======>     ] jobs=24/40 current_bucket=256  '
        'job_tokens=198  jobs_in_bucket=3/8')
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'L40S', 0)
    state = {'by_slot': {
        0: {'done': {'a'}, 'started': {}, 'total': 40,
            'current': {'shape': 256, 'tokens': 198, 'job_idx': 3, 'bucket_n': 8}},
        1: {'done': set(), 'started': {}, 'current': None, 'total': 35},
    }}
    buf = io.StringIO()  # not a terminal: plain snapshots, no escape codes
    render_progress([g0, g1], state, tty=False, out=buf)
    text = buf.getvalue()
    for want in ('GPU 0 (A100)', 'jobs=1/40', 'current_bucket=256', 'job_tokens=198',
                 'jobs_in_bucket=3/8', 'GPU 1 (L40S)', 'jobs=0/35'):
        assert want in text
    assert '\033[' not in text
    render_progress([g0, g1], state, tty=False, out=buf)
    assert buf.getvalue().count('GPU 0 (A100)') == 2

    class Tty(io.StringIO):
        def isatty(self):
            return True

    tty_state = {'by_slot': {0: {'done': set(), 'started': {}, 'current': None, 'total': 4}}}
    buf = Tty()  # a terminal: the first paint is plain, later ones rewrite in place
    render_progress([g0], tty_state, tty=True, out=buf)
    assert tty_state['painted'] and '\033[' not in buf.getvalue()
    tty_state['by_slot'][0]['done'] = {'a'}
    render_progress([g0], tty_state, tty=True, out=buf)
    assert '\033[1A' in buf.getvalue() and '\033[K' in buf.getvalue()
    assert 'jobs=1/4' in buf.getvalue()


class _Proc:
    def __init__(self, rc=0):
        self.rc = rc
        self.signals = []

    def poll(self):
        return self.rc

    def wait(self):
        return self.rc

    def send_signal(self, s):
        self.signals.append(s)

    def terminate(self):
        pass


def _patch_popen(monkeypatch, factory):
    started = []

    def popen(cmd, **kw):
        started.append((cmd, kw))
        return factory(len(started) - 1)

    monkeypatch.setattr('fullFold.engine.subprocess.Popen', popen)
    monkeypatch.setattr('fullFold.engine.worker_environ', lambda *a, **k: {})
    return started


def _two_gpus(tmp_path):
    (tmp_path / '_af3sched').mkdir(exist_ok=True)
    return [Gpu(0, '0', 'u0', 'p0', 'A100', 0), Gpu(1, '1', 'u1', 'p1', 'A100', 0)]


def test_run_workers_progress_only_when_verbose(tmp_path, capsys, monkeypatch):
    _patch_popen(monkeypatch, lambda i: _Proc())
    g0 = _two_gpus(tmp_path)[0]
    work = [[Group(shape=8, job_ids=('a',))]]
    run_workers(Config(output_dir=tmp_path, model_dir=tmp_path), [g0], jobs=[], groups=work)
    out = capsys.readouterr().out
    assert 'GPU 0' not in out and 'workers finished' in out
    run_workers(Config(output_dir=tmp_path, model_dir=tmp_path, verbose=True), [g0],
                jobs=[], groups=work)
    out = capsys.readouterr().out
    assert 'GPU 0 (A100)' in out and 'jobs=0/1' in out and 'workers finished' in out


def test_run_workers_exit_code_launch_and_signal_handlers(tmp_path, monkeypatch, capsys):
    gpus = _two_gpus(tmp_path)
    cfg = Config(output_dir=tmp_path, model_dir=tmp_path)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    # a worker killed by SIGKILL (often the OOM killer) must not read as success
    started = _patch_popen(monkeypatch, lambda i: _Proc([0, -9][i]))
    both = [[Group(8, ('a',))], [Group(8, ('b',))]]
    rc, _ = run_workers(cfg, gpus, jobs=[], groups=both)
    assert rc == 137 and 'rc=137' in capsys.readouterr().out
    assert all(kw['start_new_session'] is True for _, kw in started)  # own session
    assert {s: signal.getsignal(s) for s in before} == before  # handlers put back
    # a GPU with nothing assigned does not get a worker (it would only load a model)
    started = _patch_popen(monkeypatch, lambda i: _Proc())
    run_workers(cfg, gpus, jobs=[], groups=[[Group(8, ('a',))], []])
    assert len(started) == 1
    cmd = started[0][0]
    assert cmd[cmd.index('--slot') + 1] == '0'
    # the codes: any failure wins, a signal death maps to 128 + signal
    assert engine.worst_returncode([]) == engine.worst_returncode([0, 0]) == 0
    assert engine.worst_returncode([0, 3]) == 3
    assert engine.worst_returncode([-15, 3]) == 143


def test_forwarded_signal_reaches_each_worker_once(tmp_path, monkeypatch):
    procs = []

    class Interrupted(_Proc):
        def wait(self):  # the signal arrives while the parent waits on the workers
            if not any(p.signals for p in procs):
                signal.raise_signal(signal.SIGTERM)
            return 0

    _patch_popen(monkeypatch, lambda i: procs.append(Interrupted()) or procs[-1])
    run_workers(Config(output_dir=tmp_path, model_dir=tmp_path), _two_gpus(tmp_path),
                jobs=[], groups=[[Group(8, ('a',))], [Group(8, ('b',))]])
    assert [p.signals for p in procs] == [[signal.SIGTERM], [signal.SIGTERM]]


def test_run_with_nothing_to_run(tmp_path, monkeypatch, capsys):
    """No GPU discovery, no workers; rejected inputs are shown and make the run fail."""
    def boom(*a, **k):
        raise AssertionError('must not be reached')

    monkeypatch.setattr(engine, '_gpus_and_benches', boom)
    monkeypatch.setattr(engine, 'run_workers', boom)
    empty, bad = tmp_path / 'empty', tmp_path / 'bad'
    empty.mkdir()
    bad.mkdir()
    (bad / 'broken.json').write_text('{not json')
    assert engine.cmd_run(Config(input_dir=empty, output_dir=tmp_path / 'o1')) == 0
    assert 'nothing to run' in capsys.readouterr().out
    assert engine.cmd_run(Config(input_dir=bad, output_dir=tmp_path / 'o2')) == 1
    text = capsys.readouterr().out
    assert 'reject: broken.json' in text and 'every input was rejected' in text


def test_ledger_and_rejected_files(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _write_job(inp, 'a', 8)
    bad = inp / 'bad.json'
    bad.write_text('{nope')
    cfg = Config(input_dir=inp, output_dir=out)
    write_ledger(cfg, *scan(cfg))
    ledger = (out / '_af3sched' / 'ledger.jsonl').read_text().strip().splitlines()
    assert len(ledger) == 1 and (out / '_af3sched' / 'rejected.txt').is_file()
    bad.unlink()  # last run's rejects must not outlive the problem
    write_ledger(cfg, *scan(cfg))
    assert not (out / '_af3sched' / 'rejected.txt').exists()


def test_report_scan_flags_estimates_and_caps_long_reject_lists(capsys):
    exact, guess = _job('a', 10, 128), _job('b', 10, 128)
    guess = Job(**{**guess.__dict__, 'exact': False})
    report_scan(Config(), [exact, guess], [])
    assert '1 job(s) use an estimated token count' in capsys.readouterr().out
    report_scan(Config(), [exact], [])
    assert 'estimated' not in capsys.readouterr().out
    report_scan(Config(output_dir=Path('out')), [], [f'f{i}.json: bad' for i in range(15)],
                limit=10)
    out = capsys.readouterr().out
    assert out.count('reject: ') == 10 and '5 more in' in out and 'rejected.txt' in out


def test_weights_are_checked_once_in_the_parent(tmp_path, monkeypatch):
    import threading
    import time
    engine._WEIGHTS.clear()
    with pytest.raises(SystemExit, match='model directory not found'):
        engine.ensure_weights(Config(model_dir=tmp_path / 'nope'))
    calls = []

    def slow_resolve(cfg):
        calls.append(cfg.model)
        time.sleep(0.05)
        return tmp_path / 'resolved'

    monkeypatch.setattr('fullFold.runner.resolve_model_dir', slow_resolve)
    cfg = Config(model_dir=tmp_path, model='openfold3')
    got = []
    threads = [threading.Thread(target=lambda: got.append(engine.ensure_weights(cfg)))
               for _ in range(8)]  # one per GPU would otherwise download in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and set(got) == {tmp_path / 'resolved'}
    engine._WEIGHTS.clear()
