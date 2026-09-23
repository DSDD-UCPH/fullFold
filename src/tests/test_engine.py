"""Dry-run planning with stubbed GPUs and benches."""

from __future__ import annotations

import io
import json
from pathlib import Path

from fullFold.benchmark import Bench, Gpu
from fullFold.config import Config
from fullFold.engine import (
    consume_worker_events, format_gpu_progress, make_plan, progress_lines,
    render_progress, report, run_workers, summarise_run, write_ledger,
    write_manifests,
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


def test_plan_compiles_equal_distinct_shapes(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _write_job(inp, 's', 10)
    _write_job(inp, 'm', 200)
    _write_job(inp, 'l', 300)
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, _ = scan(cfg)
    benches = _benches()
    groups, meta = make_plan(cfg, jobs, benches)
    write_manifests(cfg, jobs, [g for g, _ in benches], groups, meta)
    for gpu_groups, n in zip(groups, meta['compiles_per_gpu']):
        primary = [g for g in gpu_groups if not g.shared]
        shapes = [g.shape for g in primary]
        assert n == len(shapes) == len(set(shapes))
        assert shapes == sorted(shapes)
    p1 = (out / '_af3sched' / 'plan.json').read_bytes()
    write_manifests(cfg, jobs, [g for g, _ in benches], groups, meta)
    p2 = (out / '_af3sched' / 'plan.json').read_bytes()
    assert p1 == p2
    man = json.loads((out / '_af3sched' / 'manifest-gpu0.json').read_text())
    assert man['config']['prefetch'] == 1
    assert 'groups' in man
    assert 'shared' in man['groups'][0]


def test_floor_warning_meta(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _write_job(inp, 'tiny', 8, seeds=(1,))
    _write_job(inp, 'huge', 8, seeds=tuple(range(400)))
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, _ = scan(cfg)
    groups, meta = make_plan(cfg, jobs, _benches())
    assert meta['floor_job'] == 'huge'
    assert meta['floor_us'] >= meta['makespan_us'] or meta['floor_us'] > 0
    assert meta['floor_us'] == meta['makespan_us']


def test_roundrobin_on_contaminated(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _write_job(inp, 'a', 8)
    _write_job(inp, 'b', 8)
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, _ = scan(cfg)
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    # t3 vs t4 disagree: S unusable
    bad = Bench((500, 400, 100, 200), 150.0, 600.0, True, 'h', 'k')
    groups, meta = make_plan(cfg, jobs, [(g0, bad)])
    assert meta['policy'] == 'roundrobin'
    assert meta['contaminated']


def test_inverted_r_keeps_contiguous(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _write_job(inp, 'a', 8)
    _write_job(inp, 'b', 8)
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, _ = scan(cfg)
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'L40S', 0)
    warm = Bench((10, 10, 100, 100), 100.0, -180.0, False, 'h', 'k')
    _, meta = make_plan(cfg, jobs, [(g0, warm), (g1, warm)])
    assert meta['policy'] == 'contiguous'
    assert not meta['contaminated']


def test_faster_gpu_gets_more_jobs(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    for i in range(20):
        _write_job(inp, f'j{i}', 64)
    cfg = Config(input_dir=inp, output_dir=out, bucket_mode='ladder')
    jobs, _ = scan(cfg)
    g0 = Gpu(0, '0', 'u0', 'pci0', 'RTX 5090', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'RTX 5080', 0)
    fast = Bench((70_000, 67_000, 45_000, 45_000), 45_000.0, 47_000.0, False, 'h', 'k')
    slow = Bench((110_000, 105_000, 74_000, 74_000), 74_000.0, 67_000.0, False, 'h', 'k')
    groups, meta = make_plan(cfg, jobs, [(g0, fast), (g1, slow)])
    assert meta['policy'] == 'contiguous'
    n0 = sum(len(g.job_ids) for g in groups[0] if not g.shared)
    n1 = sum(len(g.job_ids) for g in groups[1] if not g.shared)
    assert n0 > n1
    assert n0 + n1 == 20
    steal0 = [jid for g in groups[0] if g.shared for jid in g.job_ids]
    steal1 = [jid for g in groups[1] if g.shared for jid in g.job_ids]
    last1 = [jid for g in groups[1] if not g.shared for jid in g.job_ids][-1]
    last0 = [jid for g in groups[0] if not g.shared for jid in g.job_ids][-1]
    if steal0:
        assert steal0[0] == last1
    if steal1:
        assert steal1[0] == last0
    assert meta['gpus'][0]['infer_us_1024'] < meta['gpus'][1]['infer_us_1024']


def test_report_prints_per_gpu_bucket_order(capsys):
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'L40S', 0)
    jobs = [
        Job(path=Path('a'), name='a', sha256='x', tokens=10, bucket=128,
            seeds=(1,), exact=True),
        Job(path=Path('b'), name='b', sha256='x', tokens=20, bucket=128,
            seeds=(1, 2), exact=True),
        Job(path=Path('c'), name='c', sha256='x', tokens=400, bucket=512,
            seeds=(1,), exact=True),
    ]
    groups = [
        [Group(shape=128, job_ids=('a', 'b')), Group(shape=512, job_ids=('c',)),
         Group(shape=512, job_ids=('c',), shared=True)],
        [],
    ]
    inf1024 = 59_423_000
    meta = {
        'policy': 'contiguous', 'makespan_us': inf1024, 'floor_us': 0,
        'floor_job': '',
        'gpus': [
            {'slot': 0, 'infer_us_1024': inf1024, 'compile_us': 0, 'm_gpu': 1.0},
            {'slot': 1, 'infer_us_1024': inf1024, 'compile_us': 0, 'm_gpu': 1.23},
        ],
    }
    report(groups, meta, [g0, g1], jobs)
    out = capsys.readouterr().out
    assert 'estimated total running time=' in out
    assert 'GPU 0 (A100): 3 jobs, m_gpu=1.0' in out
    assert '1. bucket 128: 2 jobs, ' in out
    assert '/job, total ' in out
    assert '2. bucket 512: 1 jobs, ' in out
    assert 'a: 1 seed' not in out
    assert 'bucket totals:' not in out
    assert out.index('bucket 128') < out.index('bucket 512')
    assert 'GPU 1 (L40S): 0 jobs, m_gpu=1.23' in out
    assert '(no buckets)' in out
    assert 'steal tail: 1 jobs from other GPUs' in out


def test_summarise_run_compares_actual_to_estimate(tmp_path: Path, capsys):
    cfg = Config(output_dir=tmp_path)
    summarise_run(cfg, {'makespan_us': 100_000_000, 'compiles_per_gpu': []}, 120.0)
    out = capsys.readouterr().out
    assert 'estimated running time: 100.0s' in out
    assert 'actual running time:    120.0s' in out
    assert 'actual / estimated:     1.20x' in out


def test_format_gpu_progress_includes_bar_bucket_and_counts():
    line = format_gpu_progress(
        0, 'A100', 24, 40, bucket=256, tokens=198, job_idx=3, bucket_n=8)
    assert line == (
        'GPU 0 (A100) [======>     ] 24/40  bucket=256  tokens=198  job=3/8')


def test_consume_worker_events_tracks_bucket_progress(tmp_path: Path):
    root = tmp_path / '_af3sched'
    root.mkdir()
    path = root / 'gpu0.jsonl'
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    jobs = [
        Job(path=Path('a'), name='a', sha256='x', tokens=10, bucket=128,
            seeds=(1, 2), exact=True),
        Job(path=Path('c'), name='c', sha256='x', tokens=20, bucket=128,
            seeds=(1,), exact=True),
        Job(path=Path('b'), name='b', sha256='x', tokens=200, bucket=256,
            seeds=(1,), exact=True),
    ]
    groups = [[
        Group(shape=128, job_ids=('a', 'c')),
        Group(shape=256, job_ids=('b',)),
    ]]
    state: dict = {}
    path.write_text('\n'.join([
        json.dumps({'event': 'job_start', 'job': 'a', 'shape': 128}),
        json.dumps({
            'event': 'job_done', 'job': 'a', 'name': 'a', 'shape': 128,
            'tokens': 10, 'infer_ms': 8100, 'n_seeds': 2,
        }),
    ]) + '\n')
    assert consume_worker_events(root, [g0], jobs, state, groups)
    line = progress_lines([g0], state)[0]
    assert line == format_gpu_progress(
        0, 'A100', 1, 3, bucket=128, tokens=10, job_idx=1, bucket_n=2)
    with path.open('a') as f:
        f.write(json.dumps({
            'event': 'job_done', 'job': 'b', 'name': 'b', 'shape': 256,
            'tokens': 200, 'infer_ms': 15000, 'n_seeds': 1,
        }) + '\n')
    assert consume_worker_events(root, [g0], jobs, state, groups)
    line = progress_lines([g0], state)[0]
    assert line == format_gpu_progress(
        0, 'A100', 2, 3, bucket=256, tokens=200, job_idx=1, bucket_n=1)
    assert consume_worker_events(root, [g0], jobs, state, groups) is False


def test_render_progress_snapshot_when_not_tty():
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'L40S', 0)
    state = {
        'by_slot': {
            0: {
                'done': {'a'}, 'started': {},
                'current': {
                    'shape': 256, 'tokens': 198, 'job_idx': 3, 'bucket_n': 8,
                },
                'total': 40,
            },
            1: {
                'done': set(), 'started': {}, 'current': None, 'total': 35,
            },
        },
    }
    buf = io.StringIO()
    render_progress([g0, g1], state, tty=False, out=buf)
    out = buf.getvalue()
    assert 'GPU 0 (A100)' in out
    assert '1/40' in out
    assert 'bucket=256' in out
    assert 'tokens=198' in out
    assert 'job=3/8' in out
    assert 'GPU 1 (L40S)' in out
    assert '0/35' in out
    assert '\033[' not in out
    render_progress([g0, g1], state, tty=False, out=buf)
    assert buf.getvalue().count('GPU 0 (A100)') == 2


def test_render_progress_rewrites_in_place_on_tty():
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    state = {
        'by_slot': {
            0: {'done': set(), 'started': {}, 'current': None, 'total': 4},
        },
    }

    class TtyBuf(io.StringIO):
        def isatty(self):
            return True

    buf = TtyBuf()
    render_progress([g0], state, tty=True, out=buf)
    assert state['painted']
    assert '\033[' not in buf.getvalue()
    state['by_slot'][0]['done'] = {'a'}
    render_progress([g0], state, tty=True, out=buf)
    text = buf.getvalue()
    assert '\033[1A' in text
    assert '\033[K' in text
    assert '1/4' in text


def test_run_workers_progress_only_when_verbose(tmp_path, capsys, monkeypatch):
    class FakeProc:
        def poll(self):
            return 0

        def wait(self):
            return 0

        def send_signal(self, _s):
            pass

    monkeypatch.setattr(
        'fullFold.engine.subprocess.Popen', lambda *a, **k: FakeProc())
    monkeypatch.setattr(
        'fullFold.engine.worker_environ', lambda *a, **k: {})
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    root = tmp_path / '_af3sched'
    root.mkdir()
    (root / 'manifest-gpu0.json').write_text('{}\n')
    quiet = Config(output_dir=tmp_path, model_dir=tmp_path, verbose=False)
    run_workers(quiet, [g0], jobs=[], groups=[[]])
    out = capsys.readouterr().out
    assert 'GPU 0' not in out
    assert 'workers finished' in out
    loud = Config(output_dir=tmp_path, model_dir=tmp_path, verbose=True)
    run_workers(loud, [g0], jobs=[], groups=[[]])
    out = capsys.readouterr().out
    assert 'GPU 0 (A100)' in out
    assert '0/0' in out
    assert 'workers finished' in out


def test_cli_verbose_flag(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cfg):
        seen['verbose'] = cfg.verbose
        return 0

    monkeypatch.setattr('fullFold.engine.cmd_run', fake_run)
    from fullFold.cli import main
    inp, out = tmp_path / 'in', tmp_path / 'out'
    base = ['run', '--input-dir', str(inp), '--output-dir', str(out)]
    assert main(base) == 0
    assert seen['verbose'] is False
    assert main(base + ['--verbose']) == 0
    assert seen['verbose'] is True
    assert main(base + ['-v']) == 0
    assert seen['verbose'] is True


def test_cli_disable_prefetch_and_background_extract(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cfg):
        seen['prefetch'] = cfg.prefetch
        seen['background_extract'] = cfg.background_extract
        seen['bucket_mode'] = cfg.bucket_mode
        return 0

    monkeypatch.setattr('fullFold.engine.cmd_run', fake_run)
    from fullFold.cli import main
    inp, out = tmp_path / 'in', tmp_path / 'out'
    base = ['run', '--input-dir', str(inp), '--output-dir', str(out)]
    assert main(base) == 0
    assert seen['prefetch'] == 1
    assert seen['background_extract'] is True
    assert seen['bucket_mode'] == 'free'
    assert main(base + ['--no-prefetch', '--no-background-extract',
                        '--bucket-mode', 'ladder']) == 0
    assert seen['prefetch'] == 0
    assert seen['background_extract'] is False
    assert seen['bucket_mode'] == 'ladder'


def test_ledger_written(tmp_path: Path):
    inp, out = tmp_path / 'in', tmp_path / 'out'
    _write_job(inp, 'a', 8)
    cfg = Config(input_dir=inp, output_dir=out)
    jobs, rej = scan(cfg)
    write_ledger(cfg, jobs, rej)
    lines = (out / '_af3sched' / 'ledger.jsonl').read_text().strip().splitlines()
    assert len(lines) == 1
