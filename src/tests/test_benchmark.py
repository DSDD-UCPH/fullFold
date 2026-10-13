"""GPU discovery, benchmark arithmetic and the probe cache (no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fullFold import benchmark
from fullFold.benchmark import (
    COST_UNIT_US, Bench, Gpu, T_REF_S, UNPROBED_KEY, cache_path, compile_overhead_ms,
    discover_gpus, get_or_measure, gpu_multiplier, is_probed, load_bench, measure,
    predict_inference_s, save_bench, summarise_timings, to_scheduler_gpu, unprobed_bench,
)
from fullFold.config import Config, host_id, jax_cache_dir, worker_environ
from fullFold.scheduling import compilation_modifier

ROWS = [
    {'index': '1', 'uuid': 'GPU-bbb', 'pci': '0000:02:00.0', 'name': 'L40S', 'memory_bytes': 1},
    {'index': '0', 'uuid': 'GPU-aaa', 'pci': '0000:01:00.0', 'name': 'A100', 'memory_bytes': 1},
]
A100 = Gpu(0, '0', 'u', 'pci', 'A100', 0)
GOOD = Bench((1000, 200, 180, 220), 200, 400, False, 'h', 'k')


def test_summarise_timings():
    s, r, cont = summarise_timings((1000, 200, 180, 220))
    assert (s, r, cont) == (200, 1000 + 200 - 2 * 200, False)  # compile on seeds 1 and 2
    # inverted R is a warm or odd cache; S is still usable
    _, r, cont = summarise_timings((10, 10, 100, 100))
    assert r < 0 and not cont
    # tokamax compiles on both of the first two seeds; t3 and t4 are steady state
    t = (70269.0, 67573.5, 44942.2, 44978.7)
    s, r, cont = summarise_timings(t)
    assert s == pytest.approx((44942.2 + 44978.7) / 2)
    assert r == pytest.approx(t[0] + t[1] - 2 * s) and r > 0 and not cont
    # steady-state seeds that disagree make S unusable
    assert summarise_timings((500, 400, 100, 200))[2]


def test_discover_gpus(monkeypatch):
    by_pci = sorted(ROWS, key=lambda r: r['pci'])
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    gpus = discover_gpus(Config(gpus=()), rows=by_pci)  # unset: all, in PCI order
    assert [g.physical_id for g in gpus] == ['0', '1'] and gpus[0].device_kind == 'A100'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,0')
    assert [g.physical_id for g in discover_gpus(Config(gpus=()), rows=ROWS)] == ['1', '0']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')  # empty means all
    assert len(discover_gpus(Config(gpus=()), rows=by_pci)) == 2
    gpus = discover_gpus(Config(gpus=('GPU-aaa',)), rows=ROWS)  # --gpus wins
    assert len(gpus) == 1 and gpus[0].uuid == 'GPU-aaa'


def test_cached_probe_is_reused(tmp_path: Path, capsys):
    cfg = Config(cache_dir=tmp_path)
    g0, g1 = A100, Gpu(1, '1', 'u1', 'pci1', 'A100', 0)
    save_bench(cache_path(cfg, g0), GOOD)

    def no_measure(g, c):
        raise AssertionError('a cached probe must not be measured again')

    out = get_or_measure([g0, g1], cfg, measure_fn=no_measure)
    assert out[0][1].s_ms == out[1][1].s_ms == 200  # the second card of that name reuses it
    assert out[0][1].r_ms == 1000 + 200 - 2 * 200  # R is recomputed, not trusted
    assert all(is_probed(b) for _, b in out)
    assert 'benchmark of the available GPUs' not in capsys.readouterr().err
    # a stale contaminated flag in the file is ignored too
    t = (70269.0, 67573.5, 44942.2, 44978.7)
    s, r, _ = summarise_timings(t)
    save_bench(cache_path(cfg, g0), Bench(t, s, r, True, 'h', 'k'))
    assert not load_bench(cache_path(cfg, g0)).contaminated


def test_cache_misses_that_must_not_crash_or_leak(tmp_path: Path, monkeypatch):
    cfg = Config(cache_dir=tmp_path)
    path = cache_path(cfg, A100)
    path.parent.mkdir(parents=True, exist_ok=True)
    for text in ('', '{not json', '[]', 'null', '{"nope": 1}', '{"t_ms": [1, 2, 3]}',
                 '{"t_ms": "abc"}', '{"t_ms": [1, 2, "x", 4]}'):
        path.write_text(text)
        assert load_bench(path) is None, text
        assert get_or_measure([A100], cfg)[0][1].gpu_key == UNPROBED_KEY, text
    # a probe belongs to one software stack: upgrading JAX or AlphaFold 3 invalidates it
    monkeypatch.setattr(benchmark, 'runtime_versions', lambda: ('1', 'a'))
    save_bench(cache_path(cfg, A100), GOOD)
    assert get_or_measure([A100], cfg)[0][1].gpu_key != UNPROBED_KEY
    monkeypatch.setattr(benchmark, 'runtime_versions', lambda: ('2', 'a'))
    assert get_or_measure([A100], cfg)[0][1].gpu_key == UNPROBED_KEY
    # a probe made without the fast kernels is not used for fast mode
    off = Config(cache_dir=tmp_path, mode='off')
    save_bench(cache_path(off, A100), GOOD)
    fast = Config(cache_dir=tmp_path, mode='fast')
    assert get_or_measure([A100], fast)[0][1].gpu_key == UNPROBED_KEY
    # unit-cost stand-ins are never written
    assert not cache_path(fast, A100).exists()


def test_when_gpus_are_probed(tmp_path: Path, capsys):
    cfg = Config(cache_dir=tmp_path)
    g0 = Gpu(0, '0', 'u0', 'pci0', 'A100', 0)
    g1 = Gpu(1, '1', 'u1', 'pci1', 'A100', 0)
    l0 = Gpu(2, '2', 'u2', 'pci2', 'L40S', 0)
    calls = []

    def fake(g, c):
        calls.append(g.slot)
        return Bench((10, 10, 10, 10), 10, 0, False, 'h', g.device_kind)

    # one kind and no cache: no probe, unit costs, and no benchmark notice
    out = get_or_measure([g0, g1], cfg, measure_fn=fake)
    assert calls == [] and all(b.gpu_key == UNPROBED_KEY for _, b in out)
    assert capsys.readouterr().err == ''
    # two kinds: one probe per kind, shared by the cards of that kind
    out = get_or_measure([g0, g1, l0], cfg, measure_fn=fake)
    assert sorted(calls) == [0, 2]
    assert out[0][1] is out[1][1] and out[2][1].gpu_key == 'L40S'
    assert 'A few-minute benchmark of the available GPUs is running (2 GPUs).' in (
        capsys.readouterr().err)
    # `benchmark` (require) probes even a single kind
    calls.clear()
    get_or_measure([g0], cfg, measure_fn=fake, require=True)
    assert calls == [0]
    assert '(1 GPU).' in capsys.readouterr().err
    # --force-benchmark ignores a cached probe
    save_bench(cache_path(cfg, g0), GOOD)
    forced = Bench((9, 9, 9, 9), 9, 0, True, 'h', 'k')
    out = get_or_measure([g0], Config(cache_dir=tmp_path, force_benchmark=True),
                         measure_fn=lambda g, c: forced)
    assert out[0][1].s_ms == 9


def test_probe_cache_key(tmp_path: Path, monkeypatch):
    gpu = A100
    path = {m: cache_path(Config(cache_dir=tmp_path, mode=m), gpu)
            for m in ('off', 'fast', 'default')}
    assert len(set(path.values())) == 3
    for mode, p in path.items():
        assert f'|{mode}|' in p.name
    assert '|h=none' in path['off'].name
    hoisted = cache_path(Config(cache_dir=tmp_path, mode='default', hoists=(
        'cond_share', 'atom_cond_hoist', 'diffusion_hoist')), gpu)
    assert hoisted != path['default']
    assert '|h=cond_share+atom_cond_hoist+diffusion_hoist' in hoisted.name
    # disabled fast-mode levers split the cache, but only in fast mode
    fast = {k: cache_path(Config(cache_dir=tmp_path, mode='fast', **kw), gpu)
            for k, kw in (('base', {}), ('nosb', {'sampler_bf16': False}),
                          ('nohl', {'hoist_logits': False}))}
    assert len(set(fast.values())) == 3
    assert 'h=no-sampler-bf16' in fast['nosb'].name and 'h=no-hoist-logits' in fast['nohl'].name
    assert cache_path(Config(cache_dir=tmp_path, sampler_bf16=False), gpu) == path['default']
    # the software versions are part of the key
    monkeypatch.setattr(benchmark, 'runtime_versions', lambda: ('0.7.0+0.7.0', '3.0.4'))
    base = cache_path(Config(cache_dir=tmp_path), gpu)
    assert '|0.7.0+0.7.0|3.0.4|default|' in base.name
    for versions in (('0.8.1+0.8.1', '3.0.4'), ('0.7.0+0.7.0', '3.0.5')):
        monkeypatch.setattr(benchmark, 'runtime_versions', lambda v=versions: v)
        assert cache_path(Config(cache_dir=tmp_path), gpu) != base
    # ...and are read from package metadata, so the parent never imports JAX
    monkeypatch.undo()
    benchmark._dist_version.cache_clear()
    table = {'jax': '0.7.0', 'jaxlib': '0.7.1', 'alphafold3': '3.0.4', 'af3-faster': '0.2'}

    def version(name):
        if name in table:
            return table[name]
        raise benchmark.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(benchmark.metadata, 'version', version)
    assert benchmark.runtime_versions() == ('0.7.0+0.7.1', '3.0.4+faster0.2')
    benchmark._dist_version.cache_clear()
    monkeypatch.setattr(benchmark.metadata, 'version', lambda n: version('missing'))
    assert benchmark.runtime_versions() == ('', '')
    benchmark._dist_version.cache_clear()


def test_scheduler_costs_from_a_probe():
    b = Bench((1000, 200, 200, 200), 200.0, 400.0, False, 'h', 'k')
    assert to_scheduler_gpu(b) == (400_000, 200_000)  # S = 200 ms is 0.2 s at 1024
    assert gpu_multiplier(b) == pytest.approx(0.2 / T_REF_S)
    assert compile_overhead_ms(b, 1024) == pytest.approx(400.0)
    assert compile_overhead_ms(b, 256) == pytest.approx(400.0 * compilation_modifier(256))
    assert predict_inference_s(1.0, 1024) == pytest.approx(T_REF_S)
    m = gpu_multiplier(b)
    assert predict_inference_s(m, 1024) == pytest.approx(0.2)
    assert predict_inference_s(m, 2048) > 2 * predict_inference_s(m, 1024)
    assert to_scheduler_gpu(unprobed_bench()) == (COST_UNIT_US, COST_UNIT_US)
    assert not is_probed(unprobed_bench())


class _Run:
    returncode = 0
    stderr = ''

    def __init__(self, t=None, returncode=0, stderr=''):
        self.stdout = json.dumps(t) + '\n' if t else ''
        self.returncode, self.stderr = returncode, stderr


def test_measure_runs_a_probe_subprocess_and_caches_only_clean_results(
        monkeypatch, tmp_path: Path):
    cfg = Config(cache_dir=tmp_path, model_dir=tmp_path, output_dir=tmp_path,
                 flash_attention='xla')
    captured = {}

    def fake_run(cmd, **kw):
        captured.update(cmd=cmd, env=kw.get('env'))
        return _Run(stderr='UnrecognizedFlagError: Unknown command line flag probe\n',
                    returncode=1)

    monkeypatch.setattr('fullFold.benchmark.subprocess.run', fake_run)
    with pytest.raises(RuntimeError, match='Unknown command line flag probe'):
        measure(A100, cfg)  # the worker's stderr is surfaced
    assert '--flash-attention' in captured['cmd'] and 'xla' in captured['cmd']
    env = captured['env']
    assert env['CUDA_VISIBLE_DEVICES'] == '0' and env['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert env['AF3SCHED_JAX_CACHE'] == str(
        jax_cache_dir(cfg, physical_id='0', pci_bus_id='pci', device_kind='A100'))
    # a clean probe is cached
    monkeypatch.setattr('fullFold.benchmark.subprocess.run',
                        lambda cmd, **kw: _Run([5000.0, 4000.0, 1000.0, 1010.0]))
    b = measure(A100, cfg)
    assert not b.contaminated
    assert load_bench(cache_path(cfg, A100)).s_ms == pytest.approx(b.s_ms)
    cache_path(cfg, A100).unlink()
    # a noisy one is used for this run only, so it cannot pin the GPU to round-robin
    monkeypatch.setattr('fullFold.benchmark.subprocess.run',
                        lambda cmd, **kw: _Run([5000.0, 4000.0, 1000.0, 3000.0]))
    with pytest.warns(UserWarning, match='not cached'):
        assert measure(A100, cfg).contaminated
    assert not cache_path(cfg, A100).exists()


def test_jax_cache_per_host_and_gpu_name(tmp_path: Path):
    cfg = Config(cache_dir=tmp_path)
    hid = host_id()

    def cache(kind, pci='0000:01:00.0'):
        return jax_cache_dir(cfg, physical_id='0', pci_bus_id=pci, device_kind=kind)

    a0, a1 = cache('A100'), cache('A100', '0000:02:00.0')
    assert a0 == a1 and a0.name == f'{hid}__A100' and a0.parent == tmp_path / 'jax'
    assert cache('L40S') != a0 and cache('L40S').name == f'{hid}__L40S'
    assert cache('A100-SXM4-40GB') != cache('A100-SXM4-80GB')
    env0 = worker_environ(cfg, '0', pci_bus_id='0000:01:00.0', device_kind='A100')
    env1 = worker_environ(cfg, '1', pci_bus_id='0000:02:00.0', device_kind='A100')
    assert env0['AF3SCHED_JAX_CACHE'] == str(a0) == env1['AF3SCHED_JAX_CACHE']
    assert a0.is_dir()
    assert jax_cache_dir(cfg, physical_id='0', pci_bus_id='pci').name == f'{hid}__gpu'
    over = jax_cache_dir(
        Config(cache_dir=tmp_path, jax_compilation_cache_dir=tmp_path / 'xla'),
        physical_id='0', pci_bus_id='0000:01:00.0', device_kind='A100')
    assert over.parent == tmp_path / 'xla' and over.name == f'{hid}__A100'
