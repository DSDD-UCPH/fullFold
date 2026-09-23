"""GPU discovery and benchmark arithmetic (no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fullFold.benchmark import (
    Bench, Gpu, T_REF_S, cache_path, discover_gpus, get_or_measure, gpu_multiplier,
    load_bench, measure, predict_inference_s, save_bench, summarise_timings,
    to_scheduler_gpu, compile_overhead_ms,
)
from fullFold.config import Config, host_id, jax_cache_dir, worker_environ


ROWS = [
    {'index': '1', 'uuid': 'GPU-bbb', 'pci': '0000:02:00.0', 'name': 'L40S', 'memory_bytes': 1},
    {'index': '0', 'uuid': 'GPU-aaa', 'pci': '0000:01:00.0', 'name': 'A100', 'memory_bytes': 1},
]


def test_summarise_timings():
    s, r, cont = summarise_timings((1000, 200, 180, 220))
    assert s == 200
    assert r == 1000 + 200 - 2 * 200  # compile on seeds 1 and 2
    assert not cont


def test_contaminated_negative_r():
    s, r, cont = summarise_timings((10, 10, 100, 100))
    assert r < 0 and not cont  # inverted R is a warm/odd cache, S is still usable


def test_tokamax_two_seed_compile_not_contaminated():
    # Both t1 and t2 include tokamax compile; t3/t4 are steady-state.
    t = (70269.0, 67573.5, 44942.2, 44978.7)
    s, r, cont = summarise_timings(t)
    assert s == pytest.approx((44942.2 + 44978.7) / 2)
    assert r == pytest.approx(t[0] + t[1] - 2 * s)
    assert r > 0 and not cont


def test_contaminated_steady_state():
    _, _, cont = summarise_timings((500, 400, 100, 200))
    # s=150, |t3-t4|/s = 100/150 > 0.25
    assert cont


def test_discover_all_pci_order(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    cfg = Config(gpus=())
    gpus = discover_gpus(cfg, rows=sorted(ROWS, key=lambda r: r['pci']))
    assert [g.physical_id for g in gpus] == ['0', '1']
    assert gpus[0].device_kind == 'A100'


def test_discover_cuda_visible_indices(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,0')
    cfg = Config(gpus=())
    gpus = discover_gpus(cfg, rows=ROWS)
    assert [g.physical_id for g in gpus] == ['1', '0']


def test_discover_override_gpus():
    cfg = Config(gpus=('GPU-aaa',))
    gpus = discover_gpus(cfg, rows=ROWS)
    assert len(gpus) == 1 and gpus[0].uuid == 'GPU-aaa'


def test_empty_cuda_visible_means_all(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    cfg = Config(gpus=())
    gpus = discover_gpus(cfg, rows=sorted(ROWS, key=lambda r: r['pci']))
    assert len(gpus) == 2


def test_cache_hit_skips_measure(tmp_path: Path):
    cfg = Config(cache_dir=tmp_path)
    gpu = Gpu(0, '0', 'u', 'pci', 'A100', 0)
    b = Bench((1000, 200, 180, 220), 200, 400, False, 'h', 'k')
    save_bench(cache_path(cfg, gpu), b)
    calls = []

    def fake_measure(g, c):
        calls.append(g)
        return b

    out = get_or_measure([gpu], cfg, measure_fn=fake_measure)
    assert calls == []
    assert out[0][1].s_ms == 200
    assert out[0][1].r_ms == 1000 + 200 - 2 * 200  # recomputed, ignore cached R


def test_load_bench_ignores_stale_contaminated_flag(tmp_path: Path):
    cfg = Config(cache_dir=tmp_path)
    gpu = Gpu(0, '0', 'u', 'pci', '5090', 0)
    t = (70269.0, 67573.5, 44942.2, 44978.7)
    s, r, cont = summarise_timings(t)
    assert not cont
    save_bench(cache_path(cfg, gpu), Bench(t, s, r, True, 'h', 'k'))
    b = load_bench(cache_path(cfg, gpu))
    assert b is not None and not b.contaminated
    assert b.r_ms == pytest.approx(r)


def test_cache_miss_measures(tmp_path: Path, capsys):
    cfg = Config(cache_dir=tmp_path)
    gpu = Gpu(0, '0', 'u', 'pci', 'A100', 0)
    b = Bench((10, 10, 10, 10), 10, 0, True, 'h', 'k')
    out = get_or_measure([gpu], cfg, measure_fn=lambda g, c: b)
    assert out[0][1].contaminated
    err = capsys.readouterr().err
    assert 'A few-minute benchmark of the available GPUs is running (1 GPU).' in err


def test_cache_hit_skips_benchmark_notice(tmp_path: Path, capsys):
    cfg = Config(cache_dir=tmp_path)
    gpu = Gpu(0, '0', 'u', 'pci', 'A100', 0)
    b = Bench((1000, 200, 180, 220), 200, 400, False, 'h', 'k')
    save_bench(cache_path(cfg, gpu), b)
    get_or_measure([gpu], cfg, measure_fn=lambda g, c: b)
    assert 'benchmark of the available GPUs' not in capsys.readouterr().err


def test_to_scheduler_gpu_uses_probe_s():
    b = Bench((1000, 200, 200, 200), 200.0, 400.0, False, 'h', 'k')
    compile_us, inf1024 = to_scheduler_gpu(b)
    assert compile_us == 400_000
    assert inf1024 == 200_000  # S=200ms → 0.2s at 1024
    assert gpu_multiplier(b) == pytest.approx(0.2 / T_REF_S)
    from fullFold.scheduling import compilation_modifier
    assert compile_overhead_ms(b, 1024) == pytest.approx(400.0)
    assert compile_overhead_ms(b, 256) == pytest.approx(
        400.0 * compilation_modifier(256))


def test_predict_inference_scales_from_t_ref():
    assert predict_inference_s(1.0, 1024) == pytest.approx(T_REF_S)
    b = Bench((1000, 200, 200, 200), 200.0, 400.0, False, 'h', 'k')
    m = gpu_multiplier(b)
    assert predict_inference_s(m, 1024) == pytest.approx(0.2)
    assert predict_inference_s(m, 2048) > predict_inference_s(m, 1024) * 2


def test_force_benchmark(tmp_path: Path):
    cfg = Config(cache_dir=tmp_path, force_benchmark=True)
    gpu = Gpu(0, '0', 'u', 'pci', 'A100', 0)
    save_bench(cache_path(cfg, gpu), Bench((1, 1, 1, 1), 1, 0, True, 'h', 'k'))
    fresh = Bench((9, 9, 9, 9), 9, 0, True, 'h', 'k')
    out = get_or_measure([gpu], cfg, measure_fn=lambda g, c: fresh)
    assert out[0][1].s_ms == 9


def test_measure_surfaces_worker_stderr(monkeypatch, tmp_path: Path):
    cfg = Config(cache_dir=tmp_path, model_dir=tmp_path, output_dir=tmp_path,
                 flash_attention='xla')
    gpu = Gpu(0, '0', 'u', 'pci', 'A100', 0)
    captured = {}

    class R:
        returncode = 1
        stdout = ''
        stderr = 'UnrecognizedFlagError: Unknown command line flag probe\n'

    def fake_run(cmd, **kw):
        captured['cmd'] = cmd
        captured['env'] = kw.get('env')
        return R()

    monkeypatch.setattr('fullFold.benchmark.subprocess.run', fake_run)
    with pytest.raises(RuntimeError, match='Unknown command line flag probe'):
        measure(gpu, cfg)
    assert '--flash-attention' in captured['cmd']
    assert 'xla' in captured['cmd']
    assert captured['env']['CUDA_VISIBLE_DEVICES'] == '0'
    assert captured['env']['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert captured['env']['AF3SCHED_JAX_CACHE'] == str(
        jax_cache_dir(cfg, physical_id='0', pci_bus_id='pci', device_kind='A100'))


def test_jax_cache_per_host_and_gpu_name(tmp_path: Path):
    cfg = Config(cache_dir=tmp_path)
    hid = host_id()
    a0 = jax_cache_dir(cfg, physical_id='0', pci_bus_id='0000:01:00.0',
                       device_kind='A100')
    a1 = jax_cache_dir(cfg, physical_id='1', pci_bus_id='0000:02:00.0',
                       device_kind='A100')
    l40 = jax_cache_dir(cfg, physical_id='2', pci_bus_id='0000:03:00.0',
                        device_kind='L40S')
    a40 = jax_cache_dir(cfg, physical_id='0', pci_bus_id='0000:01:00.0',
                        device_kind='A100-SXM4-40GB')
    a80 = jax_cache_dir(cfg, physical_id='1', pci_bus_id='0000:02:00.0',
                        device_kind='A100-SXM4-80GB')
    assert a0 == a1
    assert a0.name == f'{hid}__A100'
    assert a0.parent == tmp_path / 'jax'
    assert l40 != a0 and l40.name == f'{hid}__L40S'
    assert a40 != a80
    env0 = worker_environ(cfg, '0', pci_bus_id='0000:01:00.0', device_kind='A100')
    env1 = worker_environ(cfg, '1', pci_bus_id='0000:02:00.0', device_kind='A100')
    assert env0['AF3SCHED_JAX_CACHE'] == str(a0) == env1['AF3SCHED_JAX_CACHE']
    assert a0.is_dir()
    missing = jax_cache_dir(cfg, physical_id='0', pci_bus_id='pci')
    assert missing.name == f'{hid}__gpu'
    cfg_override = Config(cache_dir=tmp_path, jax_compilation_cache_dir=tmp_path / 'xla')
    over = jax_cache_dir(
        cfg_override, physical_id='0', pci_bus_id='0000:01:00.0', device_kind='A100')
    assert over.parent == tmp_path / 'xla'
    assert over.name == f'{hid}__A100'
