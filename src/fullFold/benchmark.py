"""GPU discovery and 1024-token compile/throughput probe."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from fullFold.config import Config, host_id, worker_environ, write_atomic
from fullFold.scheduling import compilation_modifier, inference_modifier

T_REF_S = 59.423  # A100 tokamax inference-only seconds at bucket 1024


@dataclass(frozen=True)
class Gpu:
    slot: int
    physical_id: str
    uuid: str
    pci_bus_id: str
    device_kind: str
    memory_bytes: int


@dataclass(frozen=True)
class Bench:
    t_ms: tuple[float, float, float, float]
    s_ms: float
    r_ms: float
    contaminated: bool
    host: str
    gpu_key: str


def gpu_multiplier(bench: Bench) -> float:
    """m_gpu = S_1024 / T_REF from the probe's inference-only time."""
    s_s = bench.s_ms / 1000.0
    if T_REF_S <= 0 or s_s <= 0:
        return 1.0
    return s_s / T_REF_S


def predict_inference_s(m_gpu: float, bucket: int) -> float:
    """T_predicted = m_gpu * inference_modifier(bucket) * T_REF, in seconds."""
    return m_gpu * inference_modifier(bucket) * T_REF_S


def compile_overhead_ms(bench: Bench, shape: int) -> float:
    """Probe R at 1024 scaled by the CSV compile modifier (last-row above 5216)."""
    return bench.r_ms * compilation_modifier(shape)


def summarise_timings(t_ms: tuple[float, ...] | list[float]) -> tuple[float, float, bool]:
    """Tokamax pays compile on seeds 1 and 2; seeds 3–4 are steady-state inference.

    ``contaminated`` means S is unusable (t3 vs t4 disagree). Negative R is a
    warm cache or inverted compile, not a reason to ignore throughput.
    """
    t = tuple(float(x) for x in t_ms)
    s = (t[2] + t[3]) / 2.0
    r = t[0] + t[1] - 2.0 * s
    contaminated = s > 0 and abs(t[2] - t[3]) / s > 0.25
    return s, r, contaminated


def nvidia_smi() -> list[dict]:
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,uuid,pci.bus_id,name,memory.total',
        '--format=csv,noheader,nounits',
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            continue
        try:
            mem = int(float(parts[4])) * 1024 * 1024
        except ValueError:
            mem = 0
        rows.append({
            'index': parts[0], 'uuid': parts[1], 'pci': parts[2],
            'name': parts[3], 'memory_bytes': mem,
        })
    rows.sort(key=lambda r: r['pci'])
    return rows


def discover_gpus(cfg: Config, rows: list[dict] | None = None) -> list[Gpu]:
    rows = list(rows if rows is not None else nvidia_smi())
    env = os.environ.get('CUDA_VISIBLE_DEVICES')
    wanted = list(cfg.gpus) if cfg.gpus else (
        [x.strip() for x in env.split(',') if x.strip()]
        if env not in (None, '') else []
    )
    if wanted:
        selected = []
        for w in wanted:
            hit = next((r for r in rows if r['index'] == w or r['uuid'] == w), None)
            if hit is None and rows:
                raise ValueError(f'GPU {w!r} is not in nvidia-smi output')
            if hit:
                selected.append(hit)
        rows = selected
    gpus = [
        Gpu(slot=i, physical_id=r['index'], uuid=r['uuid'],
            pci_bus_id=r['pci'], device_kind=r['name'],
            memory_bytes=int(r['memory_bytes']))
        for i, r in enumerate(rows)
    ]
    if not gpus:
        raise RuntimeError('no GPUs discovered (nvidia-smi empty and no --gpus)')
    return gpus


def gpu_key(gpu: Gpu, jax_version: str = '', af3_version: str = '') -> str:
    return f'{gpu.pci_bus_id}|{gpu.device_kind}|{jax_version}|{af3_version}'


def cache_path(cfg: Config, gpu: Gpu, jax_version: str = '', af3_version: str = '') -> Path:
    return cfg.cache_dir / f'{host_id()}__{gpu_key(gpu, jax_version, af3_version)}.json'


def load_bench(path: Path) -> Bench | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    t = tuple(d['t_ms'])
    s, r, cont = summarise_timings(t)
    return Bench(t_ms=t, s_ms=s, r_ms=r, contaminated=cont,
                 host=d.get('host', ''), gpu_key=d.get('gpu_key', ''))


def save_bench(path: Path, bench: Bench) -> None:
    write_atomic(path, json.dumps(asdict(bench), indent=2))


def build_probe_json(n_tokens: int = 1024, rng_seed: int = 42) -> dict:
    import random
    aa = list('ACDEFGHIKLMNPQRSTVWY')
    rng = random.Random(rng_seed)
    seq = ''.join(rng.choices(aa, k=n_tokens))
    msa = [f'>query\n{seq}\n']
    for i in range(31):
        msa.append(f'>seq_{i+1}\n' + ''.join(rng.choices(aa, k=n_tokens)) + '\n')
    return {
        'name': f'fullFold_bench_{n_tokens}',
        'modelSeeds': [0, 1, 2, 3],
        'sequences': [{
            'protein': {
                'id': 'A', 'sequence': seq,
                'unpairedMsa': ''.join(msa), 'pairedMsa': '', 'templates': [],
            }
        }],
        'dialect': 'alphafold3', 'version': 3,
    }


def to_scheduler_gpu(bench: Bench) -> tuple[int, int]:
    """(compile_us, infer_us_1024) for scheduling.py.

    ``m_gpu = S_1024 / T_REF`` from the probe; inference at bucket b is
    ``m_gpu * f(b/1024) * T_REF``.
    """
    compile_us = max(int(round(compile_overhead_ms(bench, 1024) * 1000)), 0)
    infer_us_1024 = max(1, int(round(gpu_multiplier(bench) * T_REF_S * 1e6)))
    return compile_us, infer_us_1024


def measure(gpu: Gpu, cfg: Config) -> Bench:
    """Spawn a per-GPU probe subprocess so the parent never initialises JAX."""
    import sys
    env = worker_environ(
        cfg, gpu.physical_id, pci_bus_id=gpu.pci_bus_id, device_kind=gpu.device_kind)
    cmd = [
        sys.executable, '-m', 'fullFold.worker', '--probe',
        '--gpu', str(gpu.physical_id),
        '--model-dir', str(cfg.model_dir),
        '--output-dir', str(cfg.output_dir),
        '--flash-attention', cfg.flash_attention,
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or '(no output)').strip()
        raise RuntimeError(
            f'probe on GPU {gpu.physical_id} exited {r.returncode}:\n{detail}'
        )
    t = json.loads(r.stdout.strip().splitlines()[-1])
    s, r_ms, cont = summarise_timings(t)
    if cont:
        warnings.warn(f'contaminated benchmark on GPU {gpu.physical_id}: t={t}')
    b = Bench(t_ms=tuple(t), s_ms=s, r_ms=r_ms, contaminated=cont,
              host=host_id(), gpu_key=gpu_key(gpu))
    save_bench(cache_path(cfg, gpu), b)
    return b


def get_or_measure(
    gpus: list[Gpu], cfg: Config, measure_fn=None,
) -> list[tuple[Gpu, Bench]]:
    measure_fn = measure_fn or measure
    out: list[tuple[Gpu, Bench]] = []
    missing: list[Gpu] = []
    for g in gpus:
        p = cache_path(cfg, g)
        b = None if cfg.force_benchmark else load_bench(p)
        if b is None:
            missing.append(g)
        else:
            out.append((g, b))
    if missing:
        n = len(missing)
        gpu_word = 'GPU' if n == 1 else 'GPUs'
        print(
            f'A few-minute benchmark of the available GPUs is running '
            f'({n} {gpu_word}).',
            file=sys.stderr,
        )
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n) as ex:
            benches = list(ex.map(lambda g: measure_fn(g, cfg), missing))
        out.extend(zip(missing, benches))
        out.sort(key=lambda x: x[0].slot)
    return out
