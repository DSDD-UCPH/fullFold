"""Single frozen Config plus hashing helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
from pathlib import Path

DEFAULT_BUCKETS = (
    128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096,
    4608, 5120,
)

# af3-faster FAST_ENV. Set in the worker process before JAX imports. Explicit
# lever overrides are applied after setdefault so --no-sampler-bf16 sticks.
FAST_ENV = {
    'AF3_FLASHPAIRFORMER': 'both',
    'AF3_DIFFUSION_HOIST': '1',
    'AF3_JAX_DATTN': '1',
    'AF3_JAX_TTR': 'fast',
    'AF3_JAX_TRIATT_XLA': 'fast',
    'AF3_JAX_SAMPLER_BF16': '1',
    'AF3_JAX_ATOM_ATTN': '1',
    'AF3_JAX_TRIMUL_CD': 'fast',
    'AF3_JAX_LNP': 'fast',
    'AF3_JAX_HOIST_LOGITS': '1',
    'AF3_JAX_COND_SHARE': '1',
    'AF3_JAX_ATOM_COND_HOIST': '1',
}
XLA_TRITON_GEMM_OFF = '--xla_gpu_enable_triton_gemm=false'


@dataclasses.dataclass(frozen=True)
class Config:
    input_dir: Path = Path('.')
    output_dir: Path = Path('.')
    model_dir: Path = Path('~/models').expanduser()
    gpus: tuple[str, ...] = ()
    dry_run: bool = False
    verbose: bool = False
    prefetch: int = 1
    background_extract: bool = True
    policy: str = 'contiguous'  # contiguous | roundrobin
    bucket_mode: str = 'free'  # free | ladder
    reference: str = 'standard'  # standard | fast
    buckets: tuple[int, ...] = DEFAULT_BUCKETS
    bucket_margin: float = 0.05
    stale_lock_seconds: int = 900
    retry_failed: bool = False
    exact_split_threshold: int = 512
    force_benchmark: bool = False
    bench_seed: int = 42
    cache_dir: Path = Path('~/.cache/fullFold/bench').expanduser()
    jax_compilation_cache_dir: Path | None = None
    xla_mem_fraction: float = 0.97
    xla_preallocate: bool = True
    exact_tokens: bool = False
    template: Path | None = None
    records: Path | None = None
    record_type: str = 'protein'
    save_embeddings: bool = False
    save_distogram: bool = False
    compress_large_output_files: bool = False
    save_terms_of_use: bool = True
    num_recycles: int = 10
    num_diffusion_samples: int = 5
    flash_attention: str = 'triton'
    json_path: Path | None = None
    mode: str = 'off'  # off | fast (af3-faster kernels; off unless --mode fast)
    model: str = 'alphafold3'
    buckets_explicit: bool = False
    num_seeds: int | None = None
    sampler_bf16: bool | None = None
    hoist_logits: bool | None = None
    download_weights: bool = True
    weights_precision: str = 'int8'
    conformer_max_iterations: int | None = None
    resolve_msa_overlaps: bool = True
    max_template_date: str = '2021-09-30'
    fix_standalone_glycans: bool = False
    force_output_dir: bool = False


def host_id() -> str:
    p = Path('/etc/machine-id')
    if p.is_file():
        return p.read_text().strip()
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def sanitise(name: str) -> str:
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
    return ''.join(c for c in name.replace(' ', '_') if c in allowed)


def config_hash(cfg: Config) -> str:
    return sha256_bytes(json.dumps(config_to_dict(cfg), sort_keys=True).encode())[:16]


def config_to_dict(cfg: Config) -> dict:
    d = dataclasses.asdict(cfg)
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, tuple):
            d[k] = list(v)
    return d


def config_from_dict(d: dict) -> Config:
    fields = Config.__dataclass_fields__
    kw = {}
    for k, v in d.items():
        if k not in fields:
            continue
        hint = str(fields[k].type)
        if v is None:
            kw[k] = None
        elif 'Path' in hint and not isinstance(v, Path):
            kw[k] = Path(v)
        elif 'tuple' in hint and not isinstance(v, tuple):
            kw[k] = tuple(v)
        else:
            kw[k] = v
    return Config(**kw)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def jax_cache_dir(
    cfg: Config, *, physical_id: str, pci_bus_id: str = '', device_kind: str = '',
) -> Path:
    """Persistent XLA cache root for this host + GPU name. Always enabled.

    PCI / physical_id are accepted for call-site compatibility but are not
    part of the path: identical nvidia-smi names share a directory. JAX's
    file keys (HLO, jaxlib, GPU-name topology) still miss across models.
    """
    del physical_id, pci_bus_id
    root = Path(cfg.jax_compilation_cache_dir) if cfg.jax_compilation_cache_dir else (
        cfg.cache_dir / 'jax')
    kind_part = sanitise(device_kind) if device_kind else 'gpu'
    return root / f'{host_id()}__{kind_part}'


def worker_environ(
    cfg: Config, gpu_physical_id: str, *,
    pci_bus_id: str = '', device_kind: str = '',
) -> dict[str, str]:
    """Env for a one-GPU worker/probe subprocess (parent never initialises JAX)."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_physical_id)
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['XLA_CLIENT_MEM_FRACTION'] = str(cfg.xla_mem_fraction)
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true' if cfg.xla_preallocate else 'false'
    cache = jax_cache_dir(
        cfg, physical_id=gpu_physical_id, pci_bus_id=pci_bus_id,
        device_kind=device_kind)
    cache.mkdir(parents=True, exist_ok=True)
    env['AF3SCHED_JAX_CACHE'] = str(cache)
    if cfg.mode == 'fast':
        if not (env.get('XLA_FLAGS') or '').strip():
            env['XLA_FLAGS'] = XLA_TRITON_GEMM_OFF
        for key, value in FAST_ENV.items():
            env.setdefault(key, value)
        if cfg.sampler_bf16 is not None:
            env['AF3_JAX_SAMPLER_BF16'] = '1' if cfg.sampler_bf16 else '0'
        if cfg.hoist_logits is not None:
            env['AF3_JAX_HOIST_LOGITS'] = '1' if cfg.hoist_logits else '0'
    return env
