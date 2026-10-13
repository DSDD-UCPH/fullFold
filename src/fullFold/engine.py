"""Scan / benchmark / plan / run wiring."""

from __future__ import annotations

import dataclasses
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

from fullFold.benchmark import (
    Bench, Gpu, discover_gpus, get_or_measure, gpu_multiplier, is_probed, measure,
    to_scheduler_gpu,
)
from fullFold.config import (
    POLICIES, Config, config_hash, config_to_dict, worker_environ, write_atomic,
)
from fullFold.jobs import Job, scan
from fullFold.planner import plan_multi_gpu_search
from fullFold.scheduling import (
    Group, candidate_shapes, compilation_us, groups_cost_us, inference_us,
    job_floor_us, kernel_tile_shapes, listed_shapes, plan_round_robin, reference_for,
    share_tails,
)


def write_ledger(cfg: Config, jobs: list[Job], rejected: list[str]):
    root = cfg.output_dir / '_af3sched'
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            'name': j.name, 'path': str(j.path), 'sha256': j.sha256,
            'tokens': j.tokens, 'bucket': j.bucket, 'seeds': list(j.seeds),
            'exact': j.exact,
        }, sort_keys=True)
        for j in jobs
    ]
    path = root / 'ledger.jsonl'
    write_atomic(path, '\n'.join(lines) + ('\n' if lines else ''))
    if rejected:
        write_atomic(root / 'rejected.txt', '\n'.join(rejected) + '\n')
    else:
        (root / 'rejected.txt').unlink(missing_ok=True)  # do not leave last run's
    return path


def report_scan(
    cfg: Config, jobs: list[Job], rejected: list[str], *, limit: int | None = None,
) -> None:
    """Say what the scan found, including what it refused and what it only estimated."""
    print(f'scanned {len(jobs)} jobs, rejected {len(rejected)}')
    shown = rejected if limit is None else rejected[:limit]
    for r in shown:
        print(f'  reject: {r}')
    if len(shown) < len(rejected):
        print(f'  ... {len(rejected) - len(shown)} more in '
              f'{cfg.output_dir / "_af3sched" / "rejected.txt"}')
    estimated = sum(1 for j in jobs if not j.exact)
    if estimated:
        print(f'note: {estimated} job(s) use an estimated token count because exact '
              'tokenization was unavailable or failed; an underestimate fails that '
              "job with 'token dim != shape'.")


def write_manifests(
    cfg: Config, jobs: list[Job], gpus: list[Gpu],
    groups: list[list[Group]], meta: dict,
):
    root = cfg.output_dir / '_af3sched'
    root.mkdir(parents=True, exist_ok=True)
    by_id = {j.job_id: j for j in jobs}
    cfg_d = config_to_dict(cfg)
    for gpu, grps in zip(gpus, groups):
        payload = {
            'slot': gpu.slot, 'physical_id': gpu.physical_id, 'config': cfg_d,
            'groups': [{
                'shape': g.shape,
                'shared': g.shared,
                'jobs': [{
                    'name': by_id[jid].name, 'path': str(by_id[jid].path),
                    'sha256': by_id[jid].sha256, 'tokens': by_id[jid].tokens,
                    'bucket': by_id[jid].bucket, 'seeds': list(by_id[jid].seeds),
                } for jid in g.job_ids if jid in by_id],
            } for g in grps],
        }
        write_atomic(
            root / f'manifest-gpu{gpu.slot}.json',
            json.dumps(payload, indent=2, sort_keys=True) + '\n',
        )
    write_atomic(root / 'plan.json', json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return root


def make_plan(
    cfg: Config, jobs: list[Job], gpu_benches: list[tuple[Gpu, Bench]],
) -> tuple[list[list[Group]], dict]:
    jobs = sorted(jobs, key=lambda j: (j.bucket, j.tokens, j.job_id, j.sha256))
    work = [(j.tokens, j.n_seeds, j.job_id) for j in jobs]
    gpus = [g for g, _ in gpu_benches]
    ref = reference_for(cfg.reference)
    if cfg.mode == 'fast' and not cfg.buckets_explicit:
        shapes = kernel_tile_shapes(work)
    elif cfg.mode == 'fast':
        shapes = listed_shapes(work, cfg.buckets)
    else:
        shapes = candidate_shapes(work, cfg.buckets, cfg.bucket_mode, ref)
    contaminated = any(b.contaminated for _, b in gpu_benches)
    if cfg.policy not in POLICIES:
        raise ValueError(
            f'unknown policy {cfg.policy!r}; expected one of {", ".join(POLICIES)} '
            "(the contiguous-block planner was replaced by 'search')")
    policy = 'roundrobin' if (contaminated or cfg.policy == 'roundrobin') else cfg.policy
    split_mode = 'roundrobin'
    planner: dict = {}
    sched_gpus = [to_scheduler_gpu(b) for _, b in gpu_benches]
    if policy == 'roundrobin':
        groups = plan_round_robin(work, len(gpus), shapes, ref)
        loads = [
            groups_cost_us(gr, work, sg, ref) for gr, sg in zip(groups, sched_gpus)]
        makespan = max(loads) if loads else 0
        floor, floor_id = job_floor_us(work, sched_gpus, shapes, ref)
    else:
        makespan, floor, floor_id, groups = plan_multi_gpu_search(
            work, sched_gpus, shapes, ref, effort=cfg.plan_effort, info=planner)
        split_mode = planner.get('stage', 'search')
    groups = share_tails(groups, work, shapes, sched_gpus, ref)
    probed = all(is_probed(b) for _, b in gpu_benches)
    gpu_rows = []
    for (g, b), (cu, inf1024) in zip(gpu_benches, sched_gpus):
        row = {
            'slot': g.slot, 'physical_id': g.physical_id, 'kind': g.device_kind,
            's_ms': b.s_ms, 'r_ms': b.r_ms, 'contaminated': b.contaminated,
            'compile_us': cu, 'infer_us_1024': inf1024,
        }
        if probed:
            row['m_gpu'] = round(gpu_multiplier(b), 4)
        gpu_rows.append(row)
    meta = {
        'config_hash': config_hash(cfg), 'n_jobs': len(jobs), 'policy': policy,
        'bucket_mode': cfg.bucket_mode, 'reference': ref.name,
        'mode': cfg.mode, 'model': cfg.model,
        'hoists': list(cfg.hoists),
        'split_mode': split_mode, 'probed': probed,
        'makespan_us': makespan, 'floor_us': floor, 'floor_job': floor_id,
        'compiles_per_gpu': [
            len([g for g in gr if not g.shared]) for gr in groups],
        'shapes': shapes,
        'contaminated': contaminated, 'gpus': gpu_rows,
    }
    if planner:
        meta['planner'] = {'effort': cfg.plan_effort, **planner}
    return groups, meta


def _fmt_s(us: int) -> str:
    return f'{us / 1e6:.1f}s'


def _fmt_cost(us: int, probed: bool) -> str:
    if probed:
        return _fmt_s(us)
    return f'{us / 1e6:.1f}'


BAR_WIDTH = 12


def _ascii_bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    if total <= 0:
        return '[' + ' ' * width + ']'
    n = int(round(min(max(done, 0), total) / total * width))
    n = max(0, min(width, n))
    if n <= 0:
        return '[' + ' ' * width + ']'
    if n >= width:
        return '[' + '=' * width + ']'
    return '[' + '=' * (n - 1) + '>' + ' ' * (width - n) + ']'


def format_gpu_progress(
    slot: int, kind: str, done: int, total: int,
    bucket: int | None = None, tokens: int | None = None,
    job_idx: int | None = None, bucket_n: int | None = None,
    width: int = BAR_WIDTH,
) -> str:
    gpu = f'GPU {slot} ({kind})' if kind else f'GPU {slot}'
    shown = min(done, total) if total else done
    line = f'{gpu} {_ascii_bar(shown, total, width)} jobs={shown}/{total}'
    if bucket is not None:
        line += f' current_bucket={bucket}'
    if tokens is not None:
        line += f'  job_tokens={tokens}'
    if job_idx is not None and bucket_n is not None:
        line += f'  jobs_in_bucket={job_idx}/{bucket_n}'
    return line


def _group_index(groups: list[Group], job_id: str) -> int | None:
    for i, g in enumerate(groups):
        if job_id in g.job_ids:
            return i
    return None


def _ensure_slot(
    state: dict, gpu: Gpu, groups: list[Group],
) -> dict:
    by_slot = state.setdefault('by_slot', {})
    st = by_slot.get(gpu.slot)
    if st is None:
        st = {
            'done': set(),
            'started': {},
            'current': None,
            'total': sum(len(g.job_ids) for g in groups if not g.shared),
        }
        by_slot[gpu.slot] = st
    return st


def _touch_current(
    st: dict, groups: list[Group], job_id: str,
    shape: int | None, tokens: int | None,
) -> None:
    gi = _group_index(groups, job_id)
    if gi is None:
        started = st['started'].setdefault(-1, [])
        if job_id not in started:
            started.append(job_id)
        job_idx = len(started)
        bucket_n = len(started)
        bucket = shape
    else:
        started = st['started'].setdefault(gi, [])
        if job_id not in started:
            started.append(job_id)
        job_idx = started.index(job_id) + 1
        bucket_n = len(groups[gi].job_ids)
        bucket = shape if shape is not None else groups[gi].shape
    st['current'] = {
        'shape': bucket, 'tokens': tokens,
        'job_idx': job_idx, 'bucket_n': bucket_n,
    }


def progress_lines(gpus: list[Gpu], state: dict) -> list[str]:
    lines = []
    by_slot = state.get('by_slot') or {}
    for gpu in gpus:
        st = by_slot.get(gpu.slot) or {}
        cur = st.get('current') or {}
        lines.append(format_gpu_progress(
            gpu.slot, gpu.device_kind,
            len(st.get('done') or ()),
            int(st.get('total') or 0),
            bucket=cur.get('shape'), tokens=cur.get('tokens'),
            job_idx=cur.get('job_idx'), bucket_n=cur.get('bucket_n'),
        ))
    return lines


def render_progress(
    gpus: list[Gpu], state: dict, *, tty: bool | None = None, out=None,
) -> None:
    out = sys.stdout if out is None else out
    if tty is None:
        tty = bool(hasattr(out, 'isatty') and out.isatty())
    lines = progress_lines(gpus, state)
    if not lines:
        return
    if tty and state.get('painted'):
        out.write(f'\033[{len(lines)}A')
        for line in lines:
            out.write(f'\r{line}\033[K\n')
    else:
        for line in lines:
            out.write(line + '\n')
        if tty:
            state['painted'] = True
    out.flush()


def _jsonl_new_events(path: Path, offset: int) -> tuple[int, list[dict]]:
    if not path.is_file():
        return offset, []
    with path.open('rb') as f:
        f.seek(offset)
        data = f.read()
    if not data:
        return offset, []
    last = data.rfind(b'\n')
    if last < 0:
        return offset, []
    complete = data[:last + 1]
    events = []
    for line in complete.splitlines():
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return offset + len(complete), events


def consume_worker_events(
    root: Path,
    gpus: list[Gpu],
    jobs: list[Job] | None,
    state: dict,
    groups: list[list[Group]] | None = None,
) -> bool:
    """Update per-GPU progress from new ``gpu*.jsonl`` events. Return if dirty."""
    by_id = {j.job_id: j for j in (jobs or [])}
    aligned: list[list[Group]] = list(groups or [])
    while len(aligned) < len(gpus):
        aligned.append([])
    offsets: dict[int, int] = state.setdefault('offsets', {})
    dirty = False
    for gpu, grps in zip(gpus, aligned):
        st = _ensure_slot(state, gpu, grps)
        slot = gpu.slot
        off, events = _jsonl_new_events(
            root / f'gpu{slot}.jsonl', offsets.get(slot, 0))
        offsets[slot] = off
        for ev in events:
            job_id = ev.get('job')
            if not job_id:
                continue
            job_id = str(job_id)
            event = ev.get('event')
            if event not in ('job_start', 'job_done', 'job_skip'):
                continue
            job = by_id.get(job_id)
            tokens = ev.get('tokens')
            if tokens is None and job is not None:
                tokens = job.tokens
            if tokens is not None:
                tokens = int(tokens)
            shape = int(ev['shape']) if ev.get('shape') is not None else None
            if event == 'job_skip':
                continue
            _touch_current(st, grps, job_id, shape, tokens)
            if event == 'job_done':
                st['done'].add(job_id)
            dirty = True
    return dirty


def report(
    groups: list[list[Group]], meta: dict, gpus: list[Gpu],
    jobs: list[Job] | None = None,
) -> None:
    ref = reference_for(str(meta.get('reference') or 'standard'))
    probed = bool(meta.get('probed', True))
    total_label = (
        'estimated total running time=' if probed else 'relative total cost=')
    print(f"policy={meta['policy']} reference={ref.name} "
          f"{total_label}{_fmt_cost(int(meta.get('makespan_us') or 0), probed)}")
    if not probed:
        print(
            'Note: no GPU benchmark for this host and mode; cost is '
            'relative to the timing table (1.0 = one 1024-token inference). '
            'Run fullfold benchmark for time estimates.',
        )
    plan_info = meta.get('planner') or {}
    lower = int(plan_info.get('lower_bound_us') or 0)
    if lower and meta.get('makespan_us'):
        gap = 100.0 * (int(meta['makespan_us']) / lower - 1.0)
        print(f"planner: effort={plan_info.get('effort')} stage={plan_info.get('stage')}; "
              f"no plan can beat {_fmt_cost(lower, probed)}, this one is {gap:.1f}% above "
              f"that bound (loose for few jobs)")
    if (meta.get('floor_us') and meta.get('makespan_us')
            and meta['floor_us'] >= meta['makespan_us'] and len(gpus) > 1):
        print(f"WARNING: job {meta['floor_job']!r} is the makespan floor "
              f"(atomic many-seed job cannot use more than one GPU)")
    by_id = {j.job_id: j for j in (jobs or [])}
    by_slot = {g['slot']: g for g in meta.get('gpus') or []}
    shape_sets = []
    for gpu, grps in zip(gpus, groups):
        primary = [g for g in grps if not g.shared]
        steal = [g for g in grps if g.shared]
        n = sum(len(g.job_ids) for g in primary)
        info = by_slot.get(gpu.slot, {})
        inf1024 = int(info.get('infer_us_1024') or 0)
        compile_us = int(info.get('compile_us') or 0)
        m_gpu = info.get('m_gpu')
        extra = f', m_gpu={m_gpu}' if probed and m_gpu is not None else ''
        print(f"  GPU {gpu.slot} ({gpu.device_kind}): {n} jobs{extra}")
        if not primary:
            print('    (no buckets)')
            shape_sets.append(set())
        else:
            print('    run order:')
            gpu_us = 0
            for i, g in enumerate(primary, 1):
                inf_one = inference_us(inf1024, g.shape, ref) if inf1024 else 0
                n_jobs = len(g.job_ids)
                n_inf = 0
                for jid in g.job_ids:
                    job = by_id.get(jid)
                    n_inf += job.n_seeds if job else 1
                bucket_inf = n_inf * inf_one
                per_job = int(round(bucket_inf / n_jobs)) if n_jobs else 0
                gpu_us += compilation_us(compile_us, g.shape, ref) + bucket_inf
                print(f'      {i}. bucket {g.shape}: {n_jobs} jobs, '
                      f'{_fmt_cost(per_job, probed)}/job, total '
                      f'{_fmt_cost(bucket_inf, probed)}')
            gpu_label = 'estimated GPU time' if probed else 'relative GPU cost'
            print(f'    {gpu_label}: {_fmt_cost(gpu_us, probed)}')
            shape_sets.append({g.shape for g in primary})
        if steal:
            n_steal = sum(len(g.job_ids) for g in steal)
            print(f'    steal tail: {n_steal} jobs from other GPUs '
                  f'(skip if claimed)')
    if len(shape_sets) > 1 and meta.get('policy') != 'search':
        shared = set.intersection(*shape_sets)
        if len(shared) > 1:
            print(f'WARNING: {len(shared)} compiled shapes are duplicated across GPUs')


def event_log_offsets(root: Path, gpus: list[Gpu]) -> dict[int, int]:
    """Current size of each ``gpu*.jsonl``. The logs are appended to across runs,
    so a run reads from here to see only its own events."""
    out = {}
    for gpu in gpus:
        path = root / f'gpu{gpu.slot}.jsonl'
        out[gpu.slot] = path.stat().st_size if path.is_file() else 0
    return out


def summarise_run(
    cfg: Config, meta: dict, wall_s: float, offsets: dict[int, int] | None = None,
) -> None:
    root = cfg.output_dir / '_af3sched'
    probed = bool(meta.get('probed', True))
    if probed:
        pred_s = meta['makespan_us'] / 1e6 if meta.get('makespan_us') else 0
        print(f'estimated running time: {pred_s:.1f}s')
        print(f'actual running time:    {wall_s:.1f}s')
        if pred_s > 0:
            print(f'actual / estimated:     {wall_s / pred_s:.2f}x')
    else:
        print(f'actual running time:    {wall_s:.1f}s')
    offsets = offsets or {}
    for i, exp in enumerate(meta.get('compiles_per_gpu') or []):
        _, events = _jsonl_new_events(root / f'gpu{i}.jsonl', offsets.get(i, 0))
        got = sum(1 for ev in events if ev.get('event') == 'group_start')
        print(f'  GPU {i}: expected compiles={exp} realised={got}')


def cmd_scan(cfg: Config) -> int:
    jobs, rejected = scan(cfg)
    write_ledger(cfg, jobs, rejected)
    report_scan(cfg, jobs, rejected)
    return 0


_WEIGHTS: dict[tuple[str, str], Path] = {}
_WEIGHTS_LOCK = threading.Lock()


def ensure_weights(cfg: Config) -> Path:
    """Resolve ``cfg.model``'s weights once, here, before any worker starts.

    Workers resolve them again, but find them in place: with several GPUs they
    would otherwise all try to download the same file at once, and a missing
    file would only surface after the benchmark. Returns the directory to pass
    to workers (it differs from ``--model-dir`` when a download had to go elsewhere).
    """
    from fullFold.runner import canonical_model, resolve_model_dir
    key = (cfg.model, str(cfg.model_dir))
    with _WEIGHTS_LOCK:
        hit = _WEIGHTS.get(key)
        if hit is None:
            root = Path(cfg.model_dir).expanduser()
            if canonical_model(cfg.model) == 'alphafold3' and not root.is_dir():
                raise SystemExit(f'fullFold: model directory not found: {root}')
            hit = _WEIGHTS[key] = resolve_model_dir(cfg)
    return hit


def _measure_with_weights(gpu: Gpu, cfg: Config) -> Bench:
    return measure(gpu, dataclasses.replace(cfg, model_dir=ensure_weights(cfg)))


def _gpus_and_benches(cfg: Config, *, require: bool = False):
    return get_or_measure(
        discover_gpus(cfg), cfg, measure_fn=_measure_with_weights, require=require)


def cmd_benchmark(cfg: Config) -> int:
    for g, b in _gpus_and_benches(cfg, require=True):
        print(f'GPU {g.slot} {g.device_kind}: S={b.s_ms:.1f}ms R={b.r_ms:.1f}ms '
              f'contaminated={b.contaminated} t={b.t_ms}')
    return 0


class Prepared(NamedTuple):
    jobs: list[Job]
    rejected: list[str]
    gpus: list[Gpu]
    groups: list[list[Group]]
    meta: dict


def _prepare(cfg: Config) -> Prepared:
    jobs, rejected = scan(cfg)
    write_ledger(cfg, jobs, rejected)
    report_scan(cfg, jobs, rejected, limit=10)
    if not jobs:
        print('nothing to run: no pending jobs' + (
            ' (every input was rejected)' if rejected else ''))
        return Prepared(jobs, rejected, [], [], {})
    benches = _gpus_and_benches(cfg)
    gpus = [g for g, _ in benches]
    groups, meta = make_plan(cfg, jobs, benches)
    write_manifests(cfg, jobs, gpus, groups, meta)
    report(groups, meta, gpus, jobs)
    return Prepared(jobs, rejected, gpus, groups, meta)


def _exit_code(prep: Prepared) -> int:
    """Nothing scheduled is success, unless inputs were refused."""
    return 1 if (not prep.jobs and prep.rejected) else 0


def cmd_plan(cfg: Config) -> int:
    return _exit_code(_prepare(cfg))


def worst_returncode(codes: list[int]) -> int:
    """Largest failure among worker exit codes; a worker killed by a signal counts."""
    bad = [128 - c if c < 0 else c for c in codes if c != 0]
    return max(bad, default=0)


_FORWARDED = tuple(
    getattr(signal, name) for name in ('SIGINT', 'SIGTERM', 'SIGHUP')
    if hasattr(signal, name))


def run_workers(
    cfg: Config, gpus: list[Gpu], jobs: list[Job] | None = None,
    groups: list[list[Group]] | None = None,
    offsets: dict[int, int] | None = None,
) -> tuple[int, float]:
    root = cfg.output_dir / '_af3sched'
    procs: list[subprocess.Popen] = []
    logs: list = []
    watch: dict = {'offsets': dict(offsets or {})}
    tty = sys.stdout.isatty()
    previous: dict[int, object] = {}

    def _flush_jobs(*, force: bool = False) -> None:
        dirty = consume_worker_events(
            root, gpus, jobs, watch, groups=groups)
        if dirty or force:
            render_progress(gpus, watch, tty=tty)

    def _fwd(signum, _frame):
        for p in procs:
            p.send_signal(signum)

    try:
        # Workers get their own session, so a terminal's Ctrl-C reaches only this
        # process and is forwarded exactly once (not once by the tty and once by us).
        for signum in _FORWARDED:
            previous[signum] = signal.signal(signum, _fwd)
        for i, gpu in enumerate(gpus):
            if groups is not None and not (i < len(groups) and groups[i]):
                continue  # nothing assigned: do not load a model just to idle
            env = worker_environ(
                cfg, gpu.physical_id, pci_bus_id=gpu.pci_bus_id,
                device_kind=gpu.device_kind, memory_bytes=gpu.memory_bytes)
            log_path = root / f'gpu{gpu.slot}.log'
            lf = log_path.open('a')
            logs.append(lf)
            cmd = [
                sys.executable, '-m', 'fullFold.worker',
                '--manifest', str(root / f'manifest-gpu{gpu.slot}.json'),
                '--output-dir', str(cfg.output_dir),
                '--model-dir', str(cfg.model_dir),
                '--slot', str(gpu.slot), '--prefetch', str(cfg.prefetch),
            ]
            procs.append(subprocess.Popen(
                cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True))

        t0 = time.time()
        if cfg.verbose:
            _flush_jobs(force=True)
            while not all(p.poll() is not None for p in procs):
                _flush_jobs()
                time.sleep(0.2)
            codes = [p.wait() for p in procs]
            _flush_jobs()
        else:
            codes = [p.wait() for p in procs]
        rc = worst_returncode(codes)
        wall = time.time() - t0
        print(f'workers finished in {wall:.1f}s rc={rc}')
        return rc, wall
    except BaseException:
        for p in procs:  # a failure while launching must not leave workers behind
            if p.poll() is None:
                p.terminate()
        raise
    finally:
        for signum, handler in previous.items():
            if handler is not None:
                signal.signal(signum, handler)
        for lf in logs:
            lf.close()


def cmd_run(cfg: Config) -> int:
    prep = _prepare(cfg)
    if not prep.jobs or cfg.dry_run:
        return _exit_code(prep)
    run_cfg = dataclasses.replace(cfg, model_dir=ensure_weights(cfg))
    root = cfg.output_dir / '_af3sched'
    offsets = event_log_offsets(root, prep.gpus)
    rc, wall = run_workers(
        run_cfg, prep.gpus, jobs=prep.jobs, groups=prep.groups, offsets=offsets)
    summarise_run(cfg, prep.meta, wall, offsets)
    return rc
