"""Scan / benchmark / plan / run wiring."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from fullFold.benchmark import (
    Bench, Gpu, discover_gpus, get_or_measure, gpu_multiplier, to_scheduler_gpu,
)
from fullFold.config import (
    Config, config_hash, config_to_dict, worker_environ, write_atomic,
)
from fullFold.jobs import Job, scan
from fullFold.scheduling import (
    Group, candidate_shapes, compilation_us, groups_cost_us, inference_us,
    job_floor_us, plan_multi_gpu, plan_round_robin, share_tails,
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
    return path


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
    shapes = candidate_shapes(work, cfg.buckets, cfg.bucket_mode)
    contaminated = any(b.contaminated for _, b in gpu_benches)
    policy = 'roundrobin' if (contaminated or cfg.policy == 'roundrobin') else cfg.policy
    split_mode = 'exact' if len(work) <= cfg.exact_split_threshold else 'coarsened'
    sched_gpus = [to_scheduler_gpu(b) for _, b in gpu_benches]
    if policy == 'roundrobin':
        groups = plan_round_robin(work, len(gpus), shapes)
        loads = [groups_cost_us(gr, work, sg) for gr, sg in zip(groups, sched_gpus)]
        makespan = max(loads) if loads else 0
        floor, floor_id = job_floor_us(work, sched_gpus, shapes)
    else:
        makespan, floor, floor_id, groups = plan_multi_gpu(
            work, sched_gpus, shapes, cfg.exact_split_threshold)
    groups = share_tails(groups, work, shapes, sched_gpus)
    gpu_rows = []
    for (g, b), (cu, inf1024) in zip(gpu_benches, sched_gpus):
        gpu_rows.append({
            'slot': g.slot, 'physical_id': g.physical_id, 'kind': g.device_kind,
            's_ms': b.s_ms, 'r_ms': b.r_ms, 'contaminated': b.contaminated,
            'compile_us': cu, 'infer_us_1024': inf1024,
            'm_gpu': round(gpu_multiplier(b), 4),
        })
    meta = {
        'config_hash': config_hash(cfg), 'n_jobs': len(jobs), 'policy': policy,
        'bucket_mode': cfg.bucket_mode, 'split_mode': split_mode,
        'makespan_us': makespan, 'floor_us': floor, 'floor_job': floor_id,
        'compiles_per_gpu': [
            len([g for g in gr if not g.shared]) for gr in groups],
        'shapes': shapes,
        'contaminated': contaminated, 'gpus': gpu_rows,
    }
    return groups, meta


def _fmt_s(us: int) -> str:
    return f'{us / 1e6:.1f}s'


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
    line = f'{gpu} {_ascii_bar(shown, total, width)} {shown}/{total}'
    if bucket is not None:
        line += f'  bucket={bucket}'
    if tokens is not None:
        line += f'  tokens={tokens}'
    if job_idx is not None and bucket_n is not None:
        line += f'  job={job_idx}/{bucket_n}'
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
    print(f"policy={meta['policy']} estimated total running time="
          f"{_fmt_s(int(meta.get('makespan_us') or 0))}")
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
        extra = f', m_gpu={m_gpu}' if m_gpu is not None else ''
        print(f"  GPU {gpu.slot} ({gpu.device_kind}): {n} jobs{extra}")
        if not primary:
            print('    (no buckets)')
            shape_sets.append(set())
        else:
            print('    run order:')
            gpu_us = 0
            for i, g in enumerate(primary, 1):
                inf_one = inference_us(inf1024, g.shape) if inf1024 else 0
                n_jobs = len(g.job_ids)
                n_inf = 0
                for jid in g.job_ids:
                    job = by_id.get(jid)
                    n_inf += job.n_seeds if job else 1
                bucket_inf = n_inf * inf_one
                per_job = int(round(bucket_inf / n_jobs)) if n_jobs else 0
                gpu_us += compilation_us(compile_us, g.shape) + bucket_inf
                print(f'      {i}. bucket {g.shape}: {n_jobs} jobs, '
                      f'{_fmt_s(per_job)}/job, total {_fmt_s(bucket_inf)}')
            print(f'    estimated GPU time: {_fmt_s(gpu_us)}')
            shape_sets.append({g.shape for g in primary})
        if steal:
            n_steal = sum(len(g.job_ids) for g in steal)
            print(f'    steal tail: {n_steal} jobs from other GPUs '
                  f'(skip if claimed)')
    if len(shape_sets) > 1:
        shared = set.intersection(*shape_sets)
        if len(shared) > 1:
            print(f'WARNING: {len(shared)} compiled shapes are duplicated across GPUs')


def summarise_run(cfg: Config, meta: dict, wall_s: float) -> None:
    root = cfg.output_dir / '_af3sched'
    pred_s = meta['makespan_us'] / 1e6 if meta.get('makespan_us') else 0
    print(f'estimated running time: {pred_s:.1f}s')
    print(f'actual running time:    {wall_s:.1f}s')
    if pred_s > 0:
        print(f'actual / estimated:     {wall_s / pred_s:.2f}x')
    for i, exp in enumerate(meta.get('compiles_per_gpu') or []):
        path = root / f'gpu{i}.jsonl'
        got = 0
        if path.is_file():
            for line in path.read_text().splitlines():
                if line and json.loads(line).get('event') == 'group_start':
                    got += 1
        print(f'  GPU {i}: expected compiles={exp} realised={got}')


def cmd_scan(cfg: Config) -> int:
    jobs, rejected = scan(cfg)
    write_ledger(cfg, jobs, rejected)
    print(f'scanned {len(jobs)} jobs, rejected {len(rejected)}')
    for r in rejected:
        print(f'  reject: {r}')
    return 0


def _gpus_and_benches(cfg: Config):
    return get_or_measure(discover_gpus(cfg), cfg)


def cmd_benchmark(cfg: Config) -> int:
    for g, b in _gpus_and_benches(cfg):
        print(f'GPU {g.slot} {g.device_kind}: S={b.s_ms:.1f}ms R={b.r_ms:.1f}ms '
              f'contaminated={b.contaminated} t={b.t_ms}')
    return 0


def _prepare(cfg: Config):
    jobs, rejected = scan(cfg)
    write_ledger(cfg, jobs, rejected)
    benches = _gpus_and_benches(cfg)
    gpus = [g for g, _ in benches]
    groups, meta = make_plan(cfg, jobs, benches)
    write_manifests(cfg, jobs, gpus, groups, meta)
    report(groups, meta, gpus, jobs)
    return jobs, gpus, groups, meta


def cmd_plan(cfg: Config) -> int:
    _prepare(cfg)
    return 0


def run_workers(
    cfg: Config, gpus: list[Gpu], jobs: list[Job] | None = None,
    groups: list[list[Group]] | None = None,
) -> tuple[int, float]:
    root = cfg.output_dir / '_af3sched'
    procs: list[subprocess.Popen] = []
    logs: list = []
    watch: dict = {}
    tty = sys.stdout.isatty()

    def _flush_jobs(*, force: bool = False) -> None:
        dirty = consume_worker_events(
            root, gpus, jobs, watch, groups=groups)
        if dirty or force:
            render_progress(gpus, watch, tty=tty)

    try:
        for gpu in gpus:
            env = worker_environ(
                cfg, gpu.physical_id, pci_bus_id=gpu.pci_bus_id,
                device_kind=gpu.device_kind)
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
                cmd, env=env, stdout=lf, stderr=subprocess.STDOUT))

        def _fwd(signum, _frame):
            for p in procs:
                p.send_signal(signum)

        signal.signal(signal.SIGINT, _fwd)
        signal.signal(signal.SIGTERM, _fwd)
        t0 = time.time()
        rc = 0
        if cfg.verbose:
            _flush_jobs(force=True)
            while True:
                _flush_jobs()
                if all(p.poll() is not None for p in procs):
                    break
                time.sleep(0.2)
            for p in procs:
                rc = max(rc, p.wait())
            _flush_jobs()
        else:
            for p in procs:
                rc = max(rc, p.wait())
        wall = time.time() - t0
        print(f'workers finished in {wall:.1f}s rc={rc}')
        return rc, wall
    finally:
        for lf in logs:
            lf.close()


def cmd_run(cfg: Config) -> int:
    jobs, gpus, groups, meta = _prepare(cfg)
    if cfg.dry_run:
        return 0
    rc, wall = run_workers(cfg, gpus, jobs=jobs, groups=groups)
    summarise_run(cfg, meta, wall)
    return rc
