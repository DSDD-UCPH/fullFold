"""One-GPU worker: per-seed featurise/infer pipeline and streamed outputs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import io
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fullFold.config import Config, config_from_dict, jax_cache_dir
from fullFold.jobs import Job, end_claim, try_begin_claim

_LOG_LOCK = threading.Lock()


class _Bg:
    """One in-flight extract+write, or inline when disabled."""

    def __init__(self, on: bool):
        self._ex = ThreadPoolExecutor(max_workers=1) if on else None
        self._fut = None

    def submit(self, fn) -> None:
        if self._ex is None:
            fn()
            return
        if self._fut is not None:
            self._fut.result()
        self._fut = self._ex.submit(fn)

    def drain(self) -> None:
        if self._fut is not None:
            self._fut.result()
            self._fut = None

    def close(self) -> None:
        self.drain()
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None


def log_event(fp, **kw) -> None:
    kw.setdefault('ts', time.time())
    line = json.dumps(kw) + '\n'
    with _LOG_LOCK:
        fp.write(line)
        fp.flush()


def padded_len(batch: dict) -> int:
    for k in ('aatype', 'token_index', 'residue_index'):
        v = batch.get(k)
        if v is not None:
            return int(v.shape[0] if hasattr(v, 'shape') else len(v))
    return -1


def _kw(cfg: Config, job: Job) -> dict:
    return dict(
        output_dir=cfg.output_dir / job.job_id, job_name=job.job_id,
        compress=cfg.compress_large_output_files,
        save_terms=cfg.save_terms_of_use)


def _mkdir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_seed_outputs(*, seed, inference_results, embeddings, distogram,
                       output_dir, job_name, compress, save_terms, rank):
    from alphafold3.model import post_processing
    import numpy as np
    output_dir = _mkdir(output_dir)
    for i, result in enumerate(inference_results):
        post_processing.write_output(
            inference_result=result, output_dir=_mkdir(output_dir / f'seed-{seed}_sample-{i}'),
            name=f'{job_name}_seed-{seed}_sample-{i}',
            compress=compress, keep_license=save_terms)
        score = float(result.metadata['ranking_score'])
        rank['scores'].append((seed, i, score))
        if rank['best_score'] is None or score > rank['best_score']:
            rank['best_score'], rank['best_result'] = score, result
    if embeddings:
        post_processing.write_embeddings(
            embeddings=embeddings, output_dir=_mkdir(output_dir / f'seed-{seed}_embeddings'),
            name=f'{job_name}_seed-{seed}')
    if distogram is not None:
        d = _mkdir(output_dir / f'seed-{seed}_distogram')
        with io.BytesIO() as bio:
            np.savez_compressed(bio, distogram=distogram.astype(np.float16))
            (d / f'{job_name}_seed-{seed}_distogram.npz').write_bytes(bio.getvalue())


def write_job_final(*, output_dir, job_name, compress, save_terms, rank):
    from alphafold3.model import post_processing
    from etils import epath
    import alphafold3.cpp
    best = rank.get('best_result')
    if best is None:
        return
    terms = ((epath.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md').read_text()
             if save_terms else None)
    post_processing.write_output(
        inference_result=best, output_dir=output_dir, terms_of_use=terms,
        name=job_name, compress=compress, keep_license=save_terms)
    with (Path(output_dir) / f'{job_name}_ranking_scores.csv').open('w') as f:
        csv.writer(f).writerows([['seed', 'sample', 'ranking_score'], *rank['scores']])


def featurise_one(fold_input, seed: int, buckets: list[int], ccd=None):
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    one = dataclasses.replace(fold_input, rng_seeds=[seed])
    ccd = ccd or chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    return featurisation.featurise_input(
        fold_input=one, ccd=ccd, buckets=buckets, verbose=False)[0]


def _default_feat(job, seed, shape):
    from alphafold3.common import folding_input
    fold = folding_input.Input.from_json(job.path.read_text(), job.path)
    return featurise_one(fold, seed, [shape])


def run_pipeline(items, cfg: Config, *, featurise_fn=None, infer_fn=None,
                 extract_fn=None, write_seed_fn=None, write_final_fn=None,
                 log_fp=None, live=None, primed=None, stats=None, seen=None,
                 bg=None, state=None) -> dict:
    """items: (job, seed, shape). Callables are injectable for CPU tests.

    `primed` is an already-featurised first queue item so job B's first seed can
    be prepared while job A infers. Shared `stats`/`seen` let compile counts
    span jobs.
    """
    featurise_fn = featurise_fn or _default_feat
    infer_fn = infer_fn or (lambda *a: (_infer(),))
    write_seed_fn = write_seed_fn or write_seed_outputs
    write_final_fn = write_final_fn or write_job_final
    bg = bg or _Bg(False)
    prefetch = max(int(cfg.prefetch), 0)
    q: queue.Queue = queue.Queue(maxsize=max(prefetch, 1))
    own_stats = stats is None
    stats = stats or {'compiles': 0, 'jobs_ok': 0, 'jobs_fail': 0, 'max_live': 0}
    seen = seen if seen is not None else set()
    if state is None:
        state = {}

    def fail(st, job, err):
        st['failed'] = True
        stats['jobs_fail'] += 1
        if log_fp:
            log_event(log_fp, event='job_failed', job=job.job_id, error=str(err))

    def _put_ok(job, seed, shape, batch, feat_ms):
        q.put(('ok', job, seed, shape, batch, feat_ms))
        if live is not None:
            live.append(1)
            stats['max_live'] = max(stats['max_live'], len(live))

    def produce():
        try:
            first = True
            for job, seed, shape in items:
                if first and primed is not None:
                    first = False
                    q.put(primed)
                    if live is not None:
                        live.append(1)
                        stats['max_live'] = max(stats['max_live'], len(live))
                    continue
                first = False
                t0 = time.time()
                try:
                    _put_ok(job, seed, shape, featurise_fn(job, seed, shape),
                            (time.time() - t0) * 1000)
                except Exception as e:
                    q.put(('feat_fail', job, seed, shape, e, ''))
        finally:
            q.put(None)

    def handle(item):
        kind, job, seed, shape = item[0], item[1], item[2], item[3]
        st = state.get(job.job_id)
        if st is None:
            st = {'rank': {'scores': [], 'best_score': None, 'best_result': None},
                  'failed': False, 'seeds_done': 0, 'infer_ms': 0.0}
            state[job.job_id] = st
            if log_fp:
                log_event(log_fp, event='job_start', job=job.job_id, shape=shape)
        if st['failed']:
            if kind == 'ok' and live:
                live.pop()
            return
        if shape not in seen:
            seen.add(shape)
            stats['compiles'] += 1
            if log_fp:
                log_event(log_fp, event='group_start', shape=shape)
        if kind == 'feat_fail':
            return fail(st, job, item[4])
        batch, feat_ms = item[4], item[5]
        ntok = padded_len(batch) if isinstance(batch, dict) else -1
        if ntok >= 0 and ntok != shape:
            if live:
                live.pop()
            return fail(st, job, f'token dim {ntok} != shape {shape}')
        t0 = time.time()
        try:
            raw = infer_fn(job, seed, batch)
            inf_ms = (time.time() - t0) * 1000
        except Exception as e:
            if live:
                live.pop()
            return fail(st, job, e)
        if live:
            live.pop()

        def finish():
            if st['failed']:
                return
            try:
                inf, emb, dist = (
                    extract_fn(job, seed, batch, raw) if extract_fn
                    else raw)
                write_seed_fn(
                    seed=seed, inference_results=inf, embeddings=emb,
                    distogram=dist, rank=st['rank'], **_kw(cfg, job))
                st['seeds_done'] += 1
                st['infer_ms'] += inf_ms
                if log_fp:
                    log_event(
                        log_fp, event='seed_done', job=job.job_id, seed=seed,
                        feat_ms=round(feat_ms, 3), infer_ms=round(inf_ms, 3))
                if st['seeds_done'] == job.n_seeds and not st['failed']:
                    write_final_fn(rank=st['rank'], **_kw(cfg, job))
                    stats['jobs_ok'] += 1
                    if log_fp:
                        log_event(
                            log_fp, event='job_done', job=job.job_id,
                            name=job.name, shape=shape, tokens=job.tokens,
                            infer_ms=round(st['infer_ms'], 3),
                            n_seeds=job.n_seeds)
            except Exception as e:
                fail(st, job, e)

        bg.submit(finish)

    if prefetch == 0:
        first = True
        for job, seed, shape in items:
            if first and primed is not None:
                first = False
                handle(primed)
                continue
            first = False
            t0 = time.time()
            try:
                handle(('ok', job, seed, shape, featurise_fn(job, seed, shape),
                        (time.time() - t0) * 1000))
            except Exception as e:
                handle(('feat_fail', job, seed, shape, e, ''))
    else:
        threading.Thread(target=produce, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            handle(item)
    if own_stats:
        stats['shapes'] = sorted(seen)
    return stats


def run_jobs(chunks, cfg: Config, *, featurise_fn=None, infer_fn=None,
             extract_fn=None, write_seed_fn=None, write_final_fn=None,
             log_fp=None, live=None, setup_fn=None) -> dict:
    """chunks: (job, shape). Prefetch>0 prepares job B while job A infers."""
    featurise_fn = featurise_fn or _default_feat
    stats: dict[str, Any] = {
        'compiles': 0, 'jobs_ok': 0, 'jobs_fail': 0, 'max_live': 0}
    seen: set[int] = set()
    chunks = list(chunks)
    bg = _Bg(bool(cfg.background_extract))
    state: dict[str, Any] = {}
    pending: list[tuple[Job, dict]] = []
    pipe = dict(
        featurise_fn=featurise_fn, infer_fn=infer_fn, extract_fn=extract_fn,
        write_seed_fn=write_seed_fn, write_final_fn=write_final_fn,
        log_fp=log_fp, live=live, stats=stats, seen=seen, bg=bg, state=state)

    def _items(job, shape):
        return [(job, int(s), shape) for s in job.seeds]

    def retire() -> None:
        bg.drain()
        while pending:
            job, extra = pending.pop(0)
            st = state.pop(job.job_id, {})
            if st.get('failed'):
                err: BaseException | None = RuntimeError('job failed')
            elif int(st.get('seeds_done') or 0) != job.n_seeds:
                err = RuntimeError('job incomplete')
            else:
                err = None
            end_claim(job, cfg, extra, error=err)

    def _run(job, shape, primed=None) -> BaseException | None:
        if cfg.background_extract:
            run_pipeline(_items(job, shape), cfg, primed=primed, **pipe)
            return None
        fail0, ok0 = stats['jobs_fail'], stats['jobs_ok']
        run_pipeline(_items(job, shape), cfg, primed=primed, **pipe)
        if stats['jobs_fail'] > fail0:
            return RuntimeError('job failed')
        if stats['jobs_ok'] <= ok0:
            return RuntimeError('job incomplete')
        return None

    def _done(job, extra, err):
        if cfg.background_extract:
            pending.append((job, extra))
        else:
            end_claim(job, cfg, extra, error=err)

    def _one(job, shape, primed=None):
        extra = None
        try:
            extra, skip = try_begin_claim(job, cfg)
            if extra is None:
                if log_fp:
                    log_event(log_fp, event='job_skip', job=job.job_id,
                              reason=skip or 'locked', shape=shape)
                return
            if log_fp:
                log_event(log_fp, event='job_prep', job=job.job_id, shape=shape)
            if setup_fn:
                setup_fn(job)
            _done(job, extra, _run(job, shape, primed=primed))
        except Exception as e:
            bg.drain()
            if extra is not None:
                end_claim(job, cfg, extra, error=e)
            elif log_fp:
                log_event(log_fp, event='job_failed', job=job.job_id, error=str(e))
            stats['jobs_fail'] += 1

    try:
        if max(int(cfg.prefetch), 0) == 0:
            for job, shape in chunks:
                if shape not in seen:
                    retire()
                _one(job, shape)
            stats['shapes'] = sorted(seen)
            return stats

        ready: queue.Queue = queue.Queue(maxsize=1)
        start_gate: queue.Queue = queue.Queue()

        def prepare():
            try:
                for i, (job, shape) in enumerate(chunks):
                    extra = None
                    if i:
                        start_gate.get()
                    try:
                        extra, skip = try_begin_claim(job, cfg)
                        if extra is None:
                            if log_fp:
                                log_event(log_fp, event='job_skip',
                                          job=job.job_id, reason=skip or 'locked',
                                          shape=shape)
                            ready.put(('skip', job, None, None, None))
                            continue
                        if log_fp:
                            log_event(log_fp, event='job_prep', job=job.job_id,
                                      shape=shape)
                        if setup_fn:
                            setup_fn(job)
                        t0 = time.time()
                        seed0 = int(job.seeds[0])
                        batch = featurise_fn(job, seed0, shape)
                        primed = ('ok', job, seed0, shape, batch,
                                  (time.time() - t0) * 1000)
                        ready.put(('ready', job, shape, extra, primed))
                    except Exception as e:
                        if extra is not None:
                            end_claim(job, cfg, extra, error=e)
                        if log_fp:
                            log_event(log_fp, event='job_failed', job=job.job_id,
                                      error=str(e))
                        stats['jobs_fail'] += 1
                        ready.put(('skip', job, None, None, None))
            finally:
                ready.put(None)

        th = threading.Thread(target=prepare, daemon=True)
        th.start()
        try:
            while True:
                item = ready.get()
                if item is None:
                    break
                start_gate.put(True)
                kind, job, shape, extra, primed = item
                if kind != 'ready':
                    continue
                if shape not in seen:
                    retire()
                try:
                    _done(job, extra, _run(job, shape, primed=primed))
                except Exception as e:
                    bg.drain()
                    end_claim(job, cfg, extra, error=e)
                    stats['jobs_fail'] += 1
                    while True:
                        nxt = ready.get()
                        if nxt is None:
                            break
                        start_gate.put(True)
                        if nxt[0] == 'ready':
                            end_claim(nxt[1], cfg, nxt[3], error=e)
                    raise
        finally:
            th.join(timeout=5)
        stats['shapes'] = sorted(seen)
        return stats
    finally:
        retire()
        bg.close()


def _infer(*_):
    raise RuntimeError('infer_fn not bound')


def _bind_runner(cfg: Config):
    import jax
    from fullFold.runner import ModelRunner, make_model_config
    cache = os.environ.get('AF3SCHED_JAX_CACHE') or str(jax_cache_dir(
        cfg, physical_id=os.environ.get('CUDA_VISIBLE_DEVICES', '0')))
    Path(cache).mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', cache)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        devices = list(jax.local_devices(backend='gpu'))
    except Exception as e:
        raise RuntimeError(
            f'JAX GPU backend unavailable (CUDA_VISIBLE_DEVICES={visible!r}): {e}'
        ) from e
    if not devices:
        raise RuntimeError(
            f'JAX found no GPU devices (CUDA_VISIBLE_DEVICES={visible!r})'
        )
    runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=cfg.flash_attention,
            num_diffusion_samples=cfg.num_diffusion_samples,
            num_recycles=cfg.num_recycles,
            return_embeddings=cfg.save_embeddings,
            return_distogram=cfg.save_distogram),
        device=devices[0], model_dir=cfg.model_dir)
    _ = runner.model_params

    def infer(job, seed, batch):
        result = runner.run_inference(batch, jax.random.PRNGKey(seed))
        return jax.block_until_ready(result)

    def extract(job, seed, batch, raw):
        inf = runner.extract_inference_results(batch, raw, job.name)
        ntok = len(inf[0].metadata['token_chain_ids'])
        emb = runner.extract_embeddings(raw, ntok) if cfg.save_embeddings else None
        dist = runner.extract_distogram(raw, ntok) if cfg.save_distogram else None
        return inf, emb, dist
    return infer, extract


def run_probe(gpu_physical_id: str, cfg: Config) -> list[float]:
    """4-seed 1024-token probe; returns t1..t4 inference ms."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_physical_id)
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    from alphafold3.common import folding_input
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    from fullFold.benchmark import build_probe_json
    infer, extract = _bind_runner(cfg)
    fold = folding_input.Input.from_json(
        json.dumps(build_probe_json(1024, cfg.bench_seed)))
    ccd = chemical_components.Ccd(user_ccd=fold.user_ccd)
    times = []
    for seed in fold.rng_seeds:
        batch = featurisation.featurise_input(
            fold_input=dataclasses.replace(fold, rng_seeds=[seed]),
            ccd=ccd, buckets=[1024], verbose=False)[0]
        t0 = time.time()
        job = type('J', (), {'name': fold.name})()
        extract(job, seed, batch, infer(job, seed, batch))
        times.append((time.time() - t0) * 1000)
    return times


def _write_input_json(job: Job, cfg: Config) -> None:
    from alphafold3.common import folding_input
    from fullFold.runner import write_fold_input_json
    write_fold_input_json(
        folding_input.Input.from_json(job.path.read_text(), job.path),
        cfg.output_dir / job.job_id)


@contextlib.contextmanager
def gpu_stdio(path: Path):
    """Send this process's stdout/stderr (including C-level compile warnings) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = path.open('a', buffering=1)
    prev_out, prev_err = sys.stdout, sys.stderr
    fd_out, fd_err = os.dup(1), os.dup(2)
    try:
        os.dup2(fp.fileno(), 1)
        os.dup2(fp.fileno(), 2)
        sys.stdout = fp
        sys.stderr = fp
        yield fp
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(fd_out, 1)
        os.dup2(fd_err, 2)
        os.close(fd_out)
        os.close(fd_err)
        sys.stdout = prev_out
        sys.stderr = prev_err
        fp.close()


def worker_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--slot', type=int, default=0)
    p.add_argument('--prefetch', type=int, default=1)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--gpu', default='')
    p.add_argument('--flash-attention', default='')
    args = p.parse_args(argv)
    # AF3 imports absl. First logging.info() parses sys.argv; --probe/--gpu
    # are unknown to absl and would abort the process with exit 1.
    sys.argv = [sys.argv[0]]
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    kw = dict(output_dir=args.output_dir, model_dir=args.model_dir,
              prefetch=args.prefetch)
    if args.flash_attention:
        kw['flash_attention'] = args.flash_attention
    cfg = Config(**kw)
    if args.probe:
        print(json.dumps(run_probe(
            args.gpu or os.environ.get('CUDA_VISIBLE_DEVICES', '0'), cfg)),
              flush=True)
        return 0
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('config'):
        d = dict(manifest['config'])
        d.update(output_dir=str(args.output_dir), model_dir=str(args.model_dir),
                 prefetch=args.prefetch)
        cfg = config_from_dict(d)
    log_dir = cfg.output_dir / '_af3sched'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'gpu{args.slot}.jsonl'
    with gpu_stdio(log_dir / f'gpu{args.slot}.log'):
        infer, extract = _bind_runner(cfg)
        chunks = []
        for g in manifest['groups']:
            shape = int(g['shape'])
            for spec in g['jobs']:
                job = Job(
                    path=Path(spec['path']), name=spec['name'],
                    sha256=spec['sha256'], tokens=int(spec['tokens']),
                    bucket=int(spec.get('bucket', shape)),
                    seeds=tuple(spec['seeds']), exact=True)
                chunks.append((job, shape))

        def setup(job):
            # Redirect only this short write. Do not wrap featurise/infer: a
            # process-wide stdout lock would serialise them and kill overlap.
            af3log = job.state_dir(cfg) / 'af3.log'
            af3log.parent.mkdir(parents=True, exist_ok=True)
            with af3log.open('a') as lf:
                with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
                    _write_input_json(job, cfg)

        with log_path.open('a') as fp:
            run_jobs(chunks, cfg, infer_fn=infer, extract_fn=extract,
                     log_fp=fp, setup_fn=setup)
    return 0


if __name__ == '__main__':
    raise SystemExit(worker_main())
