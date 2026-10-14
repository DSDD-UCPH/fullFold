"""Look-ahead featurisation in worker processes, next to one inference stream per GPU.

A GPU worker featurises with one producer thread (plus one for the next job's first seed), and in the same process as the JAX
dispatch thread, so at most one featurisation runs at a time and it competes for the GIL. When featurising a seed takes longer than
inferring it (stock AF3 v3.0.4: ~6 s against ~4 s for 450 tokens on an RTX 5090) the GPU idles.

``FeaturePool`` knows the worker's whole plan (every (job, seed) in run order) and keeps up to ``ahead`` upcoming items being
featurised in ``--feat-workers`` spawned CPU-only processes. ``featurise(job, seed, shape)`` has the signature of the worker's
``featurise_fn``: it returns the batch of that item, waiting only if it is not ready yet.

  * a child is JAX-CPU-only (``CUDA_VISIBLE_DEVICES=""``, ``JAX_PLATFORMS=cpu``) and calls the same
    ``runner.featurise_fold`` with the same config, so the features are the same;
  * the first ``VERIFY`` items are also featurised in the worker and compared bitwise. A mismatch switches the pool off
    (the in-process result is used) and says so;
  * a failing child, an unknown item or a broken pool falls back to in-process featurisation of that item, so errors surface
    in the job exactly as without the pool;
  * extra patches that a child must apply to match its parent (e.g. a featurisation-side lever of another package) can be named
    in the environment variable ``FULLFOLD_FEAT_INIT`` as comma separated ``module:function`` callables; they are called without
    arguments in every child. The bitwise check above catches a parent/child mismatch.

Items are consumed in plan order; items that the worker skips (claimed elsewhere) are dropped when a later item is consumed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

VERIFY = 2


def _child_init(fast_feat: bool, conf_threads: int) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['JAX_PLATFORMS'] = 'cpu'
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'
    if fast_feat:
        from fullFold import featfast
        featfast.install(conf_threads=conf_threads, quiet=True)
    import importlib
    for spec in filter(None, (s.strip() for s in os.environ.get('FULLFOLD_FEAT_INIT', '').split(','))):
        mod, _, fn = spec.partition(':')
        getattr(importlib.import_module(mod), fn or 'install')()


def _child_feat(job, seed: int, shape: int, cfg):
    from fullFold.worker import _load_fold, featurise_one
    t0 = time.time()
    fold, ccd = _load_fold(job)
    batch = featurise_one(fold, seed, [shape], ccd=ccd, cfg=cfg)
    return batch, time.time() - t0, os.getpid()


def batches_equal(a, b) -> bool:
    """Bitwise equality of two feature batches (nested dicts / lists / arrays / plain objects)."""
    import numpy as np
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(batches_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(batches_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, '__dict__') and not hasattr(a, 'shape'):
        return type(a) is type(b) and batches_equal(vars(a), vars(b))
    x, y = np.asarray(a), np.asarray(b)
    if x.dtype == object:  # e.g. a Structure without __eq__: compare the type only
        return y.dtype == object and x.shape == y.shape and (
            type(x.item()) is type(y.item()) if x.shape == () else True)
    return x.shape == y.shape and x.dtype == y.dtype and bool(np.array_equal(x, y))


class FeaturePool:
    def __init__(self, chunks, cfg, *, stock_featurise, workers: int, ahead: int | None = None,
                 verify: int = VERIFY, log_path: Path | None = None):
        """chunks: the worker's (job, shape) list in run order; stock_featurise(job, seed, shape) is the in-process path."""
        self.cfg = cfg
        self.stock = stock_featurise
        self.n = max(int(workers), 1)
        self.ahead = int(ahead) if ahead else 2 * self.n + 2
        self.verify_left = int(verify)
        self.log_path = log_path
        self.plan = [(job, int(seed), int(shape)) for job, shape in chunks for seed in job.seeds]
        self.margin = self.ahead + max((len(job.seeds) for job, _ in chunks), default=1) + 4
        self.pos = {(job.job_id, seed): i for i, (job, seed, _) in enumerate(self.plan)}
        self.lock = threading.Lock()
        self.futs: dict[int, object] = {}
        self.nxt = 0
        self.off = False
        self.pool = None
        self.stats = dict(items=0, wait_s=0.0, child_s=0.0, fallbacks=0, verified=0)

    # -- bookkeeping ----------------------------------------------------------------------------------------------------
    def _log(self, **kw) -> None:
        if self.log_path is None:
            return
        try:
            with self.log_path.open('a') as f:
                f.write(json.dumps(kw) + '\n')
        except OSError:
            pass

    def _ensure_pool(self):
        if self.pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            threads = max(1, min(8, (os.cpu_count() or 2) // (self.n + 1)))
            self.pool = ProcessPoolExecutor(
                max_workers=self.n, mp_context=mp.get_context('spawn'), initializer=_child_init,
                initargs=(bool(getattr(self.cfg, 'fast_featurisation', True)), threads))
        return self.pool

    def _window(self, i: int) -> int:
        # very large complexes: smaller look-ahead (their feature batches are big)
        return self.n if self.plan[i][2] > 2048 else self.ahead

    def _submit_upto(self, i: int) -> None:
        pool = self._ensure_pool()
        hi = min(i + self._window(i), len(self.plan) - 1)
        j = max(self.nxt, i)
        while j <= hi:
            if j not in self.futs:
                job, seed, shape = self.plan[j]
                self.futs[j] = pool.submit(_child_feat, job, seed, shape, self.cfg)
            j += 1
        self.nxt = max(self.nxt, j)

    def _drop_before(self, i: int) -> None:
        # The worker consumes in plan order except that the next job's first seed is featurised while the current job still has
        # seeds to go, so an item can be asked for up to one job (max seeds) after a later one. Only items further back than
        # that belong to skipped jobs and are released.
        for j in [j for j in self.futs if j < i - self.margin]:
            self.futs.pop(j).cancel()

    def _disable(self, why: str) -> None:
        self.off = True
        self._log(event='disabled', why=why)
        print(f'fullFold: feature pool disabled ({why}); featurising in this process', flush=True)
        self.close()

    # -- the worker's featurise_fn --------------------------------------------------------------------------------------
    def featurise(self, job, seed: int, shape: int):
        i = self.pos.get((job.job_id, int(seed)))
        if i is None or self.off or self.plan[i][2] != int(shape):
            self.stats['fallbacks'] += 1
            return self.stock(job, seed, shape)
        t0 = time.time()
        try:
            with self.lock:
                if self.off:
                    raise RuntimeError('off')
                self._submit_upto(i)
                fut = self.futs.pop(i)
                self._drop_before(i)
            batch, child_s, pid = fut.result()
        except Exception as e:
            self.stats['fallbacks'] += 1
            if not self.off:
                self._log(event='fallback', job=job.job_id, seed=int(seed), error=repr(e)[:300])
            return self.stock(job, seed, shape)
        wait_s = time.time() - t0
        with self.lock:
            check = self.verify_left > 0
            if check:
                self.verify_left -= 1
        rec = dict(event='feat', job=job.job_id, seed=int(seed), shape=int(shape), child_s=round(child_s, 3),
                   wait_s=round(wait_s, 3), pid=pid, queued=len(self.futs))
        self.stats['items'] += 1
        self.stats['wait_s'] += wait_s
        self.stats['child_s'] += child_s
        if check:
            ref = self.stock(job, seed, shape)
            same = batches_equal(ref, batch)
            self.stats['verified'] += 1
            rec['verify_bitwise'] = bool(same)
            self._log(**rec)
            if not same:
                self._disable(f'child features differ from in-process for {job.job_id} seed {seed}')
                return ref
            return batch
        self._log(**rec)
        return batch

    def close(self) -> None:
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def summary(self) -> str:
        s = self.stats
        mean = s['child_s'] / s['items'] if s['items'] else 0.0
        return (f"feature pool: {self.n} workers, {s['items']} items, waited {s['wait_s']:.1f} s in total, "
                f"mean {mean:.2f} s per item in the workers, {s['fallbacks']} in-process fallbacks, "
                f"{s['verified']} checked bitwise against in-process")
